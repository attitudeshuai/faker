import random as random_module
import re
import threading

from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable, List, NamedTuple, Optional, Type, Union

from .typing import SeedType

if TYPE_CHECKING:
    from .providers import BaseProvider

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")
random = random_module.Random()
mod_random = random  # compat with name released in 0.8


Sentinel = object()


class MethodResolution(NamedTuple):
    """Describes how a method name currently resolves.

    :param method_name: name of the faker method
    :param provider: provider currently providing the method, or ``None`` when
        the method is actively shadowed by a caller
    :param providers: all providers that can provide the method, in precedence order
    :param shadowed: whether a caller-installed shadow is currently active
    """

    method_name: str
    provider: Optional["BaseProvider"]
    providers: List["BaseProvider"]
    shadowed: bool


class MethodConflict(NamedTuple):
    """Record of how a same-name collision between providers was resolved.

    :param method_name: name of the faker method
    :param provider: provider that won the resolution
    :param providers: all candidate providers, in precedence order
    :param priorities: mapping of provider name to its declared priority
    """

    method_name: str
    provider: "BaseProvider"
    providers: List["BaseProvider"]
    priorities: Dict[str, int]


class _ShadowRecord(NamedTuple):
    provider: Optional["BaseProvider"]
    function: Callable


class Generator:
    __config: Dict[str, Dict[Hashable, Any]] = {
        "arguments": {},
    }

    _is_seeded = False
    _global_seed = Sentinel

    def __init__(self, **config: Dict) -> None:
        self.providers: List["BaseProvider"] = []
        self.__config = dict(list(self.__config.items()) + list(config.items()))
        self.__random = random

        # Method dispatch resolution layer
        self._method_providers: Dict[str, List["BaseProvider"]] = {}
        self._shadow_stack: Dict[str, List[_ShadowRecord]] = {}
        self._restored_owners: Dict[str, "BaseProvider"] = {}
        self._conflicts: Dict[str, MethodConflict] = {}
        self._conflict_log: List[MethodConflict] = []
        # Bumped after every committed change to providers/shadow state, so
        # downstream consumers (e.g. multi-locale candidate mappings) can tell
        # their cached resolution is stale.
        self._dispatch_version: int = 0
        self._provider_lock = threading.RLock()

    def __getstate__(self) -> Dict[str, Any]:
        # Thread locks cannot be pickled/copied; a fresh one is created on restore.
        state = self.__dict__.copy()
        state["_provider_lock"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._provider_lock = threading.RLock()

    def add_provider(self, provider: Union["BaseProvider", Type["BaseProvider"]]) -> None:
        if isinstance(provider, type):
            provider = provider(self)

        # Runtime-registered providers must carry name metadata, otherwise they
        # could not be found by name.
        if not hasattr(provider, "__provider__"):
            provider.__provider__ = self._synthesize_provider_name(provider)

        # Gather the public methods before mutating any state, so a failure
        # here never leaves a half-registered provider behind.
        methods: List[str] = []
        for method_name in dir(provider):
            # skip 'private' method
            if method_name.startswith("_"):
                continue

            faker_function = getattr(provider, method_name)

            if callable(faker_function):
                methods.append(method_name)

        with self._provider_lock:
            self.providers.insert(0, provider)

            for method_name in methods:
                # Only callable attributes participate in dispatch, mirroring
                # the historical binding semantics (a data attribute sharing
                # the name, e.g. bank's ``country_code`` string, never hides
                # the actual method).
                candidates = [p for p in self.providers if callable(getattr(p, method_name, None))]
                self._method_providers[method_name] = candidates

                # An active caller shadow always stays on top; the new provider
                # only becomes eligible once the shadow is restored.
                if method_name not in self._shadow_stack:
                    winner = self._resolved_winner(method_name, candidates)
                    current = getattr(self, method_name, None)
                    current_provider = getattr(current, "__self__", None)
                    if current is None or current_provider is not winner:
                        # Replacement only: the old method stays callable until
                        # the new one is bound.
                        self.set_formatter(method_name, getattr(winner, method_name))

                    self._record_conflict_if_any(method_name, candidates)

            self._dispatch_version += 1

    @staticmethod
    def _synthesize_provider_name(provider: "BaseProvider") -> str:
        cls = type(provider)
        name = f"{cls.__module__}.{cls.__qualname__}".lower()
        if name.startswith("__main__."):
            name = name[len("__main__.") :]
        return name

    @staticmethod
    def _select_winner(candidates: List["BaseProvider"]) -> "BaseProvider":
        """Resolves the winning provider.

        Higher ``__priority__`` wins; ties are broken by candidate order,
        which matches ``self.providers`` order (front = most recently added),
        reproducing the historical "last registered wins" behavior.
        """
        winner = candidates[0]
        winner_priority = getattr(winner, "__priority__", 0)
        for candidate in candidates[1:]:
            priority = getattr(candidate, "__priority__", 0)
            if priority > winner_priority:
                winner = candidate
                winner_priority = priority
        return winner

    def _resolved_winner(
        self,
        method_name: str,
        candidates: List["BaseProvider"],
    ) -> "BaseProvider":
        """Resolves the winner honoring a restoration decree, if one exists."""
        restored = self._restored_owners.get(method_name)
        if restored is not None:
            return restored
        return self._select_winner(candidates)

    @staticmethod
    def _underlying_function(provider: "BaseProvider", method_name: str) -> Any:
        """Returns the function object actually implementing the method.

        Methods merely inherited from a shared base (same function object) do
        not constitute a real collision.
        """
        for cls in type(provider).__mro__:
            if method_name in cls.__dict__:
                return cls.__dict__[method_name]
        return None

    def _record_conflict_if_any(
        self,
        method_name: str,
        candidates: List["BaseProvider"],
    ) -> None:
        implementations = {self._underlying_function(p, method_name) for p in candidates}
        if len(implementations) <= 1:
            return

        record = MethodConflict(
            method_name=method_name,
            provider=self._resolved_winner(method_name, candidates),
            providers=list(candidates),
            priorities={getattr(p, "__provider__", "base"): getattr(p, "__priority__", 0) for p in candidates},
        )
        self._conflicts[method_name] = record
        self._conflict_log.append(record)

    def provider(self, name: str) -> Optional["BaseProvider"]:
        try:
            lst = [p for p in self.get_providers() if hasattr(p, "__provider__") and p.__provider__ == name.lower()]
            return lst[0]
        except IndexError:
            return None

    def get_providers(self) -> List["BaseProvider"]:
        """Returns added providers."""
        return self.providers

    def get_provider_of(self, method_name: str) -> Optional["BaseProvider"]:
        """Returns the provider currently providing the given method.

        Returns ``None`` when the method is actively shadowed by a caller or
        when no provider provides the method. Use :meth:`get_method_info` to
        distinguish the two cases.
        """
        if method_name in self._shadow_stack:
            return None

        candidates = self._method_providers.get(method_name)
        if not candidates:
            return None
        return self._resolved_winner(method_name, candidates)

    # descriptive alias
    method_provider = get_provider_of

    def get_provider_name_of(self, method_name: str) -> Optional[str]:
        """Returns the name metadata of the provider of the given method."""
        provider = self.get_provider_of(method_name)
        return getattr(provider, "__provider__", None) if provider is not None else None

    def get_method_info(self, method_name: str) -> Optional[MethodResolution]:
        """Returns a full resolution report for a single method."""
        candidates = self._method_providers.get(method_name)
        if not candidates:
            return None

        shadowed = method_name in self._shadow_stack
        provider = None if shadowed else self._resolved_winner(method_name, candidates)
        return MethodResolution(method_name, provider, list(candidates), shadowed)

    def get_method_owners(self) -> Dict[str, Optional["BaseProvider"]]:
        """Returns a mapping of every faker method name to its current provider."""
        return {
            method_name: (None if method_name in self._shadow_stack else self._resolved_winner(method_name, candidates))
            for method_name, candidates in self._method_providers.items()
        }

    def get_conflicts(self) -> Dict[str, MethodConflict]:
        """Returns the current same-name conflict records, keyed by method name."""
        return dict(self._conflicts)

    def get_conflict_log(self) -> List[MethodConflict]:
        """Returns the append-only log of conflict resolutions."""
        return list(self._conflict_log)

    def is_shadowed(self, method_name: str) -> bool:
        """Returns whether a caller-installed shadow is active for the method."""
        return method_name in self._shadow_stack

    def shadow_method(self, method_name: str, function: Callable) -> None:
        """Temporarily shadows a faker method with the given callable.

        The previous resolution (provider and function) is recorded and can be
        restored with :meth:`restore_method`. Shadows may be nested.
        """
        with self._provider_lock:
            current = getattr(self, method_name, None)
            if current is None:
                raise AttributeError(f"Unknown formatter {method_name!r}")

            previous_provider = getattr(current, "__self__", None)
            self._shadow_stack.setdefault(method_name, []).append(
                _ShadowRecord(previous_provider, current),
            )
            self.set_formatter(method_name, function)
            self._dispatch_version += 1

    def restore_method(self, method_name: str) -> Callable:
        """Restores a shadowed method to its pre-shadow resolution.

        Nested shadows are unwound one layer at a time.
        """
        with self._provider_lock:
            stack = self._shadow_stack.get(method_name)
            if not stack:
                raise ValueError(f"Method {method_name!r} is not shadowed")

            record = stack.pop()
            final_layer = not stack
            if final_layer:
                del self._shadow_stack[method_name]

            # Bind the restored function first; the method never disappears.
            self.set_formatter(method_name, record.function)

            if not final_layer:
                # A shadow is still active on an inner layer; the method
                # remains shadowed and nothing needs reconciling.
                self._dispatch_version += 1
                return record.function

            # Final layer popped: reconcile with providers registered while
            # the shadow was active.
            candidates = self._method_providers.get(method_name, [])
            natural_winner = self._select_winner(candidates) if candidates else None
            restored_owner = record.provider
            if restored_owner is None:
                restored_owner = natural_winner

            if natural_winner is restored_owner or restored_owner is None:
                # Natural resolution already matches the restored ownership
                self._restored_owners.pop(method_name, None)
            elif natural_winner is not None and (
                self._underlying_function(natural_winner, method_name)
                == self._underlying_function(restored_owner, method_name)
            ):
                # The natural winner merely inherited the same implementation:
                # let natural resolution win so the report and binding agree.
                self.set_formatter(method_name, getattr(natural_winner, method_name))
                self._restored_owners.pop(method_name, None)
            else:
                # Honor the requirement to restore the pre-shadow attribution:
                # record a restoration decree and refresh the conflict record
                # so every report stays consistent.
                self._restored_owners[method_name] = restored_owner
                self._record_conflict_if_any(method_name, candidates)

            self._dispatch_version += 1
            return record.function

    # descriptive alias
    unshadow_method = restore_method

    @property
    def random(self) -> random_module.Random:
        return self.__random

    @random.setter
    def random(self, value: random_module.Random) -> None:
        self.__random = value

    def seed_instance(self, seed: Optional[SeedType] = None) -> "Generator":
        """Calls random.seed"""
        if self.__random == random:
            # create per-instance random obj when first time seed_instance() is
            # called
            self.__random = random_module.Random()
        self.__random.seed(seed)
        self._is_seeded = True
        return self

    @classmethod
    def seed(cls, seed: Optional[SeedType] = None) -> None:
        random.seed(seed)
        cls._global_seed = seed
        cls._is_seeded = True

    def format(self, formatter: str, *args: Any, **kwargs: Any) -> str:
        """
        This is a secure way to make a fake from another Provider.
        """
        return self.get_formatter(formatter)(*args, **kwargs)

    def get_formatter(self, formatter: str) -> Callable:
        try:
            return getattr(self, formatter)
        except AttributeError:
            if "locale" in self.__config:
                msg = f'Unknown formatter {formatter!r} with locale {self.__config["locale"]!r}'
            else:
                raise AttributeError(f"Unknown formatter {formatter!r}")
            raise AttributeError(msg)

    def set_formatter(self, name: str, formatter: Callable) -> None:
        """
        This method adds a provider method to generator.
        Override this method to add some decoration or logging stuff.
        """
        setattr(self, name, formatter)

    def set_arguments(self, group: str, argument: str, value: Optional[Any] = None) -> None:
        """
        Creates an argument group, with an individual argument or a dictionary
        of arguments. The argument groups is used to apply arguments to tokens,
        when using the generator.parse() method. To further manage argument
        groups, use get_arguments() and del_arguments() methods.

        generator.set_arguments('small', 'max_value', 10)
        generator.set_arguments('small', {'min_value': 5, 'max_value': 10})
        """
        if group not in self.__config["arguments"]:
            self.__config["arguments"][group] = {}

        if isinstance(argument, dict):
            self.__config["arguments"][group] = argument
        elif not isinstance(argument, str):
            raise ValueError("Arguments must be either a string or dictionary")
        else:
            self.__config["arguments"][group][argument] = value

    def get_arguments(self, group: str, argument: Optional[str] = None) -> Any:
        """
        Get the value of an argument configured within a argument group, or
        the entire group as a dictionary. Used in conjunction with the
        set_arguments() method.

        generator.get_arguments('small', 'max_value')
        generator.get_arguments('small')
        """
        if group in self.__config["arguments"] and argument:
            result = self.__config["arguments"][group].get(argument)
        else:
            result = self.__config["arguments"].get(group)

        return result

    def del_arguments(self, group: str, argument: Optional[str] = None) -> Any:
        """
        Delete an argument from an argument group or the entire argument group.
        Used in conjunction with the set_arguments() method.

        generator.del_arguments('small')
        generator.del_arguments('small', 'max_value')
        """
        if group in self.__config["arguments"]:
            if argument:
                result = self.__config["arguments"][group].pop(argument)
            else:
                result = self.__config["arguments"].pop(group)
        else:
            result = None

        return result

    def parse(self, text: str) -> str:
        """
        Replaces tokens like '{{ tokenName }}' or '{{tokenName}}' in a string with
        the result from the token method call. Arguments can be parsed by using an
        argument group. For more information on the use of argument groups, please
        refer to the set_arguments() method.

        Example:

        generator.set_arguments('red_rgb', {'hue': 'red', 'color_format': 'rgb'})
        generator.set_arguments('small', 'max_value', 10)

        generator.parse('{{ color:red_rgb }} - {{ pyint:small }}')
        """
        return _re_token.sub(self.__format_token, text)

    def __format_token(self, matches):
        formatter, argument_group = list(matches.groups())
        argument_group = argument_group.lstrip(":").strip() if argument_group else ""

        if argument_group:
            try:
                arguments = self.__config["arguments"][argument_group]
            except KeyError:
                raise AttributeError(f"Unknown argument group {argument_group!r}")

            formatted = str(self.format(formatter, **arguments))
        else:
            formatted = str(self.format(formatter))

        return "".join(formatted)
