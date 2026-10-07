import random as random_module
import re
import threading

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Type,
    Union,
)

from .exceptions import IncompatibleSnapshotError, UnrepresentableProviderError
from .snapshot import (
    SNAPSHOT_VERSION,
    class_coords,
    decode_random_state,
    decode_value,
    encode_random_state,
    encode_value,
    faker_version,
    is_representable_class,
    provider_methods,
    resolve_class,
)
from .typing import SeedType

if TYPE_CHECKING:
    from .providers import BaseProvider

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")
random = random_module.Random()
mod_random = random  # compat with name released in 0.8


Sentinel = object()


class Generator:
    __config: Dict[str, Dict[Hashable, Any]] = {
        "arguments": {},
    }

    _is_seeded = False
    _global_seed = Sentinel

    # Names of all formatter methods currently bound on this generator.
    _formatters: set = set()
    # Number of built-in providers at the tail of ``self.providers`` once the
    # factory build is complete; ``None`` for instances built before this
    # marker existed.
    _factory_baseline: Optional[int] = None

    def __init__(self, **config: Dict) -> None:
        self.providers: List["BaseProvider"] = []
        self.__config = dict(list(self.__config.items()) + list(config.items()))
        self.__random = random
        self._formatters = set()
        self._factory_baseline = None
        self._state_lock = threading.RLock()

    def add_provider(self, provider: Union["BaseProvider", Type["BaseProvider"]]) -> None:
        if isinstance(provider, type):
            provider = provider(self)

        self.providers.insert(0, provider)

        for method_name in dir(provider):
            # skip 'private' method
            if method_name.startswith("_"):
                continue

            faker_function = getattr(provider, method_name)

            if callable(faker_function):
                # add all faker method to generator
                self.set_formatter(method_name, faker_function)
                self._formatters.add(method_name)

    def provider(self, name: str) -> Optional["BaseProvider"]:
        try:
            lst = [p for p in self.get_providers() if hasattr(p, "__provider__") and p.__provider__ == name.lower()]
            return lst[0]
        except IndexError:
            return None

    def get_providers(self) -> List["BaseProvider"]:
        """Returns added providers."""
        return self.providers

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

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Export this generator's state as a portable, JSON-compatible
        dictionary.

        The snapshot covers the random source position, the argument
        groups, and descriptions of all runtime-registered providers. It is
        captured under :attr:`_state_lock`, so a concurrent :meth:`restore`
        can only be observed as entirely applied or not applied at all.
        """
        with self._state_lock:
            return self._build_snapshot()

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore this generator's state from ``snapshot``.

        The snapshot is fully decoded and validated (all custom provider
        classes resolved and instantiated) before any live state is
        changed, so a rejected import never leaves a half-configured
        generator.
        """
        with self._state_lock:
            plan = self._plan_restore(snapshot)
            self._publish_restore(plan)

    def _build_snapshot(self) -> Dict[str, Any]:
        customs, builtins = self._split_providers()
        locale = self._current_locale()

        # Reject non-portable runtime providers at export time with a
        # locatable description.
        for index, provider in enumerate(customs):
            cls = type(provider)
            if not is_representable_class(cls):
                module, qualname = class_coords(cls)
                raise UnrepresentableProviderError(
                    f"Cannot snapshot runtime provider {module}.{qualname} registered at "
                    f"position {index} on locale generator "
                    f"{locale!r}: the class is not importable in "
                    f"another process (defined locally or in __main__)",
                    provider_class=f"{module}.{qualname}",
                    module=module,
                    locale=locale,
                    index=index,
                    methods=provider_methods(provider),
                )

        providers_data = [self._describe_provider(provider, "custom") for provider in customs]
        providers_data += [self._describe_provider(provider, "builtin") for provider in builtins]

        return {
            "object": "generator",
            "snapshot_version": SNAPSHOT_VERSION,
            "faker_version": faker_version(),
            "locale": locale,
            "use_weighting": self.__config.get("use_weighting", True),
            "arguments": encode_value(self.__config.get("arguments", {})),
            "random": {
                "shared": self.__random is random,
                "state": encode_random_state(self.__random.getstate()),
            },
            "baseline": len(builtins),
            # Factory-ordered (oldest-insertion-first) built-in provider paths.
            "builtin_spec": [getattr(provider, "__provider__", None) for provider in reversed(builtins)],
            "providers": providers_data,
        }

    def _current_locale(self) -> Optional[str]:
        """The locale recorded in this generator's config (always a string in
        factory-built generators)."""
        locale = self.__config.get("locale")
        return str(locale) if locale is not None else None

    def _describe_provider(self, provider: "BaseProvider", kind: str) -> Dict[str, Any]:
        module, qualname = class_coords(type(provider))
        entry: Dict[str, Any] = {
            "kind": kind,
            "module": module,
            "class": qualname,
            "provider": getattr(provider, "__provider__", None),
            "lang": getattr(provider, "__lang__", None),
        }
        if kind == "custom":
            entry["use_weighting"] = getattr(provider, "__use_weighting__", False)
            entry["methods"] = provider_methods(provider)
        return entry

    def _plan_restore(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(snapshot, dict):
            raise IncompatibleSnapshotError("Generator snapshot must be a dictionary")

        version = snapshot.get("snapshot_version")
        if version != SNAPSHOT_VERSION:
            raise IncompatibleSnapshotError(f"Unsupported snapshot version {version!r}; expected {SNAPSHOT_VERSION}")
        if snapshot.get("object", "generator") != "generator":
            raise IncompatibleSnapshotError(f"Expected a generator snapshot, got {snapshot.get('object')!r}")

        locale = self._current_locale()
        if snapshot.get("locale") != locale:
            raise IncompatibleSnapshotError(
                f"Snapshot locale {snapshot.get('locale')!r} does not match target " f"generator locale {locale!r}"
            )

        random_section = snapshot.get("random")
        if not isinstance(random_section, dict):
            raise IncompatibleSnapshotError("Malformed 'random' section in snapshot")
        random_shared = bool(random_section.get("shared", False))
        random_state = decode_random_state(random_section.get("state"))

        arguments = decode_value(snapshot.get("arguments", {}))
        if not isinstance(arguments, dict):
            raise IncompatibleSnapshotError("Malformed 'arguments' section in snapshot")

        customs, builtins = self._split_providers()
        entries = snapshot.get("providers", [])
        if not isinstance(entries, list):
            raise IncompatibleSnapshotError("Malformed 'providers' section in snapshot")

        custom_entries = [entry for entry in entries if entry.get("kind") == "custom"]
        builtin_entries = [entry for entry in entries if entry.get("kind") == "builtin"]

        # Validate the built-in tail: it must line up position by position.
        if len(builtin_entries) != len(builtins):
            raise IncompatibleSnapshotError(
                f"Snapshot describes {len(builtin_entries)} built-in provider(s) for locale "
                f"{locale!r}, but the target generator has {len(builtins)}"
            )
        for offset, (provider, entry) in enumerate(zip(builtins, builtin_entries)):
            module, qualname = class_coords(type(provider))
            if (module, qualname) != (entry.get("module"), entry.get("class")):
                position = len(custom_entries) + offset
                raise IncompatibleSnapshotError(
                    f"Built-in provider at position {position} on locale generator "
                    f"{locale!r} is {module}.{qualname}, but the snapshot requires "
                    f"{entry.get('module')}.{entry.get('class')}"
                )

        # Resolve and instantiate every runtime provider BEFORE mutating
        # anything, so failures are locatable and leave no partial state.
        new_custom: List[tuple] = []
        for index, entry in enumerate(custom_entries):
            module = entry.get("module")
            qualname = entry.get("class")
            coords = f"{module}.{qualname}"
            try:
                provider_cls = resolve_class(module, qualname)
            except (ImportError, AttributeError) as exc:
                raise UnrepresentableProviderError(
                    f"Runtime provider {coords} registered at position {index} on locale "
                    f"generator {locale!r} cannot be restored: the class is not importable",
                    provider_class=coords,
                    module=module,
                    locale=locale,
                    index=index,
                    methods=entry.get("methods"),
                ) from exc
            try:
                instance = provider_cls(self)
            except TypeError as exc:
                raise UnrepresentableProviderError(
                    f"Runtime provider {coords} registered at position {index} on locale "
                    f"generator {locale!r} cannot be restored: its constructor does not "
                    f"accept the generator argument",
                    provider_class=coords,
                    module=module,
                    locale=locale,
                    index=index,
                    methods=entry.get("methods"),
                ) from exc
            new_custom.append((instance, entry))

        return {
            "locale": locale,
            "random_shared": random_shared,
            "random_state": random_state,
            "arguments": arguments,
            "builtins": builtins,
            "new_custom": new_custom,
        }

    def _publish_restore(self, plan: Dict[str, Any]) -> None:
        if plan["random_shared"]:
            random.setstate(plan["random_state"])
            self.__random = random
        else:
            rng = self.__random
            if rng is random:
                rng = random_module.Random()
            rng.setstate(plan["random_state"])
            self.__random = rng

        self.__config["arguments"] = plan["arguments"]

        for instance, entry in plan["new_custom"]:
            provider_tag = entry.get("provider")
            if provider_tag is not None:
                setattr(instance, "__provider__", provider_tag)
            setattr(instance, "__lang__", entry.get("lang"))
            setattr(instance, "__use_weighting__", entry.get("use_weighting", False))

        new_instances = [instance for instance, _entry in plan["new_custom"]]
        self._rebuild_providers(new_instances, plan["builtins"])

    def _rebuild_providers(self, new_custom_instances: List["BaseProvider"], builtins: List["BaseProvider"]) -> None:
        """Replace the provider chain and rebuild every formatter binding so
        that providers removed by the import leave no stale bound methods."""
        final = list(new_custom_instances) + list(builtins)

        for name in self._known_formatter_names():
            try:
                delattr(self, name)
            except AttributeError:
                pass

        self._formatters = set()
        for provider in reversed(final):
            for method_name in dir(provider):
                if method_name.startswith("_"):
                    continue
                faker_function = getattr(provider, method_name)
                if callable(faker_function):
                    self.set_formatter(method_name, faker_function)
                    self._formatters.add(method_name)

        self.providers = final
        self._factory_baseline = len(builtins)

    def _known_formatter_names(self) -> set:
        formatters = getattr(self, "_formatters", None)
        if formatters is not None:
            return set(formatters)
        # Generators created before formatter tracking existed.
        names = set()
        for provider in self.providers:
            for method_name in dir(provider):
                if method_name.startswith("_"):
                    continue
                if callable(getattr(provider, method_name)):
                    names.add(method_name)
        return names

    def _split_providers(self) -> tuple[List["BaseProvider"], List["BaseProvider"]]:
        """Split the provider chain into (runtime providers, built-in
        providers).

        When the factory baseline marker is present, runtime providers are
        the chain head and built-in providers the tail. Otherwise fall back
        to class identity against what the factory would resolve for this
        locale.
        """
        providers = self.providers
        baseline = getattr(self, "_factory_baseline", None)
        if baseline is not None and 0 <= baseline <= len(providers):
            cut = len(providers) - baseline
            return providers[:cut], providers[cut:]

        customs, builtins = [], []
        for provider in providers:
            if self._looks_like_builtin(provider):
                builtins.append(provider)
            else:
                customs.append(provider)
        return customs, builtins

    def _looks_like_builtin(self, provider: "BaseProvider") -> bool:
        path = getattr(provider, "__provider__", None)
        if not isinstance(path, str):
            return False
        if path != "faker.providers" and not path.startswith("faker.providers."):
            return False
        try:
            from .factory import Factory

            provider_cls, _lang_found, _default = Factory._find_provider_class(path, self._current_locale())
        except Exception:  # noqa: BLE001 - unresolvable -> not a built-in
            return False
        return type(provider) is provider_cls

    def __getstate__(self) -> Dict[str, Any]:
        # Locks are not picklable and must be recreated in the new process.
        return {key: value for key, value in self.__dict__.items() if key != "_state_lock"}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._state_lock = threading.RLock()
