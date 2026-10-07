import contextlib
import copy
import random as random_module
import re
import threading

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

from .exceptions import (
    ArgumentGroupNotFound,
    ArgumentNotFound,
    TemplateCycleError,
    UnknownArgumentGroup,
    UnknownTemplate,
)
from .typing import SeedType

if TYPE_CHECKING:
    from .providers import BaseProvider

_re_token = re.compile(r"\{\{\s*(\w+)(:\s*\w+?)?\s*\}\}")
random = random_module.Random()
mod_random = random  # compat with name released in 0.8

# Maximum number of nested template expansions allowed within one parse()
# batch. Acts as a safety net on top of the explicit cycle detection.
_MAX_TEMPLATE_DEPTH = 100


Sentinel = object()


class _ArgumentGroupStore:
    """Argument group storage separated per generator instance and per scope.

    Groups configured outside of any scope live in a per-instance base mapping.
    Entering a scope pushes a deep copy of the currently visible mappings onto a
    thread-local stack; all reads and writes inside the scope target that frame.
    Leaving the scope simply drops the frame, restoring the outer snapshot, so
    scopes never leak into each other and concurrent threads cannot observe each
    other's groups.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._revisions: Dict[str, int] = {}
        self._revision = 0
        self._local = threading.local()

    def _frames(self) -> List[Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]]:
        frames = getattr(self._local, "frames", None)
        if frames is None:
            frames = []
            self._local.frames = frames
        return frames

    def _active(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
        frames = self._frames()
        if frames:
            return frames[-1]
        return self._groups, self._revisions

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_ArgumentGroupStore":
        # Locks and thread-local state cannot be copied; scope frames are
        # transient and therefore only the visible base state is duplicated.
        clone = self.__class__.__new__(self.__class__)
        clone._lock = threading.RLock()
        clone._local = threading.local()
        with self._lock:
            clone._groups = copy.deepcopy(self._groups, memo)
            clone._revisions = copy.deepcopy(self._revisions, memo)
            clone._revision = self._revision
        return clone

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "groups": self._groups,
            "revisions": self._revisions,
            "revision": self._revision,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._groups = state["groups"]
        self._revisions = state["revisions"]
        self._revision = state["revision"]

    def enter_scope(self) -> None:
        """Inherit a private snapshot of the outer groups and revisions."""
        with self._lock:
            groups, revisions = self._active()
            self._frames().append((copy.deepcopy(groups), dict(revisions)))

    def exit_scope(self) -> None:
        """Drop the current frame, restoring the outer snapshot."""
        with self._lock:
            self._frames().pop()

    def _bump_revision(self, revisions: Dict[str, int], group: str) -> None:
        self._revision += 1
        revisions[group] = self._revision

    def set(self, group: str, argument: Union[str, Dict[str, Any]], value: Optional[Any] = None) -> None:
        with self._lock:
            groups, revisions = self._active()
            if group not in groups:
                groups[group] = {}
            if isinstance(argument, dict):
                groups[group] = copy.deepcopy(argument)
            else:
                groups[group][argument] = copy.deepcopy(value)
            self._bump_revision(revisions, group)

    def get(self, group: str, argument: Optional[str] = None) -> Any:
        with self._lock:
            groups, _ = self._active()
            if group not in groups:
                raise ArgumentGroupNotFound(group)
            if argument:
                if argument not in groups[group]:
                    raise ArgumentNotFound(group, argument)
                return groups[group][argument]
            return groups[group]

    def delete(self, group: str, argument: Optional[str] = None) -> Any:
        with self._lock:
            groups, revisions = self._active()
            if group not in groups:
                raise ArgumentGroupNotFound(group)
            if argument:
                if argument not in groups[group]:
                    raise ArgumentNotFound(group, argument)
                result = groups[group].pop(argument)
            else:
                result = groups.pop(group)
                revisions.pop(group, None)
            self._revision += 1
            return result

    def revision(self, group: str) -> int:
        with self._lock:
            _, revisions = self._active()
            if group not in revisions:
                raise ArgumentGroupNotFound(group)
            return revisions[group]

    def resolve(
        self,
        group: str,
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the group for template expansion, or a locatable error."""
        with self._lock:
            groups, _ = self._active()
            if group not in groups:
                raise UnknownArgumentGroup(group, token=token, position=position, template=template)
            return groups[group]


class Generator:
    __config: Dict[str, Any] = {}

    _is_seeded = False
    _global_seed = Sentinel

    def __init__(self, **config: Any) -> None:
        self.providers: List["BaseProvider"] = []
        self.__config = dict(list(self.__config.items()) + list(config.items()))
        self.__random = random
        self.__arguments = _ArgumentGroupStore()

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

    def set_arguments(self, group: str, argument: Union[str, Dict[str, Any]], value: Optional[Any] = None) -> None:
        """
        Creates an argument group, with an individual argument or a dictionary
        of arguments. The argument groups is used to apply arguments to tokens,
        when using the generator.parse() method. To further manage argument
        groups, use get_arguments() and del_arguments() methods.

        generator.set_arguments('small', 'max_value', 10)
        generator.set_arguments('small', {'min_value': 5, 'max_value': 10})
        """
        if not isinstance(group, str):
            raise TypeError("Argument group name must be a string")
        if not group:
            raise ValueError("Argument group name must not be empty")
        if not isinstance(argument, (str, dict)):
            raise ValueError("Arguments must be either a string or dictionary")
        self.__arguments.set(group, argument, value)

    def get_arguments(self, group: str, argument: Optional[str] = None) -> Any:
        """
        Get the value of an argument configured within a argument group, or
        the entire group as a dictionary. Used in conjunction with the
        set_arguments() method.

        generator.get_arguments('small', 'max_value')
        generator.get_arguments('small')

        Raises ArgumentGroupNotFound / ArgumentNotFound instead of silently
        returning None when the group or argument has not been defined.
        """
        return self.__arguments.get(group, argument)

    def del_arguments(self, group: str, argument: Optional[str] = None) -> Any:
        """
        Delete an argument from an argument group or the entire argument group.
        Used in conjunction with the set_arguments() method. Returns the
        deleted value or group dictionary.

        generator.del_arguments('small')
        generator.del_arguments('small', 'max_value')

        Raises ArgumentGroupNotFound / ArgumentNotFound when there is nothing
        to delete instead of silently returning None.
        """
        return self.__arguments.delete(group, argument)

    def get_arguments_version(self, group: str) -> int:
        """
        Return the current revision of an argument group. The revision is a
        monotonically increasing integer that changes every time the group is
        created, overridden or deleted and recreated, allowing callers to tell
        these situations apart.
        """
        return self.__arguments.revision(group)

    @contextlib.contextmanager
    def arguments_scope(self) -> Iterator[None]:
        """
        Context manager introducing a scoped view of the argument groups.

        On entry the scope inherits a private snapshot of every currently
        visible group; set/get/delete operations inside the ``with`` block only
        affect that snapshot and the outer groups are restored on exit, even if
        the block raises. Scopes are thread-local, so concurrent callers can
        neither observe nor corrupt each other's groups. Nested scopes inherit
        their enclosing scope's snapshot.

        with generator.arguments_scope():
            generator.set_arguments('small', 'max_value', 10)
            generator.parse('{{ pyint:small }}')
        # 'small' is not visible (and the outer groups are unchanged) here
        """
        self.__arguments.enter_scope()
        try:
            yield
        finally:
            self.__arguments.exit_scope()

    def format_token(
        self,
        formatter: str,
        group: str = "",
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
        **arguments: Any,
    ) -> Any:
        """
        Resolve a single template token: validate the formatter name and its
        optional argument group, then invoke the formatter. Used by parse() and
        by structured output providers so that both report the same locatable
        errors instead of low-level AttributeError / TypeError failures.

        Additional keyword arguments are passed to the formatter; arguments
        coming from the referenced group take precedence.
        """
        if group:
            arguments.update(self.__arguments.resolve(group, token=token, position=position, template=template))

        resolved = getattr(self, formatter, Sentinel)
        if resolved is Sentinel:
            raise UnknownTemplate(
                formatter,
                token=token,
                position=position,
                template=template,
                locale=self.__config.get("locale"),
            )
        if not callable(resolved):
            raise UnknownTemplate(
                formatter,
                token=token,
                position=position,
                template=template,
                is_callable=False,
            )

        return self.format(formatter, **arguments)

    def parse(self, text: str) -> str:
        """
        Replaces tokens like '{{ tokenName }}' or '{{tokenName}}' in a string with
        the result from the token method call. Arguments can be parsed by using an
        argument group. For more information on the use of argument groups, please
        refer to the set_arguments() method.

        Tokens are expanded recursively, so formatters may return strings
        containing further tokens. Unknown formatters, missing argument groups
        and recursive expansion loops raise a locatable TemplateError. Each
        parse() call is an atomic batch: it runs against a private, thread
        local snapshot of the argument groups and nothing it writes survives a
        failed batch or leaks to concurrent callers.

        Example:

        generator.set_arguments('red_rgb', {'hue': 'red', 'color_format': 'rgb'})
        generator.set_arguments('small', 'max_value', 10)

        generator.parse('{{ color:red_rgb }} - {{ pyint:small }}')
        """
        self.__arguments.enter_scope()
        try:
            return self.__expand(text, (), 0)
        finally:
            self.__arguments.exit_scope()

    def __expand(
        self,
        text: str,
        chain: Tuple[Tuple[str, str], ...],
        depth: int,
    ) -> str:
        if depth > _MAX_TEMPLATE_DEPTH:
            raise TemplateCycleError(tuple(chain), template=text)

        parts: List[str] = []
        cursor = 0
        for match in _re_token.finditer(text):
            parts.append(text[cursor : match.start()])
            parts.append(self.__expand_token(match, text, chain, depth))
            cursor = match.end()
        parts.append(text[cursor:])
        return "".join(parts)

    def __expand_token(
        self,
        match: "re.Match[str]",
        template: str,
        chain: Tuple[Tuple[str, str], ...],
        depth: int,
    ) -> str:
        formatter, raw_group = match.groups()
        group = raw_group.lstrip(":").strip() if raw_group else ""
        token = match.group(0)
        position = match.start()
        key = (formatter, group)

        if key in chain:
            cycle = chain[chain.index(key) :] + (key,)
            raise TemplateCycleError(cycle, token=token, position=position, template=template)

        value = self.format_token(formatter, group, token=token, position=position, template=template)
        return self.__expand(str(value), chain + (key,), depth + 1)
