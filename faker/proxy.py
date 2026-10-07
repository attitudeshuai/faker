from __future__ import annotations

import copy
import functools
import logging
import math
import re
import threading

from collections import OrderedDict
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Mapping, Pattern, Sequence, TypeVar, cast

from .config import DEFAULT_LOCALE
from .exceptions import UniquenessException
from .factory import Factory
from .generator import Generator
from .typing import SeedType
from .utils.distribution import choices_distribution

_UNIQUE_ATTEMPTS = 1000

RetType = TypeVar("RetType")

logger = logging.getLogger(__name__)

#: Degradation strategy: locales that do not implement a provider method are
#: excluded from that method's candidates; selection runs over the rest.
EXCLUDE_ON_MISSING = "exclude"
#: Degradation strategy: if any declared locale does not implement the
#: requested method, the call fails instead of silently reshaping the
#: distribution.
FAIL_ON_MISSING = "fail"

_VALID_MISSING_STRATEGIES = (EXCLUDE_ON_MISSING, FAIL_ON_MISSING)


@dataclass(frozen=True)
class CandidateMapping:
    """
    Immutable per-method candidate snapshot.

    Instances are published atomically and never mutated afterwards, so
    concurrent readers may keep using a snapshot even while a new one is
    being built; they can never observe a mix of old and new candidates.
    """

    method_name: str
    locales: tuple[str, ...]
    factories: tuple[Generator | Faker, ...]
    weights: tuple[int | float, ...] | None
    excluded_locales: tuple[str, ...]


@dataclass(frozen=True)
class SelectionInfo:
    """Record describing one factory selection decision."""

    method_name: str
    locale: str | None
    candidate_locales: tuple[str, ...]
    excluded_locales: tuple[str, ...]


class Faker:
    """Proxy class capable of supporting multiple locales"""

    cache_pattern: Pattern = re.compile(r"^_cached_\w*_mapping$")
    generator_attrs = [
        attr for attr in dir(Generator) if not attr.startswith("__") and attr not in ["seed", "seed_instance", "random"]
    ]

    def __init__(
        self,
        locale: str | Sequence[str] | dict[str, int | float] | None = None,
        providers: list[str] | None = None,
        generator: Generator | None = None,
        includes: list[str] | None = None,
        use_weighting: bool = True,
        on_missing: str = EXCLUDE_ON_MISSING,
        **config: Any,
    ) -> None:
        self._factory_map: OrderedDict[str, Generator | Faker] = OrderedDict()
        self._weights = None
        self._unique_proxy = UniqueProxy(self)
        self._optional_proxy = OptionalProxy(self)

        # Unified, lock-guarded selection state: weights, per-method candidate
        # cache and selection records all share the same invalidation path.
        self._method_mappings: dict[str, CandidateMapping] = {}
        self._selection_lock = threading.RLock()
        self._last_selection: SelectionInfo | None = None

        if on_missing not in _VALID_MISSING_STRATEGIES:
            msg = (
                f"Invalid on_missing strategy {on_missing!r}; "
                f"expected one of {list(_VALID_MISSING_STRATEGIES)}."
            )
            raise ValueError(msg)
        self._on_missing = on_missing

        if isinstance(locale, str):
            locales = [locale.replace("-", "_")]

        # This guarantees a FIFO ordering of elements in `locales` based on the final
        # locale string while discarding duplicates after processing
        elif isinstance(locale, (list, tuple, set)):
            locales = []
            for code in locale:
                if not isinstance(code, str):
                    raise TypeError(f'The locale "{str(code)}" must be a string.')
                final_locale = code.replace("-", "_")
                if final_locale not in locales:
                    locales.append(final_locale)

        elif isinstance(locale, (OrderedDict, dict)):
            odict = OrderedDict()
            for k, v in locale.items():
                key = k.replace("-", "_")
                odict[key] = v
            locales = list(odict.keys())
            self._weights = list(odict.values())

        else:
            locales = [DEFAULT_LOCALE]

        # Validate the declared weight vector through the same code path used
        # for runtime updates, before any factory is created.
        self._locales = locales
        self._weights = self._normalize_weights(self._weights)

        if len(locales) == 1:
            self._factory_map[locales[0]] = Factory.create(
                locales[0],
                providers,
                generator,
                includes,
                use_weighting=use_weighting,
                **config,
            )
        else:
            for locale in locales:
                self._factory_map[locale] = Faker(
                    locale,
                    providers,
                    generator,
                    includes,
                    use_weighting=use_weighting,
                    on_missing=on_missing,
                    **config,
                )

        self._factories = list(self._factory_map.values())

    def __dir__(self):
        attributes = set(super().__dir__())
        for factory in self.factories:
            attributes |= {attr for attr in dir(factory) if not attr.startswith("_")}
        return sorted(attributes)

    def __getitem__(self, locale: str) -> Faker:
        if locale.replace("-", "_") in self.locales and len(self.locales) == 1:
            return self
        instance = self._factory_map[locale.replace("-", "_")]
        assert isinstance(instance, Faker)  # for mypy
        return instance

    def __getattribute__(self, attr: str) -> Any:
        """
        Handles the "attribute resolution" behavior for declared members of this proxy class

        The class method `seed` cannot be called from an instance.

        :param attr: attribute name
        :return: the appropriate attribute
        """
        if attr == "seed":
            msg = "Calling `.seed()` on instances is deprecated. Use the class method `Faker.seed()` instead."
            raise TypeError(msg)
        else:
            return super().__getattribute__(attr)

    def __getattr__(self, attr: str) -> Any:
        """
        Handles cache access and proxying behavior

        :param attr: attribute name
        :return: the appropriate attribute
        """
        if len(self._factories) == 1:
            return getattr(self._factories[0], attr)
        elif attr in self.generator_attrs:
            msg = "Proxying calls to `%s` is not implemented in multiple locale mode." % attr
            raise NotImplementedError(msg)
        elif self.cache_pattern.match(attr):
            msg = "Cached attribute `%s` does not exist" % attr
            raise AttributeError(msg)
        else:
            factory = self._select_factory(attr)
            return getattr(factory, attr)

    def __deepcopy__(self, memodict):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        result._locales = copy.deepcopy(self._locales, memodict)
        result._factory_map = copy.deepcopy(self._factory_map, memodict)
        result._factories = list(result._factory_map.values())
        result._weights = copy.deepcopy(self._weights, memodict)
        # Cached snapshots reference the old factories and must not be shared;
        # they are rebuilt lazily from the copied factories and weights.
        result._method_mappings = {}
        result._selection_lock = threading.RLock()
        result._on_missing = self._on_missing
        result._last_selection = None
        result._unique_proxy = UniqueProxy(result)
        result._unique_proxy._seen = {k: {result._unique_proxy._sentinel} for k in self._unique_proxy._seen.keys()}
        result._optional_proxy = OptionalProxy(result)
        return result

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Locks cannot be pickled, and candidate snapshots are rebuilt lazily
        # from the pickled factories and weights.
        state["_selection_lock"] = None
        state["_method_mappings"] = {}
        return state

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)
        # Backwards compatibility for instances pickled before the unified
        # selection state existed.
        self.__dict__.setdefault("_method_mappings", {})
        self.__dict__.setdefault("_last_selection", None)
        self.__dict__.setdefault("_on_missing", EXCLUDE_ON_MISSING)
        if self.__dict__.get("_selection_lock") is None:
            self.__dict__["_selection_lock"] = threading.RLock()

    @property
    def unique(self) -> UniqueProxy:
        return self._unique_proxy

    @property
    def optional(self) -> OptionalProxy:
        return self._optional_proxy

    def _select_factory(self, method_name: str) -> Factory:
        """
        Returns a random factory that supports the provider method

        The selection, together with the locales excluded from the candidate
        list, is recorded and can be queried through ``last_selection``.

        :param method_name: Name of provider method
        :return: A factory that supports the provider method
        """

        with self._selection_lock:
            mapping = self._get_mapping(method_name)

            if self._on_missing == FAIL_ON_MISSING and mapping.excluded_locales:
                self._last_selection = SelectionInfo(
                    method_name,
                    None,
                    mapping.locales,
                    mapping.excluded_locales,
                )
                msg = (
                    f"Provider method {method_name!r} is not available for "
                    f"locale(s) {list(mapping.excluded_locales)} and the "
                    f"{FAIL_ON_MISSING!r} degradation strategy is in effect."
                )
                raise AttributeError(msg)

            if len(mapping.factories) == 0:
                self._last_selection = SelectionInfo(
                    method_name,
                    None,
                    mapping.locales,
                    mapping.excluded_locales,
                )
                msg = f"No generator object has attribute {method_name!r}"
                raise AttributeError(msg)
            elif len(mapping.factories) == 1:
                factory = mapping.factories[0]
            elif mapping.weights is not None:
                factory = self._select_factory_distribution(
                    list(mapping.factories),
                    list(mapping.weights),
                )
            else:
                factory = self._select_factory_choice(list(mapping.factories))

            # Identity lookup: the selector returned one of the candidate
            # factories; use `is` so generators overriding equality cannot
            # misresolve the locale.
            locale = next(
                (candidate_locale
                 for candidate_locale, candidate_factory in zip(mapping.locales, mapping.factories)
                 if candidate_factory is factory),
                None,
            )
            self._last_selection = SelectionInfo(
                method_name,
                locale,
                mapping.locales,
                mapping.excluded_locales,
            )
            return cast(Factory, factory)

    def _select_factory_distribution(self, factories, weights):
        return choices_distribution(factories, weights, self.factories[0].random, length=1)[0]

    def _select_factory_choice(self, factories):
        return self._factories[0].random.choice(factories)

    def _get_mapping(self, method_name: str) -> CandidateMapping:
        """
        Returns the cached candidate mapping for the method, building it once
        atomically if it does not exist yet.
        """
        with self._selection_lock:
            mapping = self._method_mappings.get(method_name)
            if mapping is None:
                mapping = self._build_mapping(method_name)
                self._method_mappings[method_name] = mapping
            return mapping

    def _build_mapping(self, method_name: str) -> CandidateMapping:
        """
        Builds an immutable candidate snapshot for the provider method.

        Caller must hold ``_selection_lock``. The snapshot is fully assembled
        as a local object before it is published, so readers can never see a
        partially rebuilt candidate list.

        :param method_name: Name of provider method
        :return: Immutable candidate mapping
        """
        locales: list[str] = []
        factories: list[Generator | Faker] = []
        weights: list[int | float] | None = [] if self._weights is not None else None
        excluded: list[str] = []

        for index, (locale, factory) in enumerate(zip(self._locales, self._factories)):
            if hasattr(factory, method_name):
                locales.append(locale)
                factories.append(factory)
                if weights is not None:
                    weights.append(self._weights[index])  # type: ignore[index]
            else:
                excluded.append(locale)

        if excluded:
            logger.debug(
                "Provider method %r is unavailable for locale(s) %s; "
                "applying degradation strategy %r.",
                method_name,
                excluded,
                self._on_missing,
            )

        return CandidateMapping(
            method_name,
            tuple(locales),
            tuple(factories),
            tuple(weights) if weights is not None else None,
            tuple(excluded),
        )

    def _map_provider_method(self, method_name: str) -> tuple[list[Factory], list[float] | None]:
        """
        Creates a 2-tuple of factories and weights for the given provider method name

        The first element of the tuple contains a list of compatible factories.
        The second element of the tuple contains a list of distribution weights.

        :param method_name: Name of provider method
        :return: 2-tuple (factories, weights)
        """
        mapping = self._get_mapping(method_name)
        factories = cast(list[Factory], list(mapping.factories))
        if mapping.weights is None:
            return factories, None
        return factories, list(mapping.weights)

    def set_weights(
        self,
        weights: Mapping[str, int | float] | Sequence[int | float] | None,
    ) -> None:
        """
        Validates and atomically replaces the locale weights.

        ``weights`` is either a mapping of locale to weight covering exactly
        the instance's locales, a sequence aligned with ``locales``, or
        ``None`` to restore equal-probability selection. Invalid input is
        rejected in full: the current weights are kept and the candidate
        cache is left untouched. On success the per-method candidate cache is
        atomically cleared and rebuilt lazily under the selection lock.

        :param weights: New locale weights
        """
        normalized = self._normalize_weights(weights)
        with self._selection_lock:
            self._weights = normalized
            self._method_mappings = {}

    def invalidate_selection_cache(self) -> None:
        """
        Atomically drops every per-method candidate mapping.

        Use this after providers are added or removed at runtime so that the
        next call to each method rebuilds its candidate list from the current
        factories. Weight updates invalidate the cache automatically.
        """
        with self._selection_lock:
            self._method_mappings = {}

    def _normalize_weights(
        self,
        weights: Mapping[str, int | float] | Sequence[int | float] | None,
    ) -> list[int | float] | None:
        """
        Validates a candidate weight vector and returns it aligned with
        ``self._locales``.

        Raises ``TypeError`` or ``ValueError`` on any invalid element. The
        method performs no mutation, so callers can reject a bad vector
        wholesale without touching the current configuration.

        :param weights: Mapping of locale to weight, aligned sequence, or None
        :return: Aligned list of finite non-negative weights, or None
        """
        if weights is None:
            return None

        if isinstance(weights, MappingABC):
            normalized_map: dict[str, Any] = {}
            for key, value in weights.items():
                if not isinstance(key, str):
                    raise TypeError(f'The locale "{key!r}" must be a string.')
                normalized_map[key.replace("-", "_")] = value

            if set(normalized_map) != set(self._locales):
                missing = sorted(set(self._locales) - set(normalized_map))
                extra = sorted(set(normalized_map) - set(self._locales))
                msg = (
                    "Weights must be declared for exactly the instance's "
                    f"locales {self._locales}; missing={missing}, extra={extra}."
                )
                raise ValueError(msg)
            values = [normalized_map[locale] for locale in self._locales]
        elif isinstance(weights, (list, tuple)):
            if len(weights) != len(self._locales):
                msg = (
                    f"Expected {len(self._locales)} weights (one per locale "
                    f"{self._locales}), got {len(weights)}."
                )
                raise ValueError(msg)
            values = list(weights)
        else:
            msg = (
                "Weights must be None, a mapping of locale to weight, or a "
                "sequence aligned with the locales."
            )
            raise TypeError(msg)

        normalized_weights: list[int | float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Weight {value!r} is not a number.")
            if not math.isfinite(value):
                raise ValueError(f"Weight {value!r} must be finite.")
            if value < 0:
                raise ValueError(f"Weight {value!r} must be non-negative.")
            normalized_weights.append(value)

        if sum(normalized_weights) <= 0:
            msg = "The sum of weights must be strictly positive."
            raise ValueError(msg)
        return normalized_weights

    @classmethod
    def seed(cls, seed: SeedType | None = None) -> None:
        """
        Hashables the shared `random.Random` object across all factories

        :param seed: seed value
        """
        Generator.seed(seed)

    def seed_instance(self, seed: SeedType | None = None) -> None:
        """
        Creates and seeds a new `random.Random` object for each factory

        :param seed: seed value
        """
        for factory in self._factories:
            factory.seed_instance(seed)

    def seed_locale(self, locale: str, seed: SeedType | None = None) -> None:
        """
        Creates and seeds a new `random.Random` object for the factory of the specified locale

        :param locale: locale string
        :param seed: seed value
        """
        self._factory_map[locale.replace("-", "_")].seed_instance(seed)

    @property
    def random(self) -> Random:
        """
        Proxies `random` getter calls

        In single locale mode, this will be proxied to the `random` getter
        of the only internal `Generator` object. Subclasses will have to
        implement desired behavior in multiple locale mode.
        """

        if len(self._factories) == 1:
            return self._factories[0].random
        else:
            msg = "Proxying `random` getter calls is not implemented in multiple locale mode."
            raise NotImplementedError(msg)

    @random.setter
    def random(self, value: Random) -> None:
        """
        Proxies `random` setter calls

        In single locale mode, this will be proxied to the `random` setter
        of the only internal `Generator` object. Subclasses will have to
        implement desired behavior in multiple locale mode.
        """

        if len(self._factories) == 1:
            self._factories[0].random = value
        else:
            msg = "Proxying `random` setter calls is not implemented in multiple locale mode."
            raise NotImplementedError(msg)

    @property
    def locales(self) -> list[str]:
        return list(self._locales)

    @property
    def weights(self) -> list[int | float] | None:
        if self._weights is None:
            return None
        return list(self._weights)

    @property
    def factories(self) -> list[Generator | Faker]:
        return self._factories

    @property
    def on_missing(self) -> str:
        return self._on_missing

    @property
    def last_selection(self) -> SelectionInfo | None:
        return self._last_selection

    @property
    def last_selected_locale(self) -> str | None:
        selection = self._last_selection
        return selection.locale if selection is not None else None

    @property
    def last_excluded_locales(self) -> list[str]:
        selection = self._last_selection
        return list(selection.excluded_locales) if selection is not None else []

    def items(self) -> list[tuple[str, Generator | Faker]]:
        return list(self._factory_map.items())


class UniqueProxy:
    def __init__(self, proxy: Faker, excluded_types: tuple[type, ...] = ()):
        self._proxy = proxy
        self._seen: dict = {}
        self._sentinel = object()
        self._excluded_types = excluded_types

    def clear(self) -> None:
        self._seen = {}

    def exclude_types(self, types: list[type]) -> UniqueProxy:
        """Return new UniqueProxy excluding specified types from uniqueness checks.

        Args:
            types: List of types to exclude from uniqueness enforcement

        Returns:
            New UniqueProxy instance with excluded types configured

        Example:
            >>> fake = Faker()
            >>> # Bools won't enforce uniqueness, but other types will
            >>> proxy = fake.unique.exclude_types([bool])
            >>> proxy.pybool()  # Can return duplicates
            >>> proxy.name()  # Still enforces uniqueness
        """
        new_proxy = UniqueProxy(self._proxy, tuple(types))
        new_proxy._seen = self._seen
        new_proxy._sentinel = self._sentinel
        return new_proxy

    def __getitem__(self, locale: str) -> UniqueProxy:
        locale_proxy = self._proxy[locale]
        unique_proxy = UniqueProxy(locale_proxy, self._excluded_types)
        unique_proxy._seen = self._seen
        unique_proxy._sentinel = self._sentinel
        return unique_proxy

    def __getattr__(self, name: str) -> Any:
        obj = getattr(self._proxy, name)
        if callable(obj):
            return self._wrap(name, obj)
        else:
            raise TypeError("Accessing non-functions through .unique is not supported.")

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _make_hashable(self, value: Any) -> Any:
        """Convert unhashable types (e.g., dict) to a hashable representation."""
        if isinstance(value, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in value.items()))
        elif isinstance(value, list):
            return tuple(self._make_hashable(v) for v in value)
        elif isinstance(value, set):
            return frozenset(self._make_hashable(v) for v in value)
        return value

    def _wrap(self, name: str, function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # If types are excluded, call function once to check return type
            if self._excluded_types:
                retval = function(*args, **kwargs)
                # Skip uniqueness check if type is excluded
                if isinstance(retval, self._excluded_types):
                    return retval
                # If not excluded, continue with normal uniqueness logic
                # but we already have a value, so we'll use it if unique
                hashable_retval = self._make_hashable(retval)
                key = (name, args, tuple(sorted(kwargs.items())))
                generated = self._seen.setdefault(key, {self._sentinel})

                # Check if this first value is unique
                if hashable_retval not in generated:
                    generated.add(hashable_retval)
                    return retval
                # Not unique, continue with normal loop below
            else:
                # No exclusions, use original logic
                key = (name, args, tuple(sorted(kwargs.items())))
                generated = self._seen.setdefault(key, {self._sentinel})
                retval = self._sentinel
                hashable_retval = self._make_hashable(retval)

            # Original uniqueness logic (with potential first attempt already done)
            for i in range(_UNIQUE_ATTEMPTS):
                if hashable_retval not in generated:
                    break
                retval = function(*args, **kwargs)
                hashable_retval = self._make_hashable(retval)
            else:
                raise UniquenessException(f"Got duplicated values after {_UNIQUE_ATTEMPTS:,} iterations.")

            generated.add(hashable_retval)

            return retval

        return wrapper


class OptionalProxy:
    """
    Return either a fake value or None, with a customizable probability.
    """

    def __init__(self, proxy: Faker):
        self._proxy = proxy

    def __getattr__(self, name: str) -> Any:
        obj = getattr(self._proxy, name)
        if callable(obj):
            return self._wrap(name, obj)
        else:
            raise TypeError("Accessing non-functions through .optional is not supported.")

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _wrap(self, name: str, function: Callable[..., RetType]) -> Callable[..., RetType | None]:
        @functools.wraps(function)
        def wrapper(*args: Any, prob: float = 0.5, **kwargs: Any) -> RetType | None:
            if not 0 < prob <= 1.0:
                raise ValueError("prob must be between 0 and 1")
            return function(*args, **kwargs) if self._proxy.boolean(chance_of_getting_true=int(prob * 100)) else None

        return wrapper
