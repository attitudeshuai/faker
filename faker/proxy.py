from __future__ import annotations

import copy
import functools
import re
import threading

from collections import OrderedDict
from random import Random
from typing import Any, Callable, Pattern, Sequence, TypeVar

from .config import DEFAULT_LOCALE
from .exceptions import IncompatibleSnapshotError, UniquenessException
from .factory import Factory
from .generator import Generator
from .snapshot import (
    SNAPSHOT_VERSION,
    decode_seen,
    encode_seen,
    faker_version,
)
from .typing import SeedType
from .utils.distribution import choices_distribution

_UNIQUE_ATTEMPTS = 1000

RetType = TypeVar("RetType")


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
        **config: Any,
    ) -> None:
        self._factory_map: OrderedDict[str, Generator | Faker] = OrderedDict()
        self._weights = None
        self._state_lock = threading.RLock()
        self._unique_proxy = UniqueProxy(self)
        self._optional_proxy = OptionalProxy(self)

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
            assert all(isinstance(v, (int, float)) for v in locale.values())
            odict = OrderedDict()
            for k, v in locale.items():
                key = k.replace("-", "_")
                odict[key] = v
            locales = list(odict.keys())
            self._weights = list(odict.values())

        else:
            locales = [DEFAULT_LOCALE]

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
                    **config,
                )

        self._locales = locales
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
        result._state_lock = threading.RLock()
        result._locales = copy.deepcopy(self._locales, memodict)
        result._factory_map = copy.deepcopy(self._factory_map, memodict)
        result._factories = list(result._factory_map.values())
        result._weights = copy.deepcopy(self._weights, memodict)
        result._unique_proxy = UniqueProxy(result)
        result._unique_proxy._seen = {k: {result._unique_proxy._sentinel} for k in self._unique_proxy._seen.keys()}
        result._optional_proxy = OptionalProxy(result)
        return result

    def __getstate__(self) -> dict:
        # Locks are not picklable and are recreated on unpickle.
        return {key: value for key, value in self.__dict__.items() if key != "_state_lock"}

    def __setstate__(self, state: Any) -> None:
        self.__dict__.update(state)
        self._state_lock = threading.RLock()

    @property
    def unique(self) -> UniqueProxy:
        return self._unique_proxy

    @property
    def optional(self) -> OptionalProxy:
        return self._optional_proxy

    def _select_factory(self, method_name: str) -> Factory:
        """
        Returns a random factory that supports the provider method

        :param method_name: Name of provider method
        :return: A factory that supports the provider method
        """

        factories, weights = self._map_provider_method(method_name)

        if len(factories) == 0:
            msg = f"No generator object has attribute {method_name!r}"
            raise AttributeError(msg)
        elif len(factories) == 1:
            return factories[0]

        if weights:
            factory = self._select_factory_distribution(factories, weights)
        else:
            factory = self._select_factory_choice(factories)
        return factory

    def _select_factory_distribution(self, factories, weights):
        return choices_distribution(factories, weights, self.factories[0].random, length=1)[0]

    def _select_factory_choice(self, factories):
        return self._factories[0].random.choice(factories)

    def _map_provider_method(self, method_name: str) -> tuple[list[Factory], list[float] | None]:
        """
        Creates a 2-tuple of factories and weights for the given provider method name

        The first element of the tuple contains a list of compatible factories.
        The second element of the tuple contains a list of distribution weights.

        :param method_name: Name of provider method
        :return: 2-tuple (factories, weights)
        """

        # Return cached mapping if it exists for given method
        attr = f"_cached_{method_name}_mapping"
        if hasattr(self, attr):
            return getattr(self, attr)

        # Create mapping if it does not exist
        if self._weights:
            value = [
                (factory, weight)
                for factory, weight in zip(self.factories, self._weights)
                if hasattr(factory, method_name)
            ]
            factories, weights = zip(*value)
            mapping = list(factories), list(weights)
        else:
            value = [factory for factory in self.factories if hasattr(factory, method_name)]  # type: ignore
            mapping = value, None  # type: ignore

        # Then cache and return results
        setattr(self, attr, mapping)
        return mapping

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
        return self._weights

    @property
    def factories(self) -> list[Generator | Faker]:
        return self._factories

    def items(self) -> list[tuple[str, Generator | Faker]]:
        return list(self._factory_map.items())

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Export this Faker's full generator state as a portable,
        JSON-compatible dictionary.

        The snapshot covers, for every locale: the random source position,
        the argument groups, and descriptions of all runtime-registered
        providers, plus this proxy's locale weights, provider-method
        selection cache, and ``.unique`` value history.

        The export takes this instance's state lock and every child
        factory's lock, so a concurrent :meth:`restore` can only be
        observed fully applied or not applied at all -- never a mixture.
        """
        with self._state_lock:
            child_locks = self._acquire_child_locks()
            try:
                return self._snapshot_locked()
            finally:
                self._release_child_locks(child_locks)

    def restore(self, snapshot: dict) -> None:
        """Replace this Faker's generator state with ``snapshot``.

        The whole snapshot (version marker, locale layout, built-in
        providers, custom provider classes, unique history, caches) is
        decoded and validated, and all runtime provider objects are
        constructed, before any live state changes. A rejected import
        therefore raises without leaving a partially restored instance.

        The target instance must have the same locale list as the
        snapshot; use :meth:`from_snapshot` to build a new instance
        instead.
        """
        with self._state_lock:
            child_locks = self._acquire_child_locks()
            try:
                plan = self._plan_restore(snapshot)
                self._publish_restore(plan)
            finally:
                self._release_child_locks(child_locks)

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> Faker:
        """Build a new Faker instance from ``snapshot``.

        All factories are constructed from the recorded built-in provider
        lists and the runtime providers are then re-registered, so the new
        instance continues the same random sequence from the recorded
        position even in a different process.
        """
        if not isinstance(snapshot, dict):
            raise IncompatibleSnapshotError("Snapshot must be a dictionary")
        if snapshot.get("snapshot_version") != SNAPSHOT_VERSION:
            raise IncompatibleSnapshotError(
                f"Unsupported snapshot version {snapshot.get('snapshot_version')!r}; " f"expected {SNAPSHOT_VERSION}"
            )
        if snapshot.get("object", "faker") != "faker":
            raise IncompatibleSnapshotError(f"Expected a Faker snapshot, got {snapshot.get('object')!r}")

        locales = snapshot.get("locales")
        if not isinstance(locales, list) or not locales or not all(isinstance(code, str) for code in locales):
            raise IncompatibleSnapshotError("Malformed 'locales' section in snapshot")

        weights = snapshot.get("weights")
        if weights is not None and (not isinstance(weights, list) or len(weights) != len(locales)):
            raise IncompatibleSnapshotError("Malformed 'weights' section in snapshot")

        specs = cls._collect_builtin_specs(snapshot)
        use_weighting = cls._collect_use_weighting(snapshot)

        if weights:
            locale_arg: Any = OrderedDict(zip(locales, weights))
        elif len(locales) == 1:
            locale_arg = locales[0]
        else:
            locale_arg = list(locales)

        primary_spec = specs[locales[0]]
        instance = cls(locale_arg, providers=list(primary_spec), use_weighting=use_weighting)

        # Children with a different built-in provider set are rebuilt.
        for locale in locales[1:]:
            if specs[locale] != primary_spec:
                instance._factory_map[locale] = cls(locale, providers=list(specs[locale]), use_weighting=use_weighting)
        instance._factories = list(instance._factory_map.values())

        instance.restore(snapshot)
        return instance

    @classmethod
    def _collect_builtin_specs(cls, data: dict) -> dict:
        specs = {}
        for locale, child_data in data["factories"].items():
            leaf = cls._generator_data(child_data)
            spec = leaf.get("builtin_spec")
            if not isinstance(spec, list) or not all(isinstance(path, str) for path in spec):
                raise IncompatibleSnapshotError(f"Malformed built-in provider spec for locale {locale!r}")
            specs[locale] = spec
        return specs

    @classmethod
    def _collect_use_weighting(cls, data: dict) -> bool:
        return bool(cls._generator_data(data).get("use_weighting", True))

    @classmethod
    def _generator_data(cls, data: dict) -> dict:
        """Walk nested Faker snapshots until the leaf generator data."""
        if data.get("object") == "generator":
            return data
        factories = data["factories"]
        first_key = next(iter(factories))
        return cls._generator_data(factories[first_key])

    def _snapshot_locked(self) -> dict:
        factories_data: OrderedDict[str, Any] = OrderedDict()
        for locale, factory in self._factory_map.items():
            if isinstance(factory, Faker):
                factories_data[locale] = factory._snapshot_locked()
            else:
                factories_data[locale] = factory._build_snapshot()

        leaf = self._generator_data(factories_data[self._locales[0]])
        return {
            "object": "faker",
            "snapshot_version": SNAPSHOT_VERSION,
            "faker_version": faker_version(),
            "locales": list(self._locales),
            "weights": list(self._weights) if self._weights else None,
            "use_weighting": leaf.get("use_weighting", True),
            "factories": factories_data,
            "unique": encode_seen(self._unique_proxy._seen, self._unique_proxy._sentinel),
            "caches": self._export_caches(),
        }

    def _plan_restore(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise IncompatibleSnapshotError("Snapshot must be a dictionary")
        if data.get("snapshot_version") != SNAPSHOT_VERSION:
            raise IncompatibleSnapshotError(
                f"Unsupported snapshot version {data.get('snapshot_version')!r}; " f"expected {SNAPSHOT_VERSION}"
            )
        if data.get("object", "faker") != "faker":
            raise IncompatibleSnapshotError(f"Expected a Faker snapshot, got {data.get('object')!r}")

        locales = data.get("locales")
        if locales != self._locales:
            raise IncompatibleSnapshotError(
                f"Snapshot locales {locales!r} do not match target locales {self._locales!r}"
            )

        weights = data.get("weights")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(locales):
                raise IncompatibleSnapshotError("Malformed 'weights' section in snapshot")
            weights = list(weights)

        factories_data = data.get("factories")
        if not isinstance(factories_data, dict) or set(factories_data) != set(locales):
            raise IncompatibleSnapshotError("Malformed 'factories' section in snapshot")

        child_plans = []
        for locale in locales:
            child = self._factory_map[locale]
            child_data = factories_data[locale]
            if isinstance(child, Faker):
                if child_data.get("object", "faker") != "faker":
                    raise IncompatibleSnapshotError(f"Expected a Faker snapshot for locale {locale!r}")
                child_plan = child._plan_restore(child_data)
            else:
                if child_data.get("object") != "generator":
                    raise IncompatibleSnapshotError(f"Expected a generator snapshot for locale {locale!r}")
                child_plan = child._plan_restore(child_data)
            child_plans.append((locale, child_plan))

        seen = decode_seen(data.get("unique", []))
        cache_plan = self._plan_caches(data.get("caches", []))

        return {
            "weights": weights,
            "factories": child_plans,
            "seen": seen,
            "caches": cache_plan,
        }

    def _publish_restore(self, plan: dict) -> None:
        # Restore child factories first while all their locks are held.
        for locale, child_plan in plan["factories"]:
            self._factory_map[locale]._publish_restore(child_plan)

        self._weights = plan["weights"]

        # Replace the unique history atomically, tagging every pool with this
        # proxy's current sentinel.
        seen = plan["seen"]
        sentinel = self._unique_proxy._sentinel
        for values in seen.values():
            values.add(sentinel)
        self._unique_proxy._seen = seen

        # Reset the selection cache to exactly what the snapshot describes so
        # stale cached mappings never survive an import.
        for attr in list(self.__dict__):
            if self.cache_pattern.match(attr):
                del self.__dict__[attr]
        for entry in plan["caches"]:
            compatible_factories = [self._factory_map[locale] for locale in entry["factories"]]
            setattr(
                self,
                f"_cached_{entry['method']}_mapping",
                (compatible_factories, entry["weights"]),
            )

        self._factories = list(self._factory_map.values())

    # -- selection cache ------------------------------------------------

    def _export_caches(self) -> list:
        entries = []
        for attr, value in self.__dict__.items():
            if not self.cache_pattern.match(attr):
                continue
            method_name = attr[len("_cached_") : -len("_mapping")]
            compatible_factories, entry_weights = value
            factory_locales = [
                locale for locale, factory in self._factory_map.items() if factory in compatible_factories
            ]
            entries.append(
                {
                    "method": method_name,
                    "factories": factory_locales,
                    "weights": list(entry_weights) if entry_weights else None,
                }
            )
        entries.sort(key=lambda entry: entry["method"])
        return entries

    def _plan_caches(self, data: Any) -> list:
        if not isinstance(data, list):
            raise IncompatibleSnapshotError("Malformed 'caches' section in snapshot")
        planned = []
        seen_methods = set()
        for entry in data:
            if not isinstance(entry, dict):
                raise IncompatibleSnapshotError(f"Malformed cache entry: {entry!r}")
            method_name = entry.get("method")
            factory_locales = entry.get("factories")
            entry_weights = entry.get("weights")
            if not isinstance(method_name, str) or not isinstance(factory_locales, list):
                raise IncompatibleSnapshotError(f"Malformed cache entry: {entry!r}")
            if method_name in seen_methods:
                raise IncompatibleSnapshotError(f"Duplicate cache entry for {method_name!r}")
            seen_methods.add(method_name)
            for locale in factory_locales:
                if locale not in self._factory_map:
                    raise IncompatibleSnapshotError(f"Cache for {method_name!r} references unknown locale {locale!r}")
                if not hasattr(self._factory_map[locale], method_name):
                    raise IncompatibleSnapshotError(
                        f"Cache for {method_name!r} references locale {locale!r}, which "
                        f"does not provide that method"
                    )
            if entry_weights is not None and len(entry_weights) != len(factory_locales):
                raise IncompatibleSnapshotError(f"Cache for {method_name!r} has mismatched weights")
            planned.append(
                {
                    "method": method_name,
                    "factories": list(factory_locales),
                    "weights": (list(entry_weights) if entry_weights is not None else None),
                }
            )
        return planned

    # -- child lock handling --------------------------------------------

    def _acquire_child_locks(self) -> list:
        locks = []
        try:
            for factory in self._factory_map.values():
                lock = factory._state_lock
                lock.acquire()
                locks.append(lock)
        except BaseException:
            for lock in reversed(locks):
                lock.release()
            raise
        return locks

    def _release_child_locks(self, locks: list) -> None:
        for lock in reversed(locks):
            lock.release()


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
            # Hold the proxy's state lock while generating and recording the
            # value, so a concurrent snapshot/restore observes either the
            # complete old history or the complete new history.
            with self._proxy._state_lock:
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
