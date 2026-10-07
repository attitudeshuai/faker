"""Codec for explicit generator-state snapshots.

A snapshot is a portable, JSON-compatible dictionary describing everything
that determines what a :class:`~faker.Faker` instance will generate next:

* the position (full internal state) of every ``random.Random`` object,
* the ``.unique`` value history,
* the argument groups configured on each generator,
* the locale weights and the provider-method selection cache,
* descriptions of the providers registered at runtime.

The snapshot carries an explicit version marker
(:data:`SNAPSHOT_VERSION`). All structures use only primitive JSON types so
that a snapshot can be written to a file in one process and applied in
another without pickling any provider class.
"""

from __future__ import annotations

import base64
import importlib

from typing import Any

from .exceptions import IncompatibleSnapshotError, UnrepresentableValueError

#: Version of the snapshot format produced and accepted by this code.
SNAPSHOT_VERSION = 1

# Marker keys used to tag otherwise ambiguous container types.
_TYPE = "__faker_snapshot_type__"
_ITEMS = "items"


def faker_version() -> str | None:
    """Returns the Faker package version recorded in a snapshot (diagnostic
    only; never used to accept or reject a snapshot)."""
    try:
        from . import VERSION

        return VERSION
    except ImportError:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Generic portable value encoding
# ---------------------------------------------------------------------------


def encode_value(value: Any, *, path: str = "value") -> Any:
    """Encode ``value`` into JSON-compatible primitives.

    Tuples, sets, frozensets, dictionaries, and bytes are explicitly tagged
    so that they round-trip exactly. Anything that cannot be represented
    portably raises :class:`UnrepresentableValueError` pointed at ``path``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {
            _TYPE: "tuple",
            _ITEMS: [encode_value(v, path=f"{path}[{i}]") for i, v in enumerate(value)],
        }
    if isinstance(value, list):
        return {
            _TYPE: "list",
            _ITEMS: [encode_value(v, path=f"{path}[{i}]") for i, v in enumerate(value)],
        }
    if isinstance(value, (set, frozenset)):
        items = [encode_value(v, path=f"{path} element {v!r}") for v in value]
        # Sort deterministically (by repr) so equal sets encode identically
        # regardless of insertion/iteration order.
        items.sort(key=repr)
        return {_TYPE: "set", _ITEMS: items}
    if isinstance(value, dict):
        items = [
            [
                encode_value(k, path=f"{path} key {k!r}"),
                encode_value(v, path=f"{path}[{k!r}]"),
            ]
            for k, v in value.items()
        ]
        items.sort(key=lambda pair: repr(pair[0]))
        return {_TYPE: "dict", _ITEMS: items}
    if isinstance(value, (bytes, bytearray)):
        return {_TYPE: "bytes", _ITEMS: base64.b64encode(bytes(value)).decode("ascii")}
    raise UnrepresentableValueError(
        f"Cannot encode {path}: object of type {type(value).__name__!r} is not "
        f"representable in a generator-state snapshot"
    )


def decode_value(data: Any) -> Any:
    """Reverse :func:`encode_value`."""
    if isinstance(data, dict) and _TYPE in data:
        tagged_type = data[_TYPE]
        if tagged_type in ("tuple", "list", "set"):
            items = [decode_value(v) for v in data[_ITEMS]]
            if tagged_type == "tuple":
                return tuple(items)
            if tagged_type == "set":
                return set(items)
            return items
        if tagged_type == "dict":
            return {decode_value(k): decode_value(v) for k, v in data[_ITEMS]}
        if tagged_type == "bytes":
            return base64.b64decode(data[_ITEMS].encode("ascii"))
        raise IncompatibleSnapshotError(f"Unknown encoded value type {tagged_type!r}")
    return data


# ---------------------------------------------------------------------------
# Random source position
# ---------------------------------------------------------------------------


def encode_random_state(state: Any) -> list:
    """Encode a ``random.Random.getstate()`` tuple."""
    try:
        version, internal_state, gauss_next = state
        return [version, list(internal_state), gauss_next]
    except (TypeError, ValueError) as exc:
        raise IncompatibleSnapshotError(f"Malformed random state: {state!r}") from exc


def decode_random_state(data: Any) -> tuple:
    """Decode data produced by :func:`encode_random_state` back into a tuple
    suitable for ``random.Random.setstate()``."""
    try:
        version, internal_state, gauss_next = data
        return version, tuple(internal_state), gauss_next
    except (TypeError, ValueError) as exc:
        raise IncompatibleSnapshotError(f"Malformed random state in snapshot: {data!r}") from exc


# ---------------------------------------------------------------------------
# Provider class coordinates
# ---------------------------------------------------------------------------


def class_coords(cls: type) -> tuple[str, str]:
    """Returns the ``(module, qualified name)`` coordinates of a class."""
    return cls.__module__, cls.__qualname__


def is_representable_class(cls: type) -> bool:
    """A provider class is portable only if it can be imported in another
    process via a stable module path. Classes defined in ``__main__`` (scripts
    and notebooks), inside functions or other local scopes
    (``<locals>`` in the qualified name) are rejected.
    """
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    if not module or not qualname:
        return False
    if module == "__main__":
        return False
    if "<locals>" in qualname:
        return False
    return True


def resolve_class(module: str, qualname: str) -> type:
    """Import and return the class located at ``module.qualname``."""
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def provider_methods(provider: Any) -> list[str]:
    """Public callable method names exposed by a provider instance (used to
    describe and diagnose runtime providers)."""
    names = set()
    for name in dir(provider):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(provider, name)
        except Exception:  # noqa: BLE001 - broken provider attribute, skip
            continue
        if callable(attr):
            names.add(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# Unique value history
# ---------------------------------------------------------------------------


def encode_seen(seen: dict, sentinel: Any) -> list:
    """Encode the ``UniqueProxy._seen`` dictionary, filtering out the private
    per-proxy sentinel object.

    Returns a deterministic list of ``[encoded_key, encoded_value_set]``
    pairs.
    """
    entries = []
    for key, values in seen.items():
        encoded_key = encode_value(key, path=f"unique key {key!r}")
        encoded_values = [encode_value(v, path=f"unique value {v!r}") for v in values if v is not sentinel]
        encoded_values.sort(key=repr)
        entries.append([encoded_key, {_TYPE: "set", _ITEMS: encoded_values}])
    entries.sort(key=lambda pair: repr(pair[0]))
    return entries


def decode_seen(data: Any) -> dict:
    """Decode data produced by :func:`encode_seen` into a plain dictionary
    of ``key -> set``. The caller is responsible for adding its own fresh
    sentinel to every set.
    """
    if not isinstance(data, list):
        raise IncompatibleSnapshotError("Malformed unique history in snapshot")
    result = {}
    for entry in data:
        try:
            encoded_key, encoded_values = entry
        except (TypeError, ValueError) as exc:
            raise IncompatibleSnapshotError(f"Malformed unique history entry: {entry!r}") from exc
        result[decode_value(encoded_key)] = decode_value(encoded_values)
    return result
