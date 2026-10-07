"""Chunked delivery for the structured-output providers.

The one-shot structured producers (``dsv``/``csv``/``tsv``/``psv``,
``json``/``json_bytes`` and ``fixed_width``) build the whole batch in memory
before returning it.  This module adds a second, opt-in delivery pathway:
the caller declares how many rows each chunk holds and how failing rows
should be handled, and output is delivered one chunk at a time through a
:class:`ChunkedProduction` session.

Design properties:

* every delivered :class:`ChunkResult` carries its own completion status;
* a failing value is recorded as a :class:`RowFailure` locating the global
  row number, the column name (or dotted JSON path) and the raw value
  definition; failures accumulate in a retrievable list on the session;
* already delivered chunks are never recomputed -- asking for the next
  chunk always continues at the next chunk boundary;
* all session state (cursor, failure list, per-session lock) lives on the
  instance, so concurrent sessions are fully independent and can never see
  each other's half-built chunks;
* both the number of chunks and the size of the failure list are bounded
  (``ChunkSpec.max_chunks`` / ``ChunkSpec.max_failures``); exceeding either
  limit raises :class:`~faker.exceptions.ChunkCapacityExceeded` explicitly
  instead of silently dropping anything.

This module is intentionally process-local; cross-process recovery is out
of scope.
"""

from __future__ import annotations

import csv as csv_module
import io
import json
import threading

from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from faker.exceptions import ChunkCapacityExceeded, ChunkConfigurationError

#: Formats that can be produced through the chunked pathway.
CHUNKED_FORMATS: Tuple[str, ...] = (
    "dsv",
    "csv",
    "tsv",
    "psv",
    "json",
    "json_bytes",
    "fixed_width",
)

#: Default upper bound on the number of chunks a single session can deliver.
DEFAULT_MAX_CHUNKS = 1000

#: Default upper bound on the number of failures a single session keeps.
DEFAULT_MAX_FAILURES = 10_000


class FailurePolicy(str, Enum):
    """What should happen to a chunk when one of its rows cannot be produced."""

    #: the failing row is omitted from the chunk, production continues
    SKIP = "skip"
    #: the failing cell (tabular formats) or record (JSON) is substituted
    #: with ``ChunkSpec.replacement``
    REPLACE = "replace"
    #: the whole chunk is delivered with status ``INVALIDATED`` and no rows;
    #: the cursor still advances, so the next call starts at the next chunk
    ABORT_CHUNK = "abort_chunk"

    @classmethod
    def normalize(cls, value: Union["FailurePolicy", str]) -> "FailurePolicy":
        """Coerce a policy value, accepting the ``'abort'`` alias."""
        if isinstance(value, FailurePolicy):
            return value
        aliases = {
            "skip": cls.SKIP,
            "replace": cls.REPLACE,
            "abort": cls.ABORT_CHUNK,
            "abort_chunk": cls.ABORT_CHUNK,
        }
        if not isinstance(value, str) or value not in aliases:
            raise ChunkConfigurationError(
                f"Unknown failure policy {value!r}; expected one of " f"'skip', 'replace', 'abort_chunk' (or 'abort')",
            )
        return aliases[value]


class ChunkStatus(str, Enum):
    """Completion status of a delivered chunk."""

    #: every row of the chunk was produced without failure
    COMPLETE = "complete"
    #: the chunk was delivered, but one or more rows were skipped
    COMPLETE_WITH_SKIPS = "complete_with_skips"
    #: the chunk was delivered with one or more substituted values
    COMPLETE_WITH_REPLACEMENTS = "complete_with_replacements"
    #: a row failed under the ``ABORT_CHUNK`` policy; the chunk carries no rows
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class RowFailure:
    """A single value-production failure, fully locatable.

    :ivar row: global, 1-based row number across the whole production
    :ivar chunk_index: 0-based index of the chunk the failure happened in
    :ivar column: column name (DSV header / JSON dotted path); ``None`` when
        the format has no named columns
    :ivar column_index: 0-based position of the column within the row,
        ``None`` when not applicable
    :ivar definition: the raw value definition (e.g. a ``{{token}}`` string)
    :ivar action: how the failure was handled -- ``'skip'``, ``'replace'``
        or ``'abort_chunk'``
    :ivar error: the original exception raised while producing the value
    """

    row: int
    chunk_index: int
    column: Optional[str]
    column_index: Optional[int]
    definition: Any
    action: str
    error: BaseException

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the failure."""
        return {
            "row": self.row,
            "chunk": self.chunk_index,
            "column": self.column,
            "column_index": self.column_index,
            "definition": self.definition if isinstance(self.definition, str) else repr(self.definition),
            "action": self.action,
            "error_type": type(self.error).__name__,
            "error": str(self.error),
        }


@dataclass(frozen=True)
class ChunkResult:
    """One delivered chunk.

    :ivar index: 0-based chunk index
    :ivar status: completion status of this chunk
    :ivar rows: raw rows of this chunk (lists of cells for DSV, records for
        JSON, rendered lines for fixed-width); skipped rows are absent and
        invalidated chunks have an empty list
    :ivar payload: serialized fragment in the declared output format
        (``str`` for text formats, ``bytes`` for ``json_bytes``)
    :ivar failures: failures recorded while building this chunk
    :ivar row_start: global, 1-based number of the first row covered
    :ivar row_end: global, 1-based number of the last row covered
    :ivar last: ``True`` when this is the final chunk of the production
    """

    index: int
    status: ChunkStatus
    rows: List[Any]
    payload: Any
    failures: List[RowFailure]
    row_start: int
    row_end: int
    last: bool


@dataclass(frozen=True)
class ChunkSpec:
    """Declaration of how a chunked production should behave.

    :param rows_per_chunk: maximum number of rows each chunk holds (the last
        chunk may be smaller); must be a positive integer
    :param failure_policy: one of :class:`FailurePolicy` or the strings
        ``'skip'``, ``'replace'``, ``'abort'``/``'abort_chunk'``
    :param replacement: substitute value used under the ``REPLACE`` policy
    :param max_chunks: hard upper bound on the number of delivered chunks;
        a production needing more chunks is rejected up front
    :param max_failures: hard upper bound on the accumulated failure list;
        a chunk that would overflow the list is rejected explicitly
    """

    rows_per_chunk: int
    failure_policy: Union[FailurePolicy, str] = FailurePolicy.SKIP
    replacement: Any = None
    max_chunks: int = DEFAULT_MAX_CHUNKS
    max_failures: int = DEFAULT_MAX_FAILURES

    def __post_init__(self) -> None:
        if not isinstance(self.rows_per_chunk, int) or isinstance(self.rows_per_chunk, bool):
            raise ChunkConfigurationError("`rows_per_chunk` must be a positive integer")
        if self.rows_per_chunk <= 0:
            raise ChunkConfigurationError("`rows_per_chunk` must be a positive integer")
        # normalized enum is cached back into the frozen dataclass
        object.__setattr__(self, "failure_policy", FailurePolicy.normalize(self.failure_policy))
        for option_name, option_value in (
            ("max_chunks", self.max_chunks),
            ("max_failures", self.max_failures),
        ):
            if not isinstance(option_value, int) or isinstance(option_value, bool) or option_value <= 0:
                raise ChunkConfigurationError(f"`{option_name}` must be a positive integer")


class _CellError(Exception):
    """Internal signal carrying the location of a failed JSON leaf value."""

    def __init__(
        self,
        column: Optional[str],
        column_index: Optional[int],
        definition: Any,
        original: BaseException,
    ) -> None:
        super().__init__(str(original))
        self.column = column
        self.column_index = column_index
        self.definition = definition
        self.original = original


def _json_leaf(
    provider: Any,
    definition: Any,
    kwargs: Dict[str, Any],
    path: List[Union[str, int]],
) -> Any:
    """Produce one leaf value through the provider value-selection entry."""
    try:
        return provider._value_format_selection(definition, **kwargs)
    except Exception as exc:  # located and re-raised as a row failure upstream
        column = ".".join(str(part) for part in path) if path else None
        raise _CellError(column, None, definition, exc) from exc


def _build_json_list_structure(provider: Any, data: Sequence[Any], path: List[Union[str, int]]) -> Any:
    """Mirror of the list-format JSON structure builder, with path tracking."""
    entry: Dict[str, Any] = {}

    for name, definition, *arguments in data:
        kwargs = arguments[0] if arguments else {}

        if not isinstance(kwargs, dict):
            raise TypeError("Invalid arguments type. Must be a dictionary")

        if name is None:
            return _json_leaf(provider, definition, kwargs, path)

        column_path = path + [str(name)]

        if isinstance(definition, tuple):
            entry[name] = _build_json_list_structure(provider, definition, column_path)
        elif isinstance(definition, (list, set)):
            entry[name] = [
                _build_json_list_structure(provider, [item], column_path + [position])
                for position, item in enumerate(definition)
            ]
        else:
            entry[name] = _json_leaf(provider, definition, kwargs, column_path)
    return entry


def _build_json_dict_structure(provider: Any, data: Any, path: List[Union[str, int]]) -> Any:
    """Mirror of the dict-format JSON structure builder, with path tracking."""
    if isinstance(data, str):
        return _json_leaf(provider, data, {}, path)

    if isinstance(data, dict):
        entry: Dict[str, Any] = {}
        for name, definition in data.items():
            column_path = path + [str(name)]
            if isinstance(definition, (tuple, list, set)):
                entry[name] = [
                    _build_json_dict_structure(provider, item, column_path + [position])
                    for position, item in enumerate(definition)
                ]
            elif isinstance(definition, (dict, int, float, bool)):
                entry[name] = _build_json_dict_structure(provider, definition, column_path)
            else:
                entry[name] = _json_leaf(provider, definition, {}, column_path)
        return entry

    return data


def create_chunked_production(provider: Any, format_name: str, spec: Any, **config: Any) -> "ChunkedProduction":
    """Validate the chunk declaration and build a :class:`ChunkedProduction`.

    This is the single funnel used by every chunked producer method; it is
    not part of the documented public API (use ``Faker.produce_chunks()`` or
    the ``chunk=`` keyword on a structured producer instead).
    """
    if not isinstance(spec, ChunkSpec):
        raise ChunkConfigurationError(
            "chunked output requires a `ChunkSpec` declaration; build one with "
            "ChunkSpec(rows_per_chunk=..., failure_policy=...)",
        )
    if not isinstance(format_name, str) or format_name not in CHUNKED_FORMATS:
        raise ChunkConfigurationError(f"Unsupported chunked output format {format_name!r}")

    num_rows = config.pop("num_rows", 10)
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or num_rows <= 0:
        raise ChunkConfigurationError("`num_rows` must be a positive integer")

    total_chunks = (num_rows + spec.rows_per_chunk - 1) // spec.rows_per_chunk
    if total_chunks > spec.max_chunks:
        raise ChunkCapacityExceeded(
            f"Chunked production needs {total_chunks} chunks but the limit is "
            f"{spec.max_chunks} (rows_per_chunk={spec.rows_per_chunk}, "
            f"num_rows={num_rows}); raise ChunkSpec.max_chunks to allow it.",
        )

    if format_name in ("dsv", "csv", "tsv", "psv"):
        data_columns = config["data_columns"]
        if not isinstance(data_columns, (list, tuple)):
            raise TypeError("`data_columns` must be a tuple or a list")
        header = config.get("header")
        if header is not None:
            if not isinstance(header, (list, tuple)):
                raise TypeError("`header` must be a tuple or a list")
            if len(header) != len(data_columns):
                raise ValueError("`header` and `data_columns` must have matching lengths")
    elif format_name in ("json", "json_bytes"):
        data_columns = config.get("data_columns")
        if data_columns is None:
            data_columns = {
                "name": "{{name}}",
                "residency": "{{address}}",
            }
        elif not isinstance(data_columns, (dict, list)):
            raise TypeError("Invalid data_columns type. Must be a dictionary or list")
        config["data_columns"] = data_columns
    elif format_name == "fixed_width":
        if config.get("data_columns") is None:
            config["data_columns"] = [
                (20, "name"),
                (3, "pyint", {"max_value": 20}),
            ]

    return ChunkedProduction(provider, format_name, spec, num_rows, **config)


class ChunkedProduction:
    """A resumable, chunk-by-chunk structured-output session.

    Instances are normally created through ``Faker.produce_chunks(...)`` or
    by passing ``chunk=ChunkSpec(...)`` to one of the structured producer
    methods.  Iterate the session (or call :meth:`next_chunk` repeatedly) to
    receive :class:`ChunkResult` objects; :attr:`failures` exposes the
    accumulated failure list.  Sessions are independent and safe to use from
    different threads concurrently; concurrent calls on the *same* session
    are serialized so its cursor cannot skip or duplicate a chunk.
    """

    def __init__(
        self,
        provider: Any,
        format_name: str,
        spec: ChunkSpec,
        num_rows: int,
        **config: Any,
    ) -> None:
        self._provider = provider
        self._format_name = format_name
        self._spec = spec
        self._num_rows = num_rows
        self._config = config

        self._lock = threading.Lock()
        self._cursor = 0  # number of rows already handed over (chunk boundaries consumed)
        self._chunk_index = 0
        self._failures: List[RowFailure] = []

    # ------------------------------------------------------------------ #
    # public surface
    # ------------------------------------------------------------------ #

    @property
    def format_name(self) -> str:
        return self._format_name

    @property
    def spec(self) -> ChunkSpec:
        return self._spec

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def total_chunks(self) -> int:
        return (self._num_rows + self._spec.rows_per_chunk - 1) // self._spec.rows_per_chunk

    @property
    def delivered_chunks(self) -> int:
        """Number of chunks already handed over (never recomputed)."""
        return self._chunk_index

    @property
    def is_exhausted(self) -> bool:
        return self._cursor >= self._num_rows

    @property
    def failures(self) -> List[RowFailure]:
        """Copy of all failures accumulated so far (safe to mutate)."""
        with self._lock:
            return list(self._failures)

    @property
    def _policy(self) -> FailurePolicy:
        return FailurePolicy.normalize(self._spec.failure_policy)

    def next_chunk(self) -> ChunkResult:
        """Produce and deliver the next chunk.

        Raises :class:`StopIteration` once every chunk has been delivered,
        and :class:`~faker.exceptions.ChunkCapacityExceeded` if recording a
        chunk's failures would overflow ``ChunkSpec.max_failures``.
        """
        with self._lock:
            if self._cursor >= self._num_rows:
                raise StopIteration("All chunks have already been delivered")

            index = self._chunk_index
            row_start = self._cursor + 1
            row_end = min(self._cursor + self._spec.rows_per_chunk, self._num_rows)

            rows: List[Any]
            payload: Any
            chunk_failures: List[RowFailure]
            status: ChunkStatus
            if self._format_name in ("dsv", "csv", "tsv", "psv"):
                rows, payload, chunk_failures, status = self._build_dsv_chunk(index, row_start, row_end)
            elif self._format_name in ("json", "json_bytes"):
                rows, payload, chunk_failures, status = self._build_json_chunk(index, row_start, row_end)
            else:
                rows, payload, chunk_failures, status = self._build_fixed_width_chunk(index, row_start, row_end)

            if len(self._failures) + len(chunk_failures) > self._spec.max_failures:
                raise ChunkCapacityExceeded(
                    f"Recording {len(chunk_failures)} new failure(s) at chunk {index} "
                    f"would exceed the failure list limit of {self._spec.max_failures} "
                    f"(already recorded: {len(self._failures)}); the chunk was not "
                    f"delivered. Raise ChunkSpec.max_failures to allow it.",
                )

            self._failures.extend(chunk_failures)
            self._cursor = row_end
            self._chunk_index += 1

            return ChunkResult(
                index=index,
                status=status,
                rows=rows,
                payload=payload,
                failures=chunk_failures,
                row_start=row_start,
                row_end=row_end,
                last=row_end >= self._num_rows,
            )

    def close(self) -> None:
        """Mark the session as finished; further chunks cannot be requested."""
        with self._lock:
            self._cursor = self._num_rows

    def __iter__(self) -> "ChunkedProduction":
        return self

    def __next__(self) -> ChunkResult:
        return self.next_chunk()

    def __enter__(self) -> "ChunkedProduction":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # per-format chunk builders (all state stays in local variables until
    # the chunk is fully built, so nothing half-built is ever observable)
    # ------------------------------------------------------------------ #

    def _make_failure(
        self,
        global_row: int,
        chunk_index: int,
        column: Optional[str],
        column_index: Optional[int],
        definition: Any,
        error: BaseException,
    ) -> RowFailure:
        return RowFailure(
            row=global_row,
            chunk_index=chunk_index,
            column=column,
            column_index=column_index,
            definition=definition,
            action=self._policy.value,
            error=error,
        )

    def _status(self, failures: Sequence[RowFailure]) -> ChunkStatus:
        if not failures:
            return ChunkStatus.COMPLETE
        if self._policy is FailurePolicy.SKIP:
            return ChunkStatus.COMPLETE_WITH_SKIPS
        if self._policy is FailurePolicy.REPLACE:
            return ChunkStatus.COMPLETE_WITH_REPLACEMENTS
        return ChunkStatus.INVALIDATED

    def _build_dsv_chunk(
        self,
        index: int,
        row_start: int,
        row_end: int,
    ) -> Tuple[List[List[Any]], str, List[RowFailure], ChunkStatus]:
        config = self._config
        data_columns: Sequence[str] = config["data_columns"]
        header: Optional[Sequence[str]] = config.get("header")
        include_row_ids: bool = config.get("include_row_ids", False)
        dialect: str = config.get("dialect", "faker-csv")
        fmtparams: Dict[str, Any] = config.get("fmtparams", {}) or {}

        buffer = io.StringIO()
        writer = csv_module.writer(buffer, dialect=dialect, **fmtparams)

        # the header belongs to the first chunk only
        if index == 0 and header:
            header_row = list(header)
            if include_row_ids:
                header_row.insert(0, "ID")
            writer.writerow(header_row)

        rows: List[List[Any]] = []
        failures: List[RowFailure] = []
        policy = self._policy

        for global_row in range(row_start, row_end + 1):
            cells: List[Any] = []
            row_failed = False

            for column_index, definition in enumerate(data_columns):
                try:
                    value = self._provider.generator.pystr_format(definition)
                except Exception as exc:
                    column_name = header[column_index] if header else None
                    failures.append(
                        self._make_failure(
                            global_row,
                            index,
                            column_name,
                            column_index,
                            definition,
                            exc,
                        ),
                    )
                    row_failed = True
                    if policy is FailurePolicy.ABORT_CHUNK:
                        break
                    if policy is FailurePolicy.SKIP:
                        break
                    value = self._spec.replacement
                cells.append(value)

            if row_failed:
                if policy is FailurePolicy.ABORT_CHUNK:
                    break
                if policy is FailurePolicy.SKIP:
                    continue

            if include_row_ids:
                cells.insert(0, str(global_row))
            rows.append(cells)
            writer.writerow(cells)

        return rows, buffer.getvalue(), failures, self._status(failures)

    def _build_json_chunk(
        self,
        index: int,
        row_start: int,
        row_end: int,
    ) -> Tuple[List[Any], Any, List[RowFailure], ChunkStatus]:
        config = self._config
        data_columns = config["data_columns"]
        indent = config.get("indent")
        encoder_cls = config.get("cls")

        records: List[Any] = []
        failures: List[RowFailure] = []
        policy = self._policy

        for global_row in range(row_start, row_end + 1):
            try:
                if isinstance(data_columns, dict):
                    record = _build_json_dict_structure(self._provider, data_columns, [])
                else:
                    record = _build_json_list_structure(self._provider, data_columns, [])
            except _CellError as cell_error:
                failures.append(
                    self._make_failure(
                        global_row,
                        index,
                        cell_error.column,
                        cell_error.column_index,
                        cell_error.definition,
                        cell_error.original,
                    ),
                )
                if policy is FailurePolicy.ABORT_CHUNK:
                    break
                if policy is FailurePolicy.SKIP:
                    continue
                record = self._spec.replacement
            records.append(record)

        payload: Any = json.dumps(records, indent=indent, cls=encoder_cls)
        if self._format_name == "json_bytes":
            payload = payload.encode()

        return records, payload, failures, self._status(failures)

    def _build_fixed_width_chunk(
        self,
        index: int,
        row_start: int,
        row_end: int,
    ) -> Tuple[List[str], str, List[RowFailure], ChunkStatus]:
        config = self._config
        data_columns = config["data_columns"]
        align_map = {"left": "<", "middle": "^", "right": ">"}
        align = align_map.get(config.get("align", "left"), "<")

        lines: List[str] = []
        failures: List[RowFailure] = []
        policy = self._policy

        for global_row in range(row_start, row_end + 1):
            pieces: List[str] = []
            row_failed = False

            for column_index, column_spec in enumerate(data_columns):
                width, definition, *arguments = column_spec
                kwargs = arguments[0] if arguments else {}

                if not isinstance(kwargs, dict):
                    raise TypeError("Invalid arguments type. Must be a dictionary")

                try:
                    result = self._provider._value_format_selection(definition, **kwargs)
                except Exception as exc:
                    failures.append(
                        self._make_failure(global_row, index, None, column_index, definition, exc),
                    )
                    row_failed = True
                    if policy is FailurePolicy.ABORT_CHUNK:
                        break
                    if policy is FailurePolicy.SKIP:
                        break
                    result = self._spec.replacement
                pieces.append(f"{result:{align}{width}}"[:width])

            if row_failed:
                if policy is FailurePolicy.ABORT_CHUNK:
                    break
                if policy is FailurePolicy.SKIP:
                    continue

            lines.append("".join(pieces))

        return lines, "\n".join(lines), failures, self._status(failures)
