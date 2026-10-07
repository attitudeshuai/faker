import csv
import io
import json
import threading

from unittest.mock import patch

import pytest

from faker import ChunkSpec, ChunkStatus, FailurePolicy, Faker
from faker.exceptions import ChunkCapacityExceeded, ChunkConfigurationError
from faker.providers.misc import Provider as MiscProvider

BAD_COLUMN = "{{no_such_formatter}}"


class TestChunkedProduction:
    """Tests for the chunked structured-output pathway."""

    # ------------------------------------------------------------------ #
    # rubric 1: declared rows-per-chunk, chunked delivery, chunk statuses
    # ------------------------------------------------------------------ #

    def test_csv_delivered_in_chunks_with_sizes_and_boundaries(self, faker):
        faker.seed_instance(1)
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=10),
            data_columns=("{{name}}", "{{city}}"),
            num_rows=25,
            header=("Name", "City"),
        )

        chunks = list(session)

        assert session.delivered_chunks == 3
        assert session.total_chunks == 3
        assert [(c.row_start, c.row_end) for c in chunks] == [(1, 10), (11, 20), (21, 25)]
        assert [len(c.rows) for c in chunks] == [10, 10, 5]
        assert [c.status for c in chunks] == [ChunkStatus.COMPLETE] * 3
        assert [c.last for c in chunks] == [False, False, True]
        assert [c.index for c in chunks] == [0, 1, 2]

        # header is emitted exactly once, in the first chunk payload
        assert chunks[0].payload.splitlines()[0] == '"Name","City"'
        assert "Name" not in chunks[1].payload
        assert "Name" not in chunks[2].payload

        # concatenated payloads parse as one valid CSV with header + 25 rows
        reader = list(csv.reader(io.StringIO("".join(c.payload for c in chunks))))
        assert reader[0] == ["Name", "City"]
        assert len(reader) == 26

    def test_chunked_dsv_respects_row_ids_and_dialect(self, faker):
        session = faker.produce_chunks(
            "psv",
            ChunkSpec(rows_per_chunk=3),
            data_columns=("{{name}}", "{{city}}"),
            num_rows=7,
            include_row_ids=True,
        )
        chunks = list(session)
        assert [len(c.rows) for c in chunks] == [3, 3, 1]
        for row in chunks[1].rows:
            assert len(row) == 3
            assert int(row[0]) >= 4  # global row ids continue across chunks
        assert "|" in chunks[0].payload

    def test_json_and_fixed_width_chunks(self, faker):
        session = faker.produce_chunks(
            "json",
            ChunkSpec(rows_per_chunk=2),
            data_columns={"name": "name"},
            num_rows=5,
        )
        chunks = list(session)
        assert [len(c.rows) for c in chunks] == [2, 2, 1]
        for chunk in chunks:
            assert isinstance(chunk.payload, str)
            parsed = json.loads(chunk.payload)
            assert isinstance(parsed, list)
            assert all("name" in record for record in parsed)

        bytes_session = faker.produce_chunks(
            "json_bytes",
            ChunkSpec(rows_per_chunk=4),
            data_columns={"name": "name"},
            num_rows=4,
        )
        first = bytes_session.next_chunk()
        assert isinstance(first.payload, bytes)
        assert json.loads(first.payload.decode())

        fw_session = faker.produce_chunks(
            "fixed_width",
            ChunkSpec(rows_per_chunk=3),
            data_columns=[(5, "name"), (3, "pyint")],
            num_rows=7,
        )
        fw_chunks = list(fw_session)
        assert [len(c.rows) for c in fw_chunks] == [3, 3, 1]
        for chunk in fw_chunks:
            for line in chunk.payload.splitlines():
                assert len(line) == 8

    def test_inline_chunk_declaration(self, faker):
        session = faker.produce_chunks(
            "csv",
            rows_per_chunk=5,
            failure_policy="skip",
            data_columns=("{{name}}",),
            num_rows=5,
        )
        assert isinstance(session.spec, ChunkSpec)
        assert session.spec.failure_policy is FailurePolicy.SKIP
        assert len(list(session)) == 1

    def test_chunk_keyword_on_existing_producers(self, faker):
        spec = ChunkSpec(rows_per_chunk=2)
        for producer, payload_type in (
            (faker.csv, str),
            (faker.tsv, str),
            (faker.psv, str),
            (faker.json, str),
            (faker.json_bytes, bytes),
            (faker.fixed_width, str),
        ):
            session = producer(chunk=spec, num_rows=4)
            chunk = session.next_chunk()
            assert isinstance(chunk.payload, payload_type)
            assert len(chunk.rows) == 2

    # ------------------------------------------------------------------ #
    # rubric 2: per-row failure policies and failure location
    # ------------------------------------------------------------------ #

    def test_skip_policy_omits_rows_and_locates_failure(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=3, failure_policy="skip"),
            data_columns=("{{name}}", BAD_COLUMN),
            num_rows=6,
            header=("Name", "Boom"),
        )
        chunks = list(session)

        assert all(c.status is ChunkStatus.COMPLETE_WITH_SKIPS for c in chunks)
        assert all(len(c.rows) == 0 for c in chunks)

        failures = session.failures
        assert [f.row for f in failures] == [1, 2, 3, 4, 5, 6]
        assert {f.column for f in failures} == {"Boom"}
        assert {f.column_index for f in failures} == {1}
        assert {f.definition for f in failures} == {BAD_COLUMN}
        assert {f.action for f in failures} == {"skip"}
        assert {f.chunk_index for f in failures} == {0, 1}
        assert all(isinstance(f.error, AttributeError) for f in failures)

    def test_skip_policy_without_header_reports_definition_as_context(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=2, failure_policy="skip"),
            data_columns=(BAD_COLUMN, "{{name}}"),
            num_rows=2,
        )
        list(session)
        failure = session.failures[0]
        assert failure.column is None
        assert failure.column_index == 0
        assert failure.definition == BAD_COLUMN

    def test_replace_policy_substitutes_cells(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=3, failure_policy="replace", replacement="N/A"),
            data_columns=("{{name}}", BAD_COLUMN),
            num_rows=3,
        )
        chunk = session.next_chunk()

        assert chunk.status is ChunkStatus.COMPLETE_WITH_REPLACEMENTS
        assert len(chunk.rows) == 3
        assert all(row[1] == "N/A" for row in chunk.rows)
        assert all(row[0] != "N/A" for row in chunk.rows)
        assert {f.action for f in chunk.failures} == {"replace"}

    def test_abort_policy_invalidates_chunk_and_advances(self, faker):
        session = faker.produce_chunks(
            "json",
            ChunkSpec(rows_per_chunk=2, failure_policy="abort"),
            data_columns={"ok": "name", "bad": BAD_COLUMN.strip("{}")},
            num_rows=6,
        )
        chunks = list(session)

        assert [c.status for c in chunks] == [ChunkStatus.INVALIDATED] * 3
        assert all(len(c.rows) == 0 for c in chunks)
        assert all(json.loads(c.payload) == [] for c in chunks)
        # resume continues at the next chunk boundary: rows 1, 3, 5 failed
        assert [f.row for f in session.failures] == [1, 3, 5]
        assert {f.action for f in session.failures} == {"abort_chunk"}

    def test_json_failure_locates_dotted_column_paths(self, faker):
        session = faker.produce_chunks(
            "json",
            ChunkSpec(rows_per_chunk=1),
            data_columns={"outer": {"inner": "no_such_formatter"}},
            num_rows=1,
        )
        list(session)
        failure = session.failures[0]
        assert failure.row == 1
        assert failure.column == "outer.inner"
        assert failure.definition == "no_such_formatter"
        assert isinstance(failure.error, AttributeError)

    def test_json_list_format_failure_locates_named_column(self, faker):
        session = faker.produce_chunks(
            "json",
            ChunkSpec(rows_per_chunk=1),
            data_columns=[("item", "no_such_formatter")],
            num_rows=1,
        )
        list(session)
        assert session.failures[0].column == "item"
        assert session.failures[0].definition == "no_such_formatter"

    def test_fixed_width_failure_locates_column_index(self, faker):
        session = faker.produce_chunks(
            "fixed_width",
            ChunkSpec(rows_per_chunk=2, failure_policy="replace", replacement=""),
            data_columns=[(5, "name"), (3, "no_such_formatter")],
            num_rows=2,
        )
        chunk = session.next_chunk()
        failure = chunk.failures[0]
        assert failure.column is None
        assert failure.column_index == 1
        assert failure.definition == "no_such_formatter"
        assert len(chunk.rows) == 2

    def test_template_tokens_and_argument_groups_still_apply(self, faker):
        faker.set_arguments("small", "max_value", 10)
        try:
            session = faker.produce_chunks(
                "csv",
                ChunkSpec(rows_per_chunk=3),
                data_columns=("{{name}}|{{pyint:small}}",),
                num_rows=3,
            )
            chunk = session.next_chunk()
            for row in chunk.rows:
                value = int(row[0].split("|")[1])
                assert 0 <= value <= 10
        finally:
            faker.del_arguments("small")

    # ------------------------------------------------------------------ #
    # rubric 3: retrievable failure list, no recomputation, resumption
    # ------------------------------------------------------------------ #

    def test_failures_accumulate_and_are_retrievable_as_a_copy(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=2, failure_policy="skip"),
            data_columns=(BAD_COLUMN,),
            num_rows=6,
        )
        session.next_chunk()
        assert len(session.failures) == 2
        session.next_chunk()
        failures = session.failures
        assert len(failures) == 4
        assert all(isinstance(f.to_dict()["error"], str) for f in failures)
        json.dumps([f.to_dict() for f in failures])  # serializable failure list

        failures.clear()
        assert len(session.failures) == 4  # mutating the copy does not affect session

        session.next_chunk()
        assert len(session.failures) == 6

    def test_resuming_does_not_recompute_delivered_chunks(self, faker):
        def run_paused_after_first():
            faker.seed_instance(4242)
            session = faker.produce_chunks(
                "json",
                ChunkSpec(rows_per_chunk=2),
                data_columns={"name": "name", "city": "city"},
                num_rows=6,
            )
            first = session.next_chunk()
            rest = [session.next_chunk(), session.next_chunk()]
            return [first, *rest]

        def run_all_at_once():
            faker.seed_instance(4242)
            session = faker.produce_chunks(
                "json",
                ChunkSpec(rows_per_chunk=2),
                data_columns={"name": "name", "city": "city"},
                num_rows=6,
            )
            return list(session)

        paused = run_paused_after_first()
        immediate = run_all_at_once()
        assert [c.payload for c in paused] == [c.payload for c in immediate]

    def test_each_leaf_value_is_produced_exactly_once(self, faker):
        data_columns = {"a": "name", "b": "city", "c": "pyint"}
        with patch.object(
            MiscProvider,
            "_value_format_selection",
            autospec=True,
            wraps=MiscProvider._value_format_selection,
        ) as wrapped:
            session = faker.produce_chunks(
                "json",
                ChunkSpec(rows_per_chunk=3),
                data_columns=data_columns,
                num_rows=10,
            )
            list(session)
            assert wrapped.call_count == 30

    def test_stop_after_last_chunk(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=5),
            data_columns=("{{name}}",),
            num_rows=5,
        )
        session.next_chunk()
        assert session.is_exhausted
        with pytest.raises(StopIteration):
            session.next_chunk()

    # ------------------------------------------------------------------ #
    # rubric 4: concurrent independence
    # ------------------------------------------------------------------ #

    def test_concurrent_sessions_are_independent(self, faker):
        expected = [(0, 4, 1, 4), (1, 4, 5, 8), (2, 2, 9, 10)]

        def drain(output):
            local_fake = Faker()
            local_fake.seed_instance(2026)
            local_session = local_fake.produce_chunks(
                "csv",
                ChunkSpec(rows_per_chunk=4),
                data_columns=("{{name}}", "{{city}}"),
                num_rows=10,
            )
            for chunk in local_session:
                output.append((chunk.index, len(chunk.rows), chunk.row_start, chunk.row_end))

        results_a = []
        results_b = []
        threads = [threading.Thread(target=drain, args=(results,)) for results in (results_a, results_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results_a == expected
        assert results_b == expected

    def test_concurrent_calls_on_one_session_never_duplicate_a_chunk(self, faker):
        session = faker.produce_chunks(
            "json",
            ChunkSpec(rows_per_chunk=1),
            data_columns={"name": "name"},
            num_rows=30,
        )
        collected = []

        def consume():
            for chunk in session:
                collected.append(chunk.index)

        threads = [threading.Thread(target=consume) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(collected) == list(range(30))

    # ------------------------------------------------------------------ #
    # rubric 5: capacity bounds are enforced explicitly
    # ------------------------------------------------------------------ #

    def test_chunk_capacity_rejected_up_front(self, faker):
        with pytest.raises(ChunkCapacityExceeded):
            faker.produce_chunks(
                "csv",
                ChunkSpec(rows_per_chunk=2, max_chunks=3),
                data_columns=("{{name}}",),
                num_rows=10,
            )

    def test_failure_capacity_rejects_without_dropping_or_delivering(self, faker):
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=2, max_failures=3, failure_policy="skip"),
            data_columns=(BAD_COLUMN,),
            num_rows=10,
        )
        session.next_chunk()  # 2 failures recorded
        assert len(session.failures) == 2

        with pytest.raises(ChunkCapacityExceeded):
            session.next_chunk()  # would add 2 more failures, over the limit

        assert len(session.failures) == 2  # nothing silently dropped or appended
        assert session.delivered_chunks == 1  # the overflowing chunk was not delivered

    def test_invalid_chunk_declarations(self):
        for invalid in (0, -1, "5", 1.5, True):
            with pytest.raises(ChunkConfigurationError):
                ChunkSpec(rows_per_chunk=invalid)
        with pytest.raises(ChunkConfigurationError):
            ChunkSpec(rows_per_chunk=5, failure_policy="cry")
        with pytest.raises(ChunkConfigurationError):
            ChunkSpec(rows_per_chunk=5, max_chunks=0)
        with pytest.raises(ChunkConfigurationError):
            ChunkSpec(rows_per_chunk=5, max_failures=False)

    def test_produce_chunks_validates_its_declaration(self, faker):
        with pytest.raises(ChunkConfigurationError):
            faker.produce_chunks("pdf", rows_per_chunk=5)
        with pytest.raises(ChunkConfigurationError):
            faker.produce_chunks("csv")
        with pytest.raises(ChunkConfigurationError):
            faker.produce_chunks(
                "csv",
                ChunkSpec(rows_per_chunk=5),
                rows_per_chunk=5,
            )
        with pytest.raises(ChunkConfigurationError):
            faker.produce_chunks(
                "csv",
                ChunkSpec(rows_per_chunk=5),
                num_rows=0,
            )
        with pytest.raises(TypeError):
            faker.produce_chunks("csv", ChunkSpec(rows_per_chunk=5), data_columns="{{name}}")

    def test_chunk_keyword_requires_a_spec(self, faker):
        with pytest.raises(ChunkConfigurationError):
            faker.csv(chunk="not-a-spec", num_rows=2)

    # ------------------------------------------------------------------ #
    # rubric 6: backward compatibility and proxy forwarding
    # ------------------------------------------------------------------ #

    def test_legacy_one_shot_methods_unchanged_without_chunk(self, faker):
        assert isinstance(faker.csv(num_rows=2), str)
        assert isinstance(faker.tsv(num_rows=2), str)
        assert isinstance(faker.psv(num_rows=2), str)
        assert isinstance(faker.dsv(num_rows=2), str)
        assert isinstance(faker.json(num_rows=2), str)
        assert isinstance(faker.json_bytes(num_rows=2), bytes)
        assert isinstance(faker.fixed_width(num_rows=2), str)

        with pytest.raises(ValueError):
            faker.dsv(num_rows=0)
        with pytest.raises(TypeError):
            faker.dsv(data_columns=1)
        with pytest.raises(ValueError):
            faker.dsv(header=["only one"], data_columns=["{{name}}", "{{city}}"])

    def test_chunked_pathway_works_through_multi_locale_proxy(self):
        faker = Faker(["en_US", "ja_JP"])
        session = faker.produce_chunks(
            "csv",
            ChunkSpec(rows_per_chunk=3),
            data_columns=("{{name}}",),
            num_rows=6,
        )
        chunks = list(session)
        assert [len(c.rows) for c in chunks] == [3, 3]
        assert all(row for chunk in chunks for row in chunk.rows)
