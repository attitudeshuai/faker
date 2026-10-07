import copy
import json
import pickle
import threading

from collections import OrderedDict

import pytest

from faker import Faker, Generator
from faker.exceptions import (
    IncompatibleSnapshotError,
    SnapshotError,
    UnrepresentableProviderError,
)
from faker.providers import BaseProvider
from faker.snapshot import SNAPSHOT_VERSION


class TicketProvider(BaseProvider):
    """Importable runtime provider (module-level class)."""

    def ticket_number(self) -> str:
        return f"T-{self.random_int(min=1, max=9999):04d}"


def canonical(state: dict) -> str:
    return json.dumps(state, sort_keys=True)


class TestSnapshotBasics:
    def test_snapshot_is_versioned_and_json_compatible(self):
        state = Faker().snapshot()
        assert state["snapshot_version"] == SNAPSHOT_VERSION
        assert state["object"] == "faker"
        assert isinstance(state["faker_version"], str)
        # Must round-trip through JSON without any non-portable types.
        assert json.loads(json.dumps(state))

    def test_generator_direct_snapshot(self):
        generator = Generator()
        state = generator.snapshot()
        assert state["object"] == "generator"
        assert state["snapshot_version"] == SNAPSHOT_VERSION

    def test_single_locale_continues_same_sequence(self):
        fake = Faker("en_US")
        fake.seed_instance(42)
        [fake.name() for _ in range(7)]
        state = fake.snapshot()

        expected = [fake.name() for _ in range(15)]
        fake.restore(state)
        assert [fake.name() for _ in range(15)] == expected

    def test_multi_locale_continues_same_sequence_with_weights(self):
        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.seed_instance(777)
        [fake.name() for _ in range(8)]
        fake.zipcode()
        state = fake.snapshot()

        expected = [fake.name() for _ in range(15)]
        fake.restore(state)
        assert [fake.name() for _ in range(15)] == expected

    def test_multi_locale_without_weights(self):
        fake = Faker(["en_US", "ja_JP", "de_DE"])
        fake.seed_instance(55)
        [fake.name() for _ in range(5)]
        state = fake.snapshot()

        expected = [fake.name() for _ in range(10)]
        fake.restore(state)
        assert [fake.name() for _ in range(10)] == expected


class TestSnapshotContents:
    def test_argument_groups_round_trip(self):
        fake = Faker()
        fake.set_arguments("group1", "argument1", 1)
        fake.set_arguments("group1", "argument2", 2)
        fake.set_arguments("group2", {"a": "x", "b": 10})

        state = fake.snapshot()
        fake.del_arguments("group1")
        fake.del_arguments("group2")
        fake.restore(state)

        assert fake.get_arguments("group1", "argument1") == 1
        assert fake.get_arguments("group1", "argument2") == 2
        assert fake.get_arguments("group2") == {"a": "x", "b": 10}

    def test_unique_history_exact_rerun(self):
        fake = Faker("en_US")
        fake.seed_instance(1)
        initial = fake.snapshot()

        values = [fake.unique.random_int(min=1, max=100000) for _ in range(20)]
        fake.restore(initial)
        assert [fake.unique.random_int(min=1, max=100000) for _ in range(20)] == values

    def test_unique_history_is_carried_and_consulted(self):
        # With the same seed, the first draw repeats if the history were lost.
        fake = Faker("en_US")
        fake.seed_instance(1)
        first = fake.unique.random_int(min=1, max=100000)
        state = fake.snapshot()

        restored = Faker.from_snapshot(json.loads(json.dumps(state)))
        restored.seed_instance(1)
        second = restored.unique.random_int(min=1, max=100000)
        assert second != first

    def test_selection_cache_is_exported_and_restored(self):
        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.name()
        fake.zipcode()
        assert hasattr(fake, "_cached_name_mapping")
        assert hasattr(fake, "_cached_zipcode_mapping")

        state = fake.snapshot()
        methods = {entry["method"] for entry in state["caches"]}
        assert methods == {"name", "zipcode"}
        name_cache = next(entry for entry in state["caches"] if entry["method"] == "name")
        assert name_cache["factories"] == ["de_DE", "en_US", "ja_JP"]
        assert name_cache["weights"] == [3, 2, 5]
        zip_cache = next(entry for entry in state["caches"] if entry["method"] == "zipcode")
        # Locale fallback: en_US and ja_JP resolve to a zipcode provider, de_DE does not.
        assert zip_cache["factories"] == ["en_US", "ja_JP"]
        assert zip_cache["weights"] == [2, 5]

        # Build a fresh proxy and confirm the cache is materialized on import.
        restored = Faker.from_snapshot(json.loads(json.dumps(state)))
        assert hasattr(restored, "_cached_name_mapping")
        assert hasattr(restored, "_cached_zipcode_mapping")
        assert not hasattr(restored, "_cached_first_name_mapping")

    def test_stale_cache_is_removed_on_restore(self):
        fake = Faker(["en_US", "ja_JP"])
        fake.name()
        assert hasattr(fake, "_cached_name_mapping")

        # An older snapshot without the cache must clear it.
        state = fake.snapshot()
        state["caches"] = []
        fake.restore(state)
        assert not hasattr(fake, "_cached_name_mapping")

    def test_shared_random_binding_is_restored(self):
        import faker.generator as generator_module

        global_random = generator_module.random
        saved = global_random.getstate()
        try:
            fake = Faker("en_US")  # unseeded -> bound to shared random
            global_random.seed(1234)
            expected_state = global_random.getstate()
            state = fake.snapshot()
            assert state["factories"]["en_US"]["random"]["shared"] is True

            # Mutate the shared source, then restore: it must be rebound to the
            # global object at its recorded position.
            global_random.seed(999)
            fake.restore(state)
            assert fake.random is global_random
            assert fake.random.getstate() == expected_state
        finally:
            global_random.setstate(saved)


class TestRuntimeProviders:
    def test_importable_runtime_provider_round_trip(self):
        fake = Faker("en_US")
        fake.add_provider(TicketProvider(fake))
        fake.seed_instance(31)
        state = fake.snapshot()

        custom = [entry for entry in state["factories"]["en_US"]["providers"] if entry["kind"] == "custom"]
        assert len(custom) == 1
        assert custom[0]["class"] == "TicketProvider"
        assert "ticket_number" in custom[0]["methods"]

        expected = fake.ticket_number()
        fake.restore(state)
        assert fake.ticket_number() == expected

    def test_importable_runtime_provider_via_from_snapshot(self):
        fake = Faker("en_US")
        fake.add_provider(TicketProvider(fake))
        fake.seed_instance(31)
        state = fake.snapshot()

        restored = Faker.from_snapshot(json.loads(json.dumps(state)))
        assert restored.ticket_number().startswith("T-")

    def test_locally_defined_provider_rejected_at_export(self):
        class LocalProvider:
            def custom_thing(self):
                return "local"

        fake = Faker("en_US")
        fake.add_provider(LocalProvider())

        with pytest.raises(UnrepresentableProviderError) as exc_info:
            fake.snapshot()
        error = exc_info.value
        assert error.locale == "en_US"
        assert error.index == 0
        assert "LocalProvider" in error.provider_class
        assert error.methods == ["custom_thing"]
        assert "locally" in str(error)

        # Export rejection never mutates anything.
        assert fake.custom_thing() == "local"

    def test_missing_provider_class_rejected_at_import_without_half_state(self):
        fake = Faker("en_US")
        fake.add_provider(TicketProvider(fake))
        state = fake.snapshot()
        before = fake.snapshot()

        # Tamper with the custom provider coordinates.
        generator_data = state["factories"]["en_US"]
        custom_entry = next(entry for entry in generator_data["providers"] if entry["kind"] == "custom")
        custom_entry["module"] = "no_such_module_xyz_123"

        with pytest.raises(UnrepresentableProviderError) as exc_info:
            fake.restore(state)
        error = exc_info.value
        assert error.locale == "en_US"
        assert error.index == 0
        assert "no_such_module_xyz_123" in error.provider_class

        # No mutation happened: the live state is exactly the old one.
        assert canonical(fake.snapshot()) == canonical(before)

    def test_incompatible_provider_constructor_rejected(self):
        fake = Faker("en_US")
        fake.add_provider(TicketProvider(fake))
        state = fake.snapshot()
        before = fake.snapshot()

        custom_entry = next(entry for entry in state["factories"]["en_US"]["providers"] if entry["kind"] == "custom")
        # datetime.datetime cannot be constructed with a generator argument.
        custom_entry["module"] = "datetime"
        custom_entry["class"] = "datetime"

        with pytest.raises(UnrepresentableProviderError, match="constructor"):
            fake.restore(state)
        assert canonical(fake.snapshot()) == canonical(before)


class TestSnapshotValidation:
    def test_unknown_version_rejected(self):
        state = Faker().snapshot()
        state["snapshot_version"] = 999
        with pytest.raises(IncompatibleSnapshotError, match="version"):
            Faker().restore(state)

    def test_non_dict_snapshot_rejected(self):
        with pytest.raises(IncompatibleSnapshotError):
            Faker().restore(["not", "a", "snapshot"])

    def test_locale_mismatch_rejected(self):
        state = Faker(["en_US", "ja_JP"]).snapshot()
        with pytest.raises(IncompatibleSnapshotError, match="locales"):
            Faker(["en_US", "de_DE"]).restore(state)

        with pytest.raises(IncompatibleSnapshotError, match="locales"):
            Faker("en_US").restore(state)

    def test_wrong_object_type_rejected(self):
        state = Faker().snapshot()
        state["object"] = "generator"
        with pytest.raises(IncompatibleSnapshotError, match="Faker snapshot"):
            Faker().restore(state)

    def test_builtin_layout_mismatch_rejected(self):
        state = Faker("en_US").snapshot()
        builtin_entry = state["factories"]["en_US"]["providers"][0]
        builtin_entry["class"] = "DefinitelyNotARealProvider"
        with pytest.raises(IncompatibleSnapshotError, match="position"):
            Faker("en_US").restore(state)

    def test_snapshot_error_is_base_exception_type(self):
        assert issubclass(IncompatibleSnapshotError, SnapshotError)
        assert issubclass(UnrepresentableProviderError, SnapshotError)

    def test_malformed_cache_rejected(self):
        state = Faker(["en_US", "ja_JP"]).snapshot()
        state["caches"] = [{"method": "name", "factories": ["zz_ZZ"], "weights": None}]
        with pytest.raises(IncompatibleSnapshotError, match="zz_ZZ"):
            Faker(["en_US", "ja_JP"]).restore(state)


class TestPickleAndCopyCompatibility:
    """Instances without custom providers keep the exact pre-change behavior."""

    def test_pickle_plain_instance(self):
        fake = Faker()
        restored = pickle.loads(pickle.dumps(fake))
        assert isinstance(restored.name(), str)

    def test_deepcopy_unique_history_quirk_preserved(self):
        fake = Faker("en_US")
        fake.unique.boolean()
        clone = copy.deepcopy(fake)
        # Deepcopy deliberately drops the actual history, keeping only keys.
        assert clone.unique._seen == {key: {clone.unique._sentinel} for key in fake.unique._seen}
        assert clone.unique._proxy is clone

    def test_pickle_with_importable_custom_provider(self):
        fake = Faker("en_US")
        fake.add_provider(TicketProvider(fake))
        restored = pickle.loads(pickle.dumps(fake))
        assert restored.ticket_number().startswith("T-")

    def test_snapshot_after_pickle_is_usable(self):
        fake = Faker()
        restored = pickle.loads(pickle.dumps(fake))
        state = restored.snapshot()
        restored.restore(state)
        assert isinstance(restored.name(), str)


class TestSnapshotConcurrency:
    def test_concurrent_restore_observes_only_complete_states(self):
        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])

        target = Faker(locale)
        target.seed_instance(7)
        target.name()
        target.zipcode()
        state_a = target.snapshot()

        other = Faker(locale)
        other.seed_instance(99)
        [other.name() for _ in range(10)]
        other.zipcode()
        state_b = other.snapshot()

        canonical_states = {canonical(state_a), canonical(state_b)}

        observed = []
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    observed.append(canonical(target.snapshot()))
            except Exception as exc:  # pragma: no cover - failure reporting
                errors.append(repr(exc))

        def writer():
            use_b = False
            while not stop.is_set():
                target.restore(state_b if use_b else state_a)
                use_b = not use_b

        threads = [threading.Thread(target=reader) for _ in range(4)]
        writer_thread = threading.Thread(target=writer)
        for thread in threads:
            thread.start()
        writer_thread.start()

        stop.wait(1.5)
        stop.set()
        for thread in threads:
            thread.join()
        writer_thread.join()

        assert errors == []
        assert observed
        assert set(observed) <= canonical_states

    def test_concurrent_generation_during_snapshot(self):
        fake = Faker("en_US")
        fake.seed_instance(3)
        stop = threading.Event()
        errors = []
        snapshots = []

        def generate():
            try:
                while not stop.is_set():
                    fake.unique.random_int(min=1, max=10**9)
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

        def capture():
            try:
                while not stop.is_set():
                    snapshots.append(fake.snapshot())
            except Exception as exc:  # pragma: no cover
                errors.append(repr(exc))

        generators = [threading.Thread(target=generate) for _ in range(3)]
        snapshot_thread = threading.Thread(target=capture)
        for thread in generators:
            thread.start()
        snapshot_thread.start()

        stop.wait(1.0)
        stop.set()
        for thread in generators:
            thread.join()
        snapshot_thread.join()

        assert errors == []
        assert snapshots
        # Every captured snapshot is internally consistent and restorable.
        for state in snapshots[:20]:
            clone = Faker("en_US")
            clone.restore(state)
            clone.unique.random_int(min=1, max=10**9)
