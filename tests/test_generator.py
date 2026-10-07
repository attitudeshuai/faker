import threading

from unittest.mock import patch

import pytest

from faker import Faker, Generator


class BarProvider:
    def foo_formatter(self):
        return "barfoo"


class FooProvider:
    def foo_formatter(self):
        return "foobar"

    def foo_formatter_with_arguments(self, param="", append=""):
        return "baz" + str(param) + str(append)


class HighPriorityProvider:
    __priority__ = 10

    def foo_formatter(self):
        return "high"


class LowPriorityProvider:
    __priority__ = -1

    def foo_formatter(self):
        return "low"


class _SharedBase:
    def shared_formatter(self):
        return "shared"


class SharedProviderA(_SharedBase):
    pass


class SharedProviderB(_SharedBase):
    pass


@pytest.fixture(autouse=True)
def generator():
    generator = Generator()
    generator.add_provider(FooProvider())
    return generator


class TestGenerator:
    """Test Generator class"""

    def test_get_formatter_returns_correct_formatter(self, generator):
        foo_provider = generator.providers[0]
        formatter = generator.get_formatter("foo_formatter")
        assert callable(formatter) and formatter == foo_provider.foo_formatter

    def test_get_formatter_with_unknown_formatter(self, generator):
        with pytest.raises(AttributeError) as excinfo:
            generator.get_formatter("barFormatter")
        assert str(excinfo.value) == "Unknown formatter 'barFormatter'"

        fake = Faker("it_IT")
        with pytest.raises(AttributeError) as excinfo:
            fake.get_formatter("barFormatter")
        assert str(excinfo.value) == "Unknown formatter 'barFormatter' with locale 'it_IT'"

    def test_format_calls_formatter_on_provider(self, generator):
        assert generator.format("foo_formatter") == "foobar"

    def test_format_passes_arguments_to_formatter(self, generator):
        result = generator.format("foo_formatter_with_arguments", "foo", append="!")
        assert result == "bazfoo!"

    def test_add_provider_overrides_old_provider(self, generator):
        assert generator.format("foo_formatter") == "foobar"
        generator.add_provider(BarProvider())
        assert generator.format("foo_formatter") == "barfoo"

    def test_parse_without_formatter_tokens(self, generator):
        assert generator.parse("fooBar#?") == "fooBar#?"

    def test_parse_with_valid_formatter_tokens(self, generator):
        result = generator.parse('This is {{foo_formatter}} a text with "{{ foo_formatter }}"')
        assert result == 'This is foobar a text with "foobar"'

    def test_arguments_group_with_values(self, generator):
        generator.set_arguments("group1", "argument1", 1)
        generator.set_arguments("group1", "argument2", 2)
        assert generator.get_arguments("group1", "argument1") == 1
        assert generator.del_arguments("group1", "argument2") == 2
        assert generator.get_arguments("group1", "argument2") is None
        assert generator.get_arguments("group1") == {"argument1": 1}

    def test_arguments_group_with_dictionaries(self, generator):
        generator.set_arguments("group2", {"argument1": 3, "argument2": 4})
        assert generator.get_arguments("group2") == {"argument1": 3, "argument2": 4}
        assert generator.del_arguments("group2") == {"argument1": 3, "argument2": 4}
        assert generator.get_arguments("group2") is None

    def test_arguments_group_with_invalid_name(self, generator):
        assert generator.get_arguments("group3") is None
        assert generator.del_arguments("group3") is None

    def test_arguments_group_with_invalid_argument_type(self, generator):
        with pytest.raises(ValueError) as excinfo:
            generator.set_arguments("group", ["foo", "bar"])
        assert str(excinfo.value) == "Arguments must be either a string or dictionary"

    def test_parse_with_valid_formatter_arguments(self, generator):
        generator.set_arguments("format_name", {"param": "foo", "append": "bar"})
        result = generator.parse('This is "{{foo_formatter_with_arguments:format_name}}"')
        generator.del_arguments("format_name")
        assert result == 'This is "bazfoobar"'

    def test_parse_with_unknown_arguments_group(self, generator):
        with pytest.raises(AttributeError) as excinfo:
            generator.parse('This is "{{foo_formatter_with_arguments:unknown}}"')
        assert str(excinfo.value) == "Unknown argument group 'unknown'"

    def test_parse_with_unknown_formatter_token(self, generator):
        with pytest.raises(AttributeError) as excinfo:
            generator.parse("{{barFormatter}}")
        assert str(excinfo.value) == "Unknown formatter 'barFormatter'"

    def test_magic_call_calls_format(self, generator):
        assert generator.foo_formatter() == "foobar"

    def test_magic_call_calls_format_with_arguments(self, generator):
        assert generator.foo_formatter_with_arguments("foo") == "bazfoo"

    @patch("faker.generator.random_module.getstate")
    def test_get_random(self, mock_system_random, generator):
        random_instance = generator.random
        random_instance.getstate()
        mock_system_random.assert_not_called()

    @patch("faker.generator.random_module.seed")
    def test_random_seed_doesnt_seed_system_random(self, mock_system_random, generator):
        # Save original state of shared random instance to avoid affecting other tests
        state = generator.random.getstate()

        generator.seed(0)
        mock_system_random.assert_not_called()

        # Restore state of shared random instance
        generator.random.setstate(state)


class TestMethodDispatchResolution:
    """Test the method dispatch resolution and reporting layer."""

    def test_get_provider_of_returns_current_provider(self, generator):
        foo_provider = generator.providers[0]
        assert generator.get_provider_of("foo_formatter") is foo_provider

    def test_get_provider_of_unknown_method_returns_none(self, generator):
        assert generator.get_provider_of("does_not_exist_xyz") is None

    def test_get_provider_name_of_returns_name(self, generator):
        assert generator.get_provider_name_of("foo_formatter") == "tests.test_generator.fooprovider"

    def test_same_name_conflict_resolved_by_registration_order(self, generator):
        generator.add_provider(BarProvider())
        bar_provider = generator.providers[0]

        assert generator.format("foo_formatter") == "barfoo"
        assert generator.get_provider_of("foo_formatter") is bar_provider

        record = generator.get_conflicts()["foo_formatter"]
        assert record.method_name == "foo_formatter"
        assert record.provider is bar_provider
        assert len(record.providers) == 2
        assert set(record.priorities.values()) == {0}

    def test_conflict_log_appends_resolutions(self, generator):
        log = generator.get_conflict_log()
        assert log == []
        generator.add_provider(BarProvider())
        log = generator.get_conflict_log()
        assert len(log) == 1
        assert log[0].method_name == "foo_formatter"

    def test_conflict_resolved_by_declared_priority(self):
        gen = Generator()
        gen.add_provider(LowPriorityProvider())
        gen.add_provider(HighPriorityProvider())

        # Higher declared priority wins despite registration order
        assert gen.format("foo_formatter") == "high"
        assert type(gen.get_provider_of("foo_formatter")) is HighPriorityProvider

        record = gen.get_conflicts()["foo_formatter"]
        assert record.provider is gen.providers[0]  # high priority is also the newest
        assert record.priorities["tests.test_generator.highpriorityprovider"] == 10
        assert record.priorities["tests.test_generator.lowpriorityprovider"] == -1

    def test_declared_priority_beats_later_registration(self):
        gen = Generator()
        gen.add_provider(HighPriorityProvider())
        gen.add_provider(BarProvider())  # default priority, registered later

        # High priority from an earlier provider still wins
        assert gen.format("foo_formatter") == "high"
        assert type(gen.get_provider_of("foo_formatter")) is HighPriorityProvider

    def test_inherited_shared_method_is_not_conflict(self):
        gen = Generator()
        gen.add_provider(SharedProviderA())
        gen.add_provider(SharedProviderB())

        assert gen.format("shared_formatter") == "shared"
        assert "shared_formatter" not in gen.get_conflicts()

    def test_runtime_provider_gets_name_metadata_and_is_findable(self):
        gen = Generator()
        gen.add_provider(FooProvider())
        assert gen.provider("tests.test_generator.fooprovider") is gen.providers[0]

    def test_shadow_method_and_restore(self, generator):
        original_provider = generator.providers[0]

        generator.shadow_method("foo_formatter", lambda: "shadowed")
        assert generator.is_shadowed("foo_formatter")
        assert generator.format("foo_formatter") == "shadowed"
        assert generator.get_provider_of("foo_formatter") is None

        info = generator.get_method_info("foo_formatter")
        assert info.shadowed is True
        assert info.provider is None
        assert original_provider in info.providers

        generator.restore_method("foo_formatter")
        assert not generator.is_shadowed("foo_formatter")
        assert generator.format("foo_formatter") == "foobar"
        assert generator.get_provider_of("foo_formatter") is original_provider

    def test_restore_returns_to_pre_shadow_attribution(self, generator):
        generator.add_provider(BarProvider())
        bar_provider = generator.providers[0]
        assert generator.format("foo_formatter") == "barfoo"

        generator.shadow_method("foo_formatter", lambda: "shadowed")
        generator.restore_method("foo_formatter")

        # Must revert to the pre-shadow winner (Bar), never to Foo
        assert generator.get_provider_of("foo_formatter") is bar_provider
        assert generator.format("foo_formatter") == "barfoo"

    def test_nested_shadows_unwind_in_order(self, generator):
        generator.shadow_method("foo_formatter", lambda: "first")
        generator.shadow_method("foo_formatter", lambda: "second")
        assert generator.format("foo_formatter") == "second"

        generator.restore_method("foo_formatter")
        assert generator.format("foo_formatter") == "first"

        generator.restore_method("foo_formatter")
        assert generator.format("foo_formatter") == "foobar"

    def test_shadow_unknown_method_raises(self, generator):
        with pytest.raises(AttributeError):
            generator.shadow_method("unknown_xyz", lambda: None)

    def test_restore_unshadowed_method_raises(self, generator):
        with pytest.raises(ValueError):
            generator.restore_method("foo_formatter")

    def test_get_method_owners_covers_all_methods(self, generator):
        owners = generator.get_method_owners()
        assert owners["foo_formatter"] is generator.providers[0]
        assert "foo_formatter_with_arguments" in owners

    def test_owners_report_and_dispatch_are_consistent(self):
        # Default assembly: every reported owner must be the provider actually
        # bound, and every method must be callable.
        fake = Faker()
        gen = fake.factories[0]

        owners = gen.get_method_owners()
        for name, owner in owners.items():
            function = getattr(gen, name)
            assert callable(function)
            if owner is not None:
                assert function.__self__ is owner
            else:
                assert gen.is_shadowed(name)

    def test_default_priority_owners_match_historical_precedence(self):
        # With no declared priorities the winner must always be the most
        # recently added provider that carries the method.
        fake = Faker()
        gen = fake.factories[0]

        for name, owner in gen.get_method_owners().items():
            historical = next(p for p in gen.providers if callable(getattr(p, name, None)))
            assert owner is historical

    def test_registration_while_shadowed_does_not_overtake(self, generator):
        generator.shadow_method("foo_formatter", lambda: "shadowed")
        generator.add_provider(BarProvider())

        # Shadow stays on top despite the later registration
        assert generator.format("foo_formatter") == "shadowed"

        # Restore goes back to the pre-shadow attribution (Foo)
        generator.restore_method("foo_formatter")
        assert generator.format("foo_formatter") == "foobar"

    def test_concurrent_registration_never_half_state(self):
        gen = Generator()
        errors = []

        def register(index):
            class Provider:
                def tracked_method(self):
                    return index

            Provider.__qualname__ = f"TrackedProvider{index}"
            try:
                gen.add_provider(Provider())
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(30)]
        for thread in threads:
            thread.start()

            # While registrations are in flight the method must be callable
            # and the report must agree with the binding.
            if hasattr(gen, "tracked_method"):
                assert callable(gen.tracked_method)
                owner = gen.get_provider_of("tracked_method")
                assert gen.tracked_method.__self__ is owner

        for thread in threads:
            thread.join()

        assert errors == []
        assert callable(gen.tracked_method)
        assert gen.tracked_method() in range(30)
        owner = gen.get_provider_of("tracked_method")
        assert owner is gen.providers[0]
