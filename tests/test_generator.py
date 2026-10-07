import threading

from unittest.mock import patch

import pytest

from faker import Faker, Generator, exceptions


class BarProvider:
    def foo_formatter(self):
        return "barfoo"


class FooProvider:
    def foo_formatter(self):
        return "foobar"

    def foo_formatter_with_arguments(self, param="", append=""):
        return "baz" + str(param) + str(append)


class NestedProvider:
    def top_token(self):
        return "{{middle_token}}"

    def middle_token(self):
        return "{{bottom_token}}"

    def bottom_token(self):
        return "deep"


class CycleProvider:
    def cycle_a(self):
        return "{{cycle_b}}"

    def cycle_b(self):
        return "{{cycle_a}}"

    def cycle_self(self):
        return "{{cycle_self}}"


class MutatingProvider:
    def __init__(self, generator):
        self.generator = generator

    def mutating_token(self):
        self.generator.set_arguments("leaked_group", "param", "leaked")
        return "mutated"

    def trailing_token(self):
        # Reaching this token means the earlier token already mutated groups
        return "trailing"


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
        with pytest.raises(exceptions.ArgumentNotFound):
            generator.get_arguments("group1", "argument2")
        assert generator.get_arguments("group1") == {"argument1": 1}

    def test_arguments_group_with_dictionaries(self, generator):
        generator.set_arguments("group2", {"argument1": 3, "argument2": 4})
        assert generator.get_arguments("group2") == {"argument1": 3, "argument2": 4}
        assert generator.del_arguments("group2") == {"argument1": 3, "argument2": 4}
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("group2")

    def test_arguments_group_with_invalid_name(self, generator):
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("group3")
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.del_arguments("group3")

    def test_arguments_group_with_invalid_argument_type(self, generator):
        with pytest.raises(ValueError) as excinfo:
            generator.set_arguments("group", ["foo", "bar"])
        assert str(excinfo.value) == "Arguments must be either a string or dictionary"

    def test_arguments_group_with_invalid_group_name(self, generator):
        with pytest.raises(TypeError):
            generator.set_arguments(123, "argument", 1)
        with pytest.raises(ValueError):
            generator.set_arguments("", "argument", 1)

    def test_set_arguments_dictionary_does_not_alias_input(self, generator):
        arguments = {"argument1": {"nested": 1}}
        generator.set_arguments("group", arguments)
        arguments["argument1"]["nested"] = 2
        assert generator.get_arguments("group") == {"argument1": {"nested": 1}}

    def test_arguments_groups_are_isolated_between_instances(self):
        first = Generator()
        second = Generator()
        first.set_arguments("shared_group", "argument", "first")
        # The inner arguments mapping must not be shared through a class attr
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            second.get_arguments("shared_group")
        second.set_arguments("shared_group", "argument", "second")
        assert first.get_arguments("shared_group", "argument") == "first"
        assert second.get_arguments("shared_group", "argument") == "second"

    def test_arguments_group_versions(self, generator):
        generator.set_arguments("versioned", "argument", 1)
        created = generator.get_arguments_version("versioned")
        assert isinstance(created, int) and created >= 1

        generator.set_arguments("versioned", "argument", 2)
        overridden = generator.get_arguments_version("versioned")
        assert overridden > created

        generator.del_arguments("versioned")
        generator.set_arguments("versioned", "argument", 3)
        rebuilt = generator.get_arguments_version("versioned")
        assert rebuilt > overridden

        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments_version("missing")

    def test_arguments_scope_restores_outer_snapshot(self, generator):
        generator.set_arguments("outer", "argument", "outside")
        generator.set_arguments("shadowed", "argument", "before")
        before = generator.get_arguments_version("shadowed")

        with generator.arguments_scope():
            generator.set_arguments("inner", "argument", "inside")
            generator.set_arguments("shadowed", "argument", "during")
            generator.del_arguments("outer")
            assert generator.get_arguments("inner", "argument") == "inside"
            assert generator.get_arguments("shadowed", "argument") == "during"
            with pytest.raises(exceptions.ArgumentGroupNotFound):
                generator.get_arguments("outer")

        assert generator.get_arguments("outer", "argument") == "outside"
        assert generator.get_arguments("shadowed", "argument") == "before"
        assert generator.get_arguments_version("shadowed") == before
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("inner")

    def test_nested_arguments_scopes_inherit_and_restore(self, generator):
        generator.set_arguments("outer", "argument", 1)
        with generator.arguments_scope():
            generator.set_arguments("middle", "argument", 2)
            with generator.arguments_scope():
                assert generator.get_arguments("outer", "argument") == 1
                assert generator.get_arguments("middle", "argument") == 2
                generator.set_arguments("inner", "argument", 3)
            with pytest.raises(exceptions.ArgumentGroupNotFound):
                generator.get_arguments("inner")
            assert generator.get_arguments("middle", "argument") == 2
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("middle")
        assert generator.get_arguments("outer", "argument") == 1

    def test_arguments_scope_restores_after_exception(self, generator):
        generator.set_arguments("outer", "argument", 1)
        with pytest.raises(RuntimeError):
            with generator.arguments_scope():
                generator.set_arguments("outer", "argument", 2)
                raise RuntimeError("boom")
        assert generator.get_arguments("outer", "argument") == 1

    def test_concurrent_argument_scopes_are_isolated(self, generator):
        generator.add_provider(FooProvider())
        generator.set_arguments("base_group", {"param": "base"})
        barrier = threading.Barrier(2)
        errors = []

        def worker(group_name: str, value: str, other: str) -> None:
            try:
                with generator.arguments_scope():
                    generator.set_arguments(group_name, {"param": value})
                    barrier.wait(timeout=5)
                    result = generator.parse("{{ foo_formatter_with_arguments:" + group_name + " }}")
                    assert result == "baz" + value
                    # The outer base snapshot is visible...
                    assert generator.get_arguments("base_group", "param") == "base"
                    # ...but the other thread's scope is not.
                    with pytest.raises(exceptions.ArgumentGroupNotFound):
                        generator.get_arguments(other)
                    with pytest.raises(exceptions.UnknownArgumentGroup):
                        generator.parse("{{ foo_formatter:" + other + " }}")
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("scope_a", "A", "scope_b")),
            threading.Thread(target=worker, args=("scope_b", "B", "scope_a")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert errors == []
        # Scopes never leak to the base storage
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("scope_a")
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("scope_b")

    def test_parse_expands_nested_tokens_recursively(self):
        generator = Generator()
        generator.add_provider(NestedProvider())
        assert generator.parse("{{top_token}}") == "deep"
        assert generator.parse("x-{{ top_token }}-y") == "x-deep-y"

    def test_parse_cycle_is_detected_for_mutual_templates(self):
        generator = Generator()
        generator.add_provider(CycleProvider())
        with pytest.raises(exceptions.TemplateCycleError) as excinfo:
            generator.parse("{{cycle_a}}")
        assert excinfo.value.chain[0] == excinfo.value.chain[-1]
        assert ("cycle_a", "") in excinfo.value.chain
        assert ("cycle_b", "") in excinfo.value.chain
        assert "cycle_a -> cycle_b -> cycle_a" in str(excinfo.value)

    def test_parse_cycle_is_detected_for_self_reference(self):
        generator = Generator()
        generator.add_provider(CycleProvider())
        with pytest.raises(exceptions.TemplateCycleError) as excinfo:
            generator.parse("{{cycle_self}}")
        assert len(excinfo.value.chain) == 2

    def test_failed_parse_batch_leaves_no_partial_writes(self):
        generator = Generator()
        generator.add_provider(MutatingProvider(generator))
        with pytest.raises(exceptions.UnknownTemplate):
            generator.parse("{{mutating_token}} {{missing_token}}")
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("leaked_group")

    def test_successful_parse_batch_does_not_leak_internal_writes(self):
        generator = Generator()
        generator.add_provider(MutatingProvider(generator))
        assert generator.parse("{{mutating_token}}-{{trailing_token}}") == "mutated-trailing"
        with pytest.raises(exceptions.ArgumentGroupNotFound):
            generator.get_arguments("leaked_group")

    def test_parse_with_valid_formatter_arguments(self, generator):
        generator.set_arguments("format_name", {"param": "foo", "append": "bar"})
        result = generator.parse('This is "{{foo_formatter_with_arguments:format_name}}"')
        generator.del_arguments("format_name")
        assert result == 'This is "bazfoobar"'

    def test_parse_with_unknown_arguments_group(self, generator):
        template = 'This is "{{foo_formatter_with_arguments:unknown}}"'
        with pytest.raises(exceptions.UnknownArgumentGroup) as excinfo:
            generator.parse(template)
        assert excinfo.value.group == "unknown"
        assert excinfo.value.token == "{{foo_formatter_with_arguments:unknown}}"
        assert excinfo.value.position == 9
        assert excinfo.value.template == template
        assert "Unknown argument group 'unknown'" in str(excinfo.value)
        assert "at position 9" in str(excinfo.value)

    def test_parse_with_unknown_formatter_token(self, generator):
        template = "{{barFormatter}}"
        with pytest.raises(exceptions.UnknownTemplate) as excinfo:
            generator.parse(template)
        assert excinfo.value.formatter == "barFormatter"
        assert excinfo.value.token == template
        assert excinfo.value.position == 0
        assert "Unknown formatter 'barFormatter'" in str(excinfo.value)

    def test_parse_with_unknown_formatter_token_in_nested_result(self):
        # The locatable error must surface from a token produced by another
        # formatter, not as a raw AttributeError from deep inside the batch.
        generator = Generator()
        generator.add_provider(NestedProvider())

        class BrokenProvider:
            def broken_token(self):
                return "{{missing_formatter}}"

        generator.add_provider(BrokenProvider())
        with pytest.raises(exceptions.UnknownTemplate) as excinfo:
            generator.parse("{{broken_token}}")
        assert excinfo.value.formatter == "missing_formatter"
        assert excinfo.value.template == "{{missing_formatter}}"

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
