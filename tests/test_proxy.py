import copy
import pickle
import random
import threading
import time

from collections import OrderedDict
from unittest.mock import patch

import pytest

from faker import Faker
from faker.config import DEFAULT_LOCALE
from faker.generator import Generator
from faker.providers import BaseProvider


class TestFakerProxyClass:
    """Test Faker proxy class"""

    def test_unspecified_locale(self):
        fake = Faker()
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == DEFAULT_LOCALE

    def test_locale_as_string(self):
        locale = "en_US"
        fake = Faker(locale)
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == locale

    def test_locale_as_list(self):
        locale = ["en-US", "en_PH", "ja_JP", "de-DE"]
        expected = ["en_US", "en_PH", "ja_JP", "de_DE"]
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)

        locale = ["en-US", "en_PH", "ja_JP", "de-DE", "ja-JP", "de_DE", "en-US"] * 3
        expected = ["en_US", "en_PH", "ja_JP", "de_DE"]
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)

    def test_locale_as_list_invalid_value_type(self):
        locale = [1, 2]
        with pytest.raises(TypeError) as exc:
            Faker(locale)
        assert str(exc.value) == 'The locale "1" must be a string.'

    def test_locale_as_ordereddict(self):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )

        fake = Faker(locale)
        assert fake.locales == ["de_DE", "en_US", "en_PH", "ja_JP"]
        assert len(fake.factories) == 4
        assert fake.weights == [3, 2, 1, 5]

        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
                ("de-DE", 4),
                ("ja-JP", 2),
                ("en-US", 1),
            ]
        )
        fake = Faker(locale)
        assert fake.locales == ["de_DE", "en_US", "en_PH", "ja_JP"]
        assert len(fake.factories) == 4
        assert fake.weights == [4, 1, 1, 2]

    def test_invalid_locale(self):
        with pytest.raises(AttributeError):
            Faker("foo_Bar")

        with pytest.raises(AttributeError):
            Faker(["en_US", "foo_Bar"])

        with pytest.raises(AttributeError):
            Faker(
                OrderedDict(
                    [
                        ("de_DE", 3),
                        ("en-US", 2),
                        ("en-PH", 1),
                        ("foo_Bar", 5),
                    ]
                )
            )

    def test_items(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP", "de-DE", "ja-JP", "en-US"]
        processed_locale = list({code.replace("-", "_") for code in locale})
        fake = Faker(locale)
        for locale_name, factory in fake.items():
            assert locale_name in processed_locale
            assert isinstance(factory, (Generator, Faker))

    def test_dunder_getitem(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        for code in locale:
            assert isinstance(fake[code], (Generator, Faker))

        with pytest.raises(KeyError):
            fake["en_GB"]

    def test_seed_classmethod(self):
        fake = Faker()

        # Verify `seed()` is not callable from a class instance
        with pytest.raises(TypeError):
            fake.seed(0)

        # Verify calls to `seed()` from a class object are proxied properly
        with patch("faker.generator.Generator.seed") as mock_seed:
            mock_seed.assert_not_called()
            Faker.seed(0)
            mock_seed.assert_called_once_with(0)

    def test_seed_class_locales(self):
        Faker.seed(2043)
        count = 5
        fake = Faker(["en_GB", "fr_FR", "en_IN"])
        first_list = [fake.name() for _ in range(count)]
        # We convert the list to a set to remove duplicates and ensure
        # that we have exactly `count` unique fake values
        assert len(set(first_list)) == count

        Faker.seed(2043)
        fake = Faker(["en_GB", "fr_FR", "en_IN"])
        second_list = [fake.name() for _ in range(count)]

        assert first_list == second_list

    def test_seed_instance(self):
        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        with patch("faker.generator.Generator.seed_instance") as mock_seed_instance:
            mock_seed_instance.assert_not_called()
            fake.seed_instance(0)

            # Verify `seed_instance(0)` was called 4 times (one for each locale)
            calls = mock_seed_instance.call_args_list
            assert len(calls) == 4
            for call in calls:
                args, kwargs = call
                assert args == (0,)
                assert kwargs == {}

    def test_seed_locale(self):
        from faker.generator import random as shared_random_instance

        locale = ["de_DE", "en-US", "en-PH", "ja_JP"]
        fake = Faker(locale)

        # Get current state of each factory's random instance
        states = {}
        for locale, factory in fake.items():
            states[locale] = factory.random.getstate()

        # Create a new random instance for en_US factory with seed value
        fake.seed_locale("en_US", 0)

        for locale, factory in fake.items():
            # en_US factory should have changed
            if locale == "en_US":
                assert factory.random != shared_random_instance
                assert factory.random.getstate() != states[locale]

            # There should be no changes for the rest
            else:
                assert factory.random == shared_random_instance
                assert factory.random.getstate() == states[locale]

    def test_single_locale_proxy_behavior(self):
        fake = Faker()
        internal_factory = fake.factories[0]

        # Test if `Generator` attributes are proxied properly
        for attr in fake.generator_attrs:
            assert getattr(fake, attr) == getattr(internal_factory, attr)

        # Test if `random` getter and setter are proxied properly
        tmp_random = fake.random
        assert internal_factory.random != 1
        fake.random = 1
        assert internal_factory.random == 1
        fake.random = tmp_random

        # Test if a valid provider method is proxied properly
        # Factory selection logic should not be triggered
        with patch("faker.proxy.Faker._select_factory") as mock_select_factory:
            mock_select_factory.assert_not_called()
            assert fake.name == internal_factory.name
            fake.name()
            mock_select_factory.assert_not_called()

    def test_multiple_locale_proxy_behavior(self):
        fake = Faker(["de-DE", "en-US", "en-PH", "ja-JP"])

        # `Generator` attributes are not implemented
        for attr in fake.generator_attrs:
            with pytest.raises(NotImplementedError):
                getattr(fake, attr)

        # The `random` getter is not implemented
        with pytest.raises(NotImplementedError):
            random = fake.random
            random.seed(0)

        # The `random` setter is not implemented
        with pytest.raises(NotImplementedError):
            fake.random = 1

    def test_multiple_locale_caching_behavior(self):
        fake = Faker(["de_DE", "en-US", "en-PH", "ja_JP"])

        with patch("faker.proxy.Faker._build_mapping", wraps=fake._build_mapping) as mock_build_mapping:
            mock_build_mapping.assert_not_called()
            assert "name" not in fake._method_mappings

            # Test cache creation
            fake.name()
            assert "name" in fake._method_mappings
            mock_build_mapping.assert_called_once_with("name")

            # Test subsequent cache access: the same snapshot is reused and the
            # mapping is never rebuilt
            mapping = fake._method_mappings["name"]
            for _ in range(100):
                fake.name()
            assert fake._method_mappings["name"] is mapping
            mock_build_mapping.assert_called_once_with("name")

            # A weight replacement atomically invalidates the cached snapshot
            fake.set_weights({"de_DE": 1, "en_US": 1, "en_PH": 1, "ja_JP": 1})
            assert "name" not in fake._method_mappings
            fake.name()
            assert fake._method_mappings["name"] is not mapping
            assert mock_build_mapping.call_count == 2

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_no_weights(self, mock_factory_distribution, mock_factory_choice):
        fake = Faker(["de_DE", "en-US", "en-PH", "ja_JP"])

        # There are no distribution weights, so factory selection logic will use `random.choice`
        # if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            mock_select_factory.assert_not_called()
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()

            # All factories for the listed locales have the `name` provider method
            fake.name()
            mock_select_factory.assert_called_once_with("name")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with(fake.factories)
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()

            # Only `en_PH` factory has provider method `luzon_province`, so there is no
            # need for `random.choice` factory selection logic to run
            fake.luzon_province()
            mock_select_factory.assert_called_with("luzon_province")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()

            # Both `en_US` and `ja_JP` factories have provider method `zipcode`
            fake.zipcode()
            mock_select_factory.assert_called_once_with("zipcode")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with(
                [fake["en_US"], fake["ja_JP"]],
            )

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_with_weights(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)
        mock_factory_distribution.assert_not_called()
        mock_factory_choice.assert_not_called()

        # Distribution weights have been specified, so factory selection logic will use
        # `choices_distribution` if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # All factories for the listed locales have the `name` provider method
            fake.name()
            mock_select_factory.assert_called_once_with("name")
            mock_factory_distribution.assert_called_once_with(fake.factories, fake.weights)
            mock_factory_choice.assert_not_called()

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_single_provider(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)

        # Distribution weights have been specified, so factory selection logic will use
        # `choices_distribution` if multiple factories have the specified provider method
        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # Only `en_PH` factory has provider method `luzon_province`, so there is no
            # need for `choices_distribution` factory selection logic to run
            fake.luzon_province()
            mock_select_factory.assert_called_once_with("luzon_province")
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_multiple_locale_factory_selection_shared_providers(self, mock_factory_distribution, mock_factory_choice):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale)

        with patch("faker.proxy.Faker._select_factory", wraps=fake._select_factory) as mock_select_factory:
            # Both `en_US` and `ja_JP` factories have provider method `zipcode`
            fake.zipcode()
            mock_select_factory.assert_called_once_with("zipcode")
            mock_factory_distribution.assert_called_once_with([fake["en_US"], fake["ja_JP"]], [2, 5])
            mock_factory_choice.assert_not_called()

    def test_multiple_locale_factory_selection_unsupported_method(self):
        fake = Faker(["en_US", "en_PH"])
        with pytest.raises(AttributeError):
            fake.obviously_invalid_provider_method_a23f()

    @patch("random.Random.choice")
    @patch("random.Random.choices")
    def test_weighting_disabled_single_choice(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=False)
        fake.first_name()
        mock_choice_fn.assert_called()
        mock_choices_fn.assert_not_called()

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_with_locales(self, mock_choices_fn, mock_choice_fn):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()  # select provider
        mock_choice_fn.assert_called()  # select within provider

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_multiple_locales(self, mock_choices_fn, mock_choice_fn):
        locale = OrderedDict(
            [
                ("de_DE", 3),
                ("en-US", 2),
                ("en-PH", 1),
                ("ja_JP", 5),
            ]
        )
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()  # select provider
        mock_choice_fn.assert_called()  # select within provider

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_disabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=False)
        fake.uri_path(deep=3)

        assert mock_choices_fn.mock_calls[0][2]["k"] == 3
        assert mock_choices_fn.mock_calls[0][2]["weights"] is None
        mock_choice_fn.assert_not_called()

    @patch("random.Random.choice")
    @patch("random.Random.choices", wraps=random.Random().choices)
    def test_weighting_enabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        fake = Faker(use_weighting=True)
        fake.uri_path(deep=3)

        assert mock_choices_fn.mock_calls[0][2]["k"] == 3
        assert mock_choices_fn.mock_calls[0][2]["weights"] is None
        mock_choice_fn.assert_not_called()

    def test_dir_include_all_providers_attribute_in_list(self):
        fake = Faker(["en_US", "en_PH"])
        expected = set(
            dir(Faker)
            + [
                "_factories",
                "_locales",
                "_factory_map",
                "_weights",
                "_unique_proxy",
                "_optional_proxy",
                "_method_mappings",
                "_selection_lock",
                "_last_selection",
                "_on_missing",
            ]
        )
        for factory in fake.factories:
            expected |= {attr for attr in dir(factory) if not attr.startswith("_")}
        expected = sorted(expected)
        attributes = dir(fake)
        assert attributes == expected

    def test_select_factory_distribution_uses_instance_random(self):
        from faker.utils.distribution import choices_distribution

        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.seed_instance(12345)

        instance_random = fake._factories[0].random
        with patch("faker.proxy.choices_distribution", wraps=choices_distribution) as mock_dist:
            fake.name()
            mock_dist.assert_called_once()
            args = mock_dist.call_args[0]
            assert args[2] is instance_random

    def test_seed_instance_deterministic_multi_locale_no_weights(self):
        fake = Faker(["en_US", "ja_JP", "de_DE"])
        fake.seed_instance(12345)
        first_run = [fake.name() for _ in range(20)]

        fake2 = Faker(["en_US", "ja_JP", "de_DE"])
        fake2.seed_instance(12345)
        second_run = [fake2.name() for _ in range(20)]

        assert first_run == second_run

    def test_seed_instance_deterministic_multi_locale_with_weights(self):
        locale = OrderedDict([("de_DE", 3), ("en_US", 2), ("ja_JP", 5)])
        fake = Faker(locale)
        fake.seed_instance(12345)
        first_run = [fake.name() for _ in range(20)]

        fake2 = Faker(locale)
        fake2.seed_instance(12345)
        second_run = [fake2.name() for _ in range(20)]

        assert first_run == second_run

    def test_copy(self):
        fake = Faker("it_IT")
        fake2 = copy.deepcopy(fake)
        assert fake.locales == fake2.locales
        assert fake.locales is not fake2.locales
        assert fake2.factories[0] is fake2._factory_map["it_IT"]

    def test_copy_rebinds_single_locale_proxies(self):
        fake = Faker("en_US")
        fake2 = copy.deepcopy(fake)

        assert fake2.unique._proxy is fake2
        assert fake2.optional._proxy is fake2
        assert fake2.factories[0] is fake2._factory_map["en_US"]
        assert fake2.optional.name(prob=1.0)

    def test_copy_rebinds_multiple_locale_proxies(self):
        fake = Faker(["en_US", "ja_JP"])
        fake2 = copy.deepcopy(fake)

        assert fake2.unique._proxy is fake2
        assert fake2.optional._proxy is fake2
        assert fake2.factories[0] is fake2["en_US"]
        assert fake2.factories[1] is fake2["ja_JP"]
        assert fake2.optional.name(prob=1.0)

    def test_copy_unique_uses_copied_proxy_state(self):
        source = Faker("en_US")
        source.seed_instance(999)
        clone_a = copy.deepcopy(source)
        clone_b = copy.deepcopy(source)

        clone_a.seed_instance(200)
        clone_b.seed_instance(200)

        assert clone_a.unique._proxy is clone_a
        assert clone_b.unique._proxy is clone_b
        assert clone_a.unique.name() == clone_b.unique.name()

    def test_pickle(self):
        fake = Faker()
        pickled = pickle.dumps(fake)
        pickle.loads(pickled)


class TestRuntimeWeightedSelection:
    """Runtime weight management, candidate observability and cache consistency."""

    LOCALES = ["de_DE", "en_US", "en_PH", "ja_JP"]

    def test_weights_default_none_and_readonly_copy(self):
        fake = Faker(self.LOCALES)
        assert fake.weights is None

        fake.set_weights(OrderedDict(zip(self.LOCALES, [3, 2, 1, 5])))
        snapshot = fake.weights
        assert snapshot == [3, 2, 1, 5]
        # Mutating the returned copy must not affect the instance
        snapshot.append(99)
        assert fake.weights == [3, 2, 1, 5]

    def test_set_weights_with_sequence(self):
        fake = Faker(self.LOCALES)
        fake.set_weights([1, 2, 3, 4])
        assert fake.weights == [1, 2, 3, 4]

    def test_set_weights_normalizes_dashes(self):
        fake = Faker(self.LOCALES)
        fake.set_weights({"de-DE": 1, "en-US": 2, "en-PH": 3, "ja-JP": 4})
        assert fake.weights == [1, 2, 3, 4]

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_set_weights_none_restores_uniform(self, mock_distribution, mock_choice):
        fake = Faker(self.LOCALES)
        fake.set_weights([1, 2, 3, 4])

        fake.set_weights(None)
        assert fake.weights is None
        fake.name()
        mock_choice.assert_called_once_with(fake.factories)
        mock_distribution.assert_not_called()

    @pytest.mark.parametrize(
        "bad_weights,exc_type",
        [
            ({"de_DE": 1, "en_US": 2, "en_PH": 3}, ValueError),
            ({"de_DE": 1, "en_US": 2, "en_PH": 3, "ja_JP": 4, "fr_FR": 1}, ValueError),
            ([1, 2, 3], ValueError),
            ([1, 2, 3, 4, 5], ValueError),
            ([1, 2, 3, -1], ValueError),
            ([1, 2, 3, float("nan")], ValueError),
            ([1, 2, 3, float("inf")], ValueError),
            ([1, 2, "3", 4], TypeError),
            ([1, 2, True, 4], TypeError),
            ([0, 0, 0, 0], ValueError),
            ("1234", TypeError),
        ],
    )
    def test_invalid_weights_rejected_wholesale(self, bad_weights, exc_type):
        fake = Faker(self.LOCALES)
        original = [3, 2, 1, 5]
        fake.set_weights(OrderedDict(zip(self.LOCALES, original)))

        with pytest.raises(exc_type):
            fake.set_weights(bad_weights)

        # The previous weights remain exactly as declared
        assert fake.weights == original

    def test_failed_replacement_keeps_existing_cache(self):
        fake = Faker(self.LOCALES)
        fake.set_weights({"de_DE": 3, "en_US": 2, "en_PH": 1, "ja_JP": 5})
        fake.zipcode()
        mapping = fake._method_mappings["zipcode"]

        with pytest.raises(ValueError):
            fake.set_weights([1, 2])

        assert fake._method_mappings["zipcode"] is mapping

    @patch("faker.proxy.Faker._select_factory_choice")
    @patch("faker.proxy.Faker._select_factory_distribution")
    def test_set_weights_rebuilds_distribution(self, mock_distribution, mock_choice):
        fake = Faker(self.LOCALES)
        fake.zipcode()
        mock_choice.assert_called_once_with([fake["en_US"], fake["ja_JP"]])

        fake.set_weights({"de_DE": 30, "en_US": 20, "en_PH": 10, "ja_JP": 50})
        fake.zipcode()
        mock_distribution.assert_called_once_with(
            [fake["en_US"], fake["ja_JP"]],
            [20, 50],
        )

    def test_zero_weights_pin_selection_to_locale(self):
        fake = Faker(self.LOCALES)
        fake.seed_instance(42)
        fake.set_weights({"de_DE": 0, "en_US": 1, "en_PH": 0, "ja_JP": 0})

        for _ in range(100):
            fake.name()
            assert fake.last_selected_locale == "en_US"

    def test_last_selection_records_locale_and_exclusions(self):
        fake = Faker(self.LOCALES)
        assert fake.last_selection is None
        assert fake.last_selected_locale is None
        assert fake.last_excluded_locales == []

        fake.zipcode()
        selection = fake.last_selection
        assert selection.method_name == "zipcode"
        assert selection.locale in ("en_US", "ja_JP")
        assert fake.last_selected_locale == selection.locale
        assert selection.candidate_locales == ("en_US", "ja_JP")
        assert selection.excluded_locales == ("de_DE", "en_PH")
        assert fake.last_excluded_locales == ["de_DE", "en_PH"]

    def test_last_selection_single_candidate_records_exclusions(self):
        fake = Faker(self.LOCALES)
        fake.luzon_province()
        assert fake.last_selected_locale == "en_PH"
        assert fake.last_excluded_locales == ["de_DE", "en_US", "ja_JP"]

    def test_last_selection_unsupported_method(self):
        fake = Faker(self.LOCALES)
        with pytest.raises(AttributeError):
            fake.obviously_invalid_provider_method_a23f()

        selection = fake.last_selection
        assert selection.locale is None
        assert selection.method_name == "obviously_invalid_provider_method_a23f"
        assert selection.candidate_locales == ()
        assert selection.excluded_locales == tuple(self.LOCALES)

    def test_on_missing_exclude_is_default(self):
        fake = Faker(self.LOCALES)
        assert fake.on_missing == "exclude"
        # Missing locales are excluded instead of failing the call
        fake.zipcode()
        assert fake.last_excluded_locales == ["de_DE", "en_PH"]

    def test_on_missing_fail_allows_methods_available_everywhere(self):
        fake = Faker(self.LOCALES, on_missing="fail")
        fake.seed_instance(1)
        fake.name()
        assert fake.last_selected_locale in self.LOCALES
        assert fake.last_excluded_locales == []

    def test_on_missing_fail_raises_and_records(self):
        fake = Faker(self.LOCALES, on_missing="fail")

        with pytest.raises(AttributeError):
            fake.zipcode()
        assert fake.last_selected_locale is None
        assert fake.last_excluded_locales == ["de_DE", "en_PH"]

        # The cached mapping does not resurrect the method on later calls
        with pytest.raises(AttributeError):
            fake.zipcode()

    def test_invalid_on_missing_strategy(self):
        with pytest.raises(ValueError):
            Faker(self.LOCALES, on_missing="invalid")

    def test_invalidate_selection_cache(self):
        fake = Faker(self.LOCALES)
        fake.name()
        assert "name" in fake._method_mappings

        fake.invalidate_selection_cache()
        assert fake._method_mappings == {}

    def test_invalidate_picks_up_runtime_candidate_change(self):
        fake = Faker(["en_US", "en_PH"])

        class ExtraProvider(BaseProvider):
            def runtime_only_method(self):
                return "runtime"

        # Only the en_PH factory gains the method at runtime
        fake["en_PH"].add_provider(ExtraProvider)
        fake.invalidate_selection_cache()

        factories, _ = fake._map_provider_method("runtime_only_method")
        assert factories == [fake["en_PH"]]

    def test_concurrent_selections_never_hit_excluded_locales(self):
        fake = Faker(self.LOCALES)
        fake.seed_instance(7)
        allowed = {fake["en_US"], fake["ja_JP"]}
        selected = []
        errors = []
        barrier = threading.Barrier(5)

        def worker():
            try:
                barrier.wait()
                for _ in range(500):
                    selected.append(fake._select_factory("zipcode"))
            except Exception as exc:  # pragma: no cover - surfaces thread failures
                errors.append(exc)

        def invalidator():
            barrier.wait()
            for index in range(200):
                fake.invalidate_selection_cache()
                fake.set_weights({"de_DE": index, "en_US": 1, "en_PH": 2, "ja_JP": 3})

        threads = [threading.Thread(target=worker) for _ in range(4)]
        threads.append(threading.Thread(target=invalidator))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert [thread for thread in threads if thread.is_alive()] == []
        assert errors == []
        assert selected
        assert all(factory in allowed for factory in selected)

    def test_weight_swap_waits_for_in_flight_rebuild(self):
        fake = Faker(self.LOCALES)
        entered = threading.Event()
        can_finish = threading.Event()
        original_build = fake._build_mapping

        def blocking_build(method_name):
            entered.set()
            assert can_finish.wait(timeout=5)
            return original_build(method_name)

        # Force the first build to pause while holding the selection lock
        fake._build_mapping = blocking_build
        first = threading.Thread(target=lambda: fake.name())
        first.start()
        assert entered.wait(5)

        swapped = threading.Event()
        new_weights = {"de_DE": 10, "en_US": 20, "en_PH": 30, "ja_JP": 40}

        def swap():
            fake.set_weights(new_weights)
            swapped.set()

        second = threading.Thread(target=swap)
        second.start()

        # While the rebuild is paused, the swap cannot take effect
        time.sleep(0.2)
        assert not swapped.is_set()
        assert fake.weights is None

        can_finish.set()
        first.join(5)
        second.join(5)

        assert swapped.is_set()
        assert fake.weights == [10, 20, 30, 40]
        # The in-flight snapshot was built under the old configuration and is
        # discarded wholesale, never published as a mixed cache entry
        assert fake._method_mappings == {}
