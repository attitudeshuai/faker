class BaseFakerException(Exception):
    """The base exception for all Faker exceptions."""


class UniquenessException(BaseFakerException):
    """To avoid infinite loops, after a certain number of attempts,
    the "unique" attribute of the Proxy will throw this exception.
    """


class UnsupportedFeature(BaseFakerException):
    """The requested feature is not available on this system."""

    def __init__(self, msg: str, name: str) -> None:
        self.name = name
        super().__init__(msg)


class SnapshotError(BaseFakerException):
    """Base class for errors raised while exporting or importing a
    generator-state snapshot."""


class IncompatibleSnapshotError(SnapshotError):
    """The snapshot cannot be applied to the target instance because its
    structure (version marker, locale list, or built-in provider layout)
    does not match the target."""


class UnrepresentableProviderError(SnapshotError):
    """A runtime-registered provider cannot be represented in a snapshot.

    The error pinpoints *which* provider is responsible (its class, its
    module, the locale generator it was registered on, and its position in
    that generator's provider chain) so that the offending registration can
    be located. It is raised while validating/planning an export or import,
    before any live state is mutated.
    """

    def __init__(
        self,
        msg: str,
        *,
        provider_class: str,
        module: str | None,
        locale: str | None,
        index: int,
        methods: list[str] | None = None,
    ) -> None:
        self.provider_class = provider_class
        self.module = module
        self.locale = locale
        self.index = index
        self.methods = methods
        super().__init__(msg)


class UnrepresentableValueError(SnapshotError):
    """A piece of state (e.g. an argument of a ``.unique`` call) cannot be
    encoded into the portable snapshot format."""
