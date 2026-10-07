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


class ChunkedProductionError(BaseFakerException):
    """Base class for errors raised by the chunked structured-output pathway."""


class ChunkConfigurationError(ChunkedProductionError, ValueError):
    """A chunk declaration (``ChunkSpec`` or chunked session arguments) is invalid.

    Subclasses :class:`ValueError` so callers that validated the one-shot
    producers with ``isinstance(..., ValueError)`` keep working.
    """


class ChunkCapacityExceeded(ChunkedProductionError):
    """Raised when a chunk or failure-list capacity limit would be exceeded.

    The request is rejected explicitly instead of silently truncating output
    or dropping failures.
    """
