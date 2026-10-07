from __future__ import annotations

from typing import List, Optional, Tuple


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


class TemplateError(BaseFakerException):
    """Base class for failures encountered while expanding a ``parse()`` template.

    Every failure carries locating information: the offending ``token``, its
    ``position`` in the template, and the ``template`` string being expanded.
    """

    def __init__(
        self,
        msg: str,
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
    ) -> None:
        super().__init__(msg)
        self.token = token
        self.position = position
        self.template = template


class UnknownTemplate(TemplateError):
    """Raised when a token references a formatter that is not registered."""

    def __init__(
        self,
        formatter: str,
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
        locale: Optional[str] = None,
        is_callable: bool = True,
    ) -> None:
        if is_callable:
            msg = f"Unknown formatter {formatter!r}"
        else:
            msg = f"Formatter {formatter!r} is not callable"
        if token is not None:
            msg += f" in token {token!r}"
        if position is not None:
            msg += f" at position {position}"
        if template is not None:
            msg += f" of template {template!r}"
        if locale:
            msg += f" with locale {locale!r}"
        super().__init__(msg, token=token, position=position, template=template)
        self.formatter = formatter


class UnknownArgumentGroup(TemplateError):
    """Raised when a token references an argument group that does not exist."""

    def __init__(
        self,
        group: str,
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
    ) -> None:
        msg = f"Unknown argument group {group!r}"
        if token is not None:
            msg += f" referenced by token {token!r}"
        if position is not None:
            msg += f" at position {position}"
        if template is not None:
            msg += f" of template {template!r}"
        super().__init__(msg, token=token, position=position, template=template)
        self.group = group


class TemplateCycleError(TemplateError):
    """Raised when template tokens keep expanding into each other.

    ``chain`` contains the ``(formatter, group)`` pairs of the expansion loop,
    with the first pair repeated at the end.
    """

    def __init__(
        self,
        chain: Tuple[Tuple[str, str], ...],
        *,
        token: Optional[str] = None,
        position: Optional[int] = None,
        template: Optional[str] = None,
    ) -> None:
        rendered = " -> ".join(formatter if not group else f"{formatter}:{group}" for formatter, group in chain)
        msg = f"Recursive template expansion detected ({rendered})"
        if token is not None:
            msg += f" while expanding token {token!r}"
        if position is not None:
            msg += f" at position {position}"
        if template is not None:
            msg += f" of template {template!r}"
        super().__init__(msg, token=token, position=position, template=template)
        self.chain: List[Tuple[str, str]] = list(chain)


class ArgumentGroupError(BaseFakerException):
    """Base class for argument group bookkeeping failures."""


class ArgumentGroupNotFound(ArgumentGroupError):
    """Raised when an operation targets an argument group that does not exist."""

    def __init__(self, group: str) -> None:
        super().__init__(f"Argument group {group!r} is not defined")
        self.group = group


class ArgumentNotFound(ArgumentGroupError):
    """Raised when an operation targets an argument that does not exist in a group."""

    def __init__(self, group: str, argument: str) -> None:
        super().__init__(f"Argument {argument!r} is not defined in argument group {group!r}")
        self.group = group
        self.argument = argument
