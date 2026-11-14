"""Reversible attribute replacement."""

from collections.abc import Callable
from collections.abc import Mapping
from inspect import signature
from typing import Any


def receiver_of(original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """The receiver of a method call."""
    receiver_name: str = next(iter(signature(original).parameters))
    if args:
        return args[0]
    return kwargs[receiver_name]


class Patch:
    """A reversible replacement of one attribute on a class or module."""

    def __init__(self, owner: Any, name: str) -> None:
        self.owner: Any = owner
        self.name: str = name
        self._installed: bool = False

    def install(self, factory: Callable[[Any], Any]) -> None:
        """Replace the attribute with ``factory(original)``; no-op when installed."""
        if self._installed:
            return
        original: Any = getattr(self.owner, self.name)
        self._own: bool = self.name in vars(self.owner)
        self._descriptor: Any = vars(self.owner).get(self.name)
        setattr(self.owner, self.name, factory(original))
        self._installed = True

    def uninstall(self) -> None:
        """Restore the exact original attribute; no-op when not installed."""
        if not self._installed:
            return
        if self._own:
            setattr(self.owner, self.name, self._descriptor)
        else:
            delattr(self.owner, self.name)
        self._installed = False
