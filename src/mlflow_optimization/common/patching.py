"""Reversible attribute replacement."""

import sys
from collections.abc import Callable, Mapping
from inspect import signature
from types import ModuleType
from typing import Any


def receiver_of(original: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """The receiver of a method call."""
    receiver_name: str | None = next(iter(signature(original).parameters), None)
    if receiver_name is None:
        return None
    if args:
        return args[0]
    return kwargs.get(receiver_name)


def rebind_module_names(
    original: Any,
    replacement: Any,
) -> list[tuple[ModuleType, str]]:
    """The module-level names rebound from an original object to a replacement."""
    rebound_names: list[tuple[ModuleType, str]] = []
    modules: tuple[Any, ...] = tuple(sys.modules.values())
    for module_value in modules:
        if not isinstance(module_value, ModuleType):
            continue
        module: ModuleType = module_value
        module_values: dict[str, Any] = vars(module)
        module_bindings: tuple[tuple[str, Any], ...] = tuple(module_values.items())
        for name, value in module_bindings:
            if value is not original:
                continue
            module_values[name] = replacement
            rebound_names.append((module, name))
    return rebound_names


def restore_module_names(
    rebound_names: list[tuple[ModuleType, str]],
    original: Any,
) -> None:
    """The restoration of module-level names changed by rebind_module_names."""
    for module, name in reversed(rebound_names):
        vars(module)[name] = original


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
