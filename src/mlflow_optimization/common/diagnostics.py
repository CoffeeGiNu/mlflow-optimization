"""Notification channel shared by every feature and integration."""

import warnings


class MlflowOptimizationWarning(UserWarning):
    """Notification that observing, validating or recording a solve failed."""


def warn(message: str, *, stacklevel: int = 2) -> None:
    """Emit an ``MlflowOptimizationWarning``."""
    warnings.warn(message, MlflowOptimizationWarning, stacklevel=stacklevel + 1)
