"""Notification channel shared by every feature and integration."""

import logging
import warnings


class MlflowOptimizationWarning(UserWarning):
    """The category of every warning emitted by this library."""


def warn(message: str, *, stacklevel: int = 2) -> None:
    """Emit an ``MlflowOptimizationWarning``."""
    try:
        warnings.warn(message, MlflowOptimizationWarning, stacklevel=stacklevel + 1)
    except BaseException:
        try:
            logging.getLogger("mlflow_optimization").warning(message)
        except BaseException:
            pass
