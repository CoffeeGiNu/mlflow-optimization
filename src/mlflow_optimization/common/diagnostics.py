"""Notification channel shared by every feature and integration."""

import logging
import warnings


class MlflowOptimizationWarning(UserWarning):
    """Notification that observing, validating or recording a solve failed."""


def warn(message: str, *, stacklevel: int = 2) -> None:
    """Emit an ``MlflowOptimizationWarning``."""
    try:
        warnings.warn(message, MlflowOptimizationWarning, stacklevel=stacklevel + 1)
    except MlflowOptimizationWarning:
        logging.getLogger("mlflow_optimization").warning(message)
