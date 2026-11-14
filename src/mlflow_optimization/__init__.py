"""Automatic MLflow logging of mathematical optimization solves."""

from collections.abc import Iterable

from mlflow_optimization import _bootstrap
from mlflow_optimization.common.diagnostics import MlflowOptimizationWarning
from mlflow_optimization.features.annotations.annotate import context
from mlflow_optimization.features.runs.record import SolveRun, last_solve

__all__ = ["MlflowOptimizationWarning", "SolveRun", "autolog", "context", "last_solve"]


def autolog(
    disable: bool = False,
    *,
    solvers: Iterable[str] | None = None,
    log_solver_parameters: bool = True,
    log_solver_metrics: bool = True,
    log_artifacts: bool = True,
) -> None:
    """Switch automatic logging of solves on, off, or to a new configuration."""
    _bootstrap.configure(
        disable=disable,
        solvers=solvers,
        log_solver_parameters=log_solver_parameters,
        log_solver_metrics=log_solver_metrics,
        log_artifacts=log_artifacts,
    )
