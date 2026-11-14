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
    libraries: Iterable[str] | None = None,
    log_solver_parameters: bool = True,
    log_solver_metrics: bool = True,
    log_artifacts: bool = True,
) -> None:
    """Configure automatic logging of optimization solves.

    Parameters
    ----------
    disable : bool, default False
        Disable automatic logging; other settings are ignored when True.
    libraries : Iterable[str] or None, optional
        Supported names: gurobipy, pyscipopt, pyomo, mathopt, cp_sat, routing.
        None selects all available integrations; an empty iterable selects none.
    log_solver_parameters : bool, default True
        Whether solver parameters are logged.
    log_solver_metrics : bool, default True
        Whether native solver metrics are logged.
    log_artifacts : bool, default True
        Whether files attached through ``SolveRun.log_artifact`` are logged.

    Raises
    ------
    ImportError
        An explicitly selected integration cannot be imported.
    ValueError
        An unsupported library name is selected while logging is enabled.
    """
    _bootstrap.configure(
        disable=disable,
        libraries=libraries,
        log_solver_parameters=log_solver_parameters,
        log_solver_metrics=log_solver_metrics,
        log_artifacts=log_artifacts,
    )
