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
    log_progress: bool = True,
    log_solver_log: bool = False,
    log_solver_model: bool = False,
    diagnose_infeasibility: bool = False,
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
    log_progress : bool, default True
        Whether incumbent, bound and gaps are logged as metric history while a solve runs.
    log_solver_log : bool, default False
        Whether the solver log is attached to the run; requires ``log_artifacts``.
    log_solver_model : bool, default False
        Whether the model is attached to the run just before solving; requires ``log_artifacts``.
    diagnose_infeasibility : bool, default False
        Whether an IIS or core is attached when a solve is infeasible; requires
        ``log_artifacts``. An IIS may require an additional solve.

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
        log_progress=log_progress,
        log_solver_log=log_solver_log,
        log_solver_model=log_solver_model,
        diagnose_infeasibility=diagnose_infeasibility,
    )
