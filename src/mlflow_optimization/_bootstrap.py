"""Composition root: wires feature ports to concrete adapters."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow_optimization.features.autolog.ports import SolverIntegration
    from mlflow_optimization.features.autolog.service import AutologManager
    from mlflow_optimization.features.runs.types import RecordOptions
    from mlflow_optimization.features.runs.record import RecordSolve
    from mlflow_optimization.integrations.mlflow.artifacts import MlflowArtifactStore
    from mlflow_optimization.integrations.mlflow.runs import MlflowRunStore

INTEGRATION_MODULES: dict[str, str] = {
    integration_api: f"mlflow_optimization.integrations.{integration_api}.adapter"
    for integration_api in ("gurobipy", "pyscipopt", "pyomo", "mathopt", "cp_sat", "routing")
}

_record_solve: RecordSolve | None = None
_autolog_manager: AutologManager | None = None


def configure(
    *,
    disable: bool,
    solvers: Iterable[str] | None,
    log_solver_parameters: bool,
    log_solver_metrics: bool,
    log_artifacts: bool,
) -> None:
    """Apply an autolog configuration to the process-wide wiring."""
    global _autolog_manager
    global _record_solve

    integrations: list[SolverIntegration] = []
    if not disable:
        integrations = _resolve_integrations(solvers)

    if _autolog_manager is None:
        from mlflow_optimization.features.autolog.service import AutologManager
        from mlflow_optimization.features.runs.types import RecordOptions
        from mlflow_optimization.features.runs.record import RecordSolve
        from mlflow_optimization.integrations.mlflow.artifacts import MlflowArtifactStore
        from mlflow_optimization.integrations.mlflow.runs import MlflowRunStore

        run_store: MlflowRunStore = MlflowRunStore()
        artifact_store: MlflowArtifactStore = MlflowArtifactStore()
        record_options: RecordOptions = RecordOptions(
            log_solver_parameters=log_solver_parameters,
            log_solver_metrics=log_solver_metrics,
            log_artifacts=log_artifacts,
        )
        record_solve: RecordSolve = RecordSolve(run_store, artifact_store, record_options)
        autolog_manager: AutologManager = AutologManager(record_solve)

        _record_solve = record_solve
        _autolog_manager = autolog_manager
    else:
        assert _record_solve is not None
        _record_solve.options.log_solver_parameters = log_solver_parameters
        _record_solve.options.log_solver_metrics = log_solver_metrics
        _record_solve.options.log_artifacts = log_artifacts

    assert _autolog_manager is not None
    if disable:
        _autolog_manager.disable()
        return

    _autolog_manager.enable(integrations)


def _resolve_integrations(solvers: Iterable[str] | None) -> list[SolverIntegration]:
    """The selected solver integrations."""
    module_names: dict[str, str] = dict(INTEGRATION_MODULES)
    selected_names: tuple[str, ...] = (
        tuple(module_names) if solvers is None else tuple(dict.fromkeys(solvers))
    )

    for solver_name in selected_names:
        if solver_name not in module_names:
            valid_names: str = ", ".join(module_names)
            raise ValueError(f"Unknown solver {solver_name!r}. Valid names: {valid_names}")

    integrations: list[SolverIntegration] = []
    for solver_name in selected_names:
        module_name: str = module_names[solver_name]
        try:
            integration_module: ModuleType = import_module(module_name)
            integration: SolverIntegration = integration_module.Integration()
        except ImportError:
            if solvers is not None:
                raise
            continue
        integrations.append(integration)

    return integrations
