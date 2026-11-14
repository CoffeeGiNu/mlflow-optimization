"""Shared fixtures for optimization tests."""

import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import mlflow
import pytest
from mlflow.entities import Experiment, Run
from mlflow.tracking import MlflowClient
from mlflow.tracking import fluent as mlflow_fluent

from mlflow_optimization import autolog
from mlflow_optimization import _bootstrap


@dataclass(frozen=True)
class Tracking:
    """An isolated local MLflow tracking store."""

    client: MlflowClient
    experiment_id: str

    def solve_runs(self) -> list[Run]:
        """The recorded solve runs ordered by solve index."""
        experiments: list[Experiment] = self.client.search_experiments()
        experiment_ids: list[str] = [experiment.experiment_id for experiment in experiments]
        runs: list[Run] = self.client.search_runs(experiment_ids=experiment_ids)
        solve_runs: list[Run] = [
            run for run in runs if "optimization.solve_index" in run.data.tags
        ]
        return sorted(
            solve_runs,
            key=lambda run: int(run.data.tags["optimization.solve_index"]),
        )


@pytest.fixture
def tracking(tmp_path: Path) -> Iterator[Tracking]:
    """An isolated local MLflow tracking store for a test."""
    previous_tracking_uri: str = mlflow.get_tracking_uri()
    previous_tracking_environment: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    previous_active_experiment_id: str | None = mlflow_fluent._active_experiment_id
    previous_experiment_environment: str | None = os.environ.get("MLFLOW_EXPERIMENT_ID")
    previous_file_store_environment: str | None = os.environ.get("MLFLOW_ALLOW_FILE_STORE")
    previous_fake_module: ModuleType | None = sys.modules.get("fakes")
    previous_fake_frontend_module: ModuleType | None = sys.modules.get("fakes_frontend")
    original_integration_modules: dict[str, str] = _bootstrap.INTEGRATION_MODULES.copy()
    tests_directory: str = str(Path(__file__).parent)
    added_tests_directory: bool = tests_directory not in sys.path
    if added_tests_directory:
        sys.path.insert(0, tests_directory)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    tracking_uri: str = (tmp_path / "mlruns").as_uri()
    try:
        autolog(disable=True)
        while mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_tracking_uri(tracking_uri)
        experiment: Experiment = mlflow.set_experiment(
            experiment_name="optimization-tests"
        )
        client: MlflowClient = MlflowClient(tracking_uri=tracking_uri)
        _bootstrap.INTEGRATION_MODULES["fake"] = "fakes"
        _bootstrap.INTEGRATION_MODULES["fake_frontend"] = "fakes_frontend"
        yield Tracking(client=client, experiment_id=experiment.experiment_id)
    finally:
        try:
            autolog(disable=True)
        finally:
            while mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.set_tracking_uri(previous_tracking_uri)
            if previous_tracking_environment is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = previous_tracking_environment
            mlflow_fluent._active_experiment_id = previous_active_experiment_id
            if previous_experiment_environment is None:
                os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
            else:
                os.environ["MLFLOW_EXPERIMENT_ID"] = previous_experiment_environment
            if previous_file_store_environment is None:
                os.environ.pop("MLFLOW_ALLOW_FILE_STORE", None)
            else:
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = previous_file_store_environment
            _bootstrap.INTEGRATION_MODULES.clear()
            _bootstrap.INTEGRATION_MODULES.update(original_integration_modules)
            if previous_fake_module is None:
                sys.modules.pop("fakes", None)
            else:
                sys.modules["fakes"] = previous_fake_module
            if previous_fake_frontend_module is None:
                sys.modules.pop("fakes_frontend", None)
            else:
                sys.modules["fakes_frontend"] = previous_fake_frontend_module
            if added_tests_directory:
                sys.path.remove(tests_directory)
