"""MLflow implementation of the artifact storage."""

from __future__ import annotations

from collections.abc import Callable

import mlflow
from mlflow.tracking import MlflowClient

from mlflow_optimization.features.artifacts.ports import ArtifactStoreError


class MlflowArtifactStore:
    """Run artifacts stored through the MLflow tracking client."""

    def log_artifact(
        self: MlflowArtifactStore,
        run_id: str,
        local_path: str,
        artifact_path: str | None,
    ) -> None:
        tracking_uri: str = _artifact_store_sdk_call(mlflow.get_tracking_uri)
        tracking_client: MlflowClient = _artifact_store_sdk_call(
            lambda: MlflowClient(tracking_uri=tracking_uri)
        )
        _artifact_store_sdk_call(
            lambda: tracking_client.log_artifact(
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            )
        )


def _artifact_store_sdk_call[ReturnType](
    mlflow_call: Callable[[], ReturnType],
) -> ReturnType:
    """An MLflow SDK call converted to an artifact-store error."""
    try:
        return mlflow_call()
    except Exception as error:
        raise ArtifactStoreError("MLflow artifact storage failed.") from error
