"""MLflow implementation of the artifact storage."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient


class MlflowArtifactStore:
    """Run artifacts stored through the MLflow tracking client."""

    def log_artifact(
        self: MlflowArtifactStore,
        run_id: str,
        local_path: str,
        artifact_path: str | None,
    ) -> None:
        tracking_client: MlflowClient = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        tracking_client.log_artifact(
            run_id=run_id,
            local_path=local_path,
            artifact_path=artifact_path,
        )
