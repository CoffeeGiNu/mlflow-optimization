"""MLflow implementation of the artifact storage."""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.utils.server_info import ServerInfoRequestError
from requests.exceptions import RequestException

from mlflow_optimization.features.artifacts.ports import ArtifactStoreError


class MlflowArtifactStore:
    """Run artifacts stored through the MLflow tracking client."""

    def log_artifact(
        self: MlflowArtifactStore,
        run_id: str,
        local_path: str,
        artifact_path: str | None,
    ) -> None:
        try:
            tracking_client: MlflowClient = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            tracking_client.log_artifact(
                run_id=run_id,
                local_path=local_path,
                artifact_path=artifact_path,
            )
        except (MlflowException, ServerInfoRequestError, RequestException, OSError) as error:
            raise ArtifactStoreError("MLflow artifact storage failed.") from error
