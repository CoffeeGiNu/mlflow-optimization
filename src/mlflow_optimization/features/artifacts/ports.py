"""Contract of the artifact storage."""

from typing import Protocol


class ArtifactStoreError(Exception):
    """An error raised when artifact storage fails."""


class ArtifactStore(Protocol):
    """The storage of files attached to runs."""

    def log_artifact(
        self, run_id: str, local_path: str, artifact_path: str | None
    ) -> None:
        """Attach a local file to a run."""
        ...
