"""What automatic artifacts are requested and attached for a solve."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactOptions:
    """Which automatic artifacts are requested for a solve."""

    solver_log: bool = False
    solver_model: bool = False
    diagnose_infeasibility: bool = False


@dataclass(frozen=True)
class ArtifactWorkspace:
    """The private directory and requested artifacts of one observed solve."""

    directory: Path
    options: ArtifactOptions


@dataclass(frozen=True)
class ArtifactRequest:
    """A local file to attach to a solve's run."""

    local_path: str
    artifact_directory: str
