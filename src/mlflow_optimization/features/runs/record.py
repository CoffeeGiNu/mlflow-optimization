"""Recording of completed solves and later additions to them."""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

from mlflow_optimization.common.diagnostics import warn
from mlflow_optimization.features.annotations.annotate import (
    annotation_tags,
    current_annotation,
    user_metric,
    user_tags,
)
from mlflow_optimization.features.annotations.types import Annotation, UserMetric
from mlflow_optimization.features.artifacts.policy import artifact_refusal
from mlflow_optimization.features.artifacts.ports import (
    ArtifactStore,
    ArtifactStoreError,
)
from mlflow_optimization.features.autolog.types import CompletedSolve, SolveObservation
from mlflow_optimization.features.metadata.collect import collect_metadata
from mlflow_optimization.features.metadata.types import CollectedMetadata
from mlflow_optimization.features.metrics.types import NormalizedMetrics
from mlflow_optimization.features.metrics.normalize import normalize_metrics
from mlflow_optimization.features.outcomes.classify import (
    classify_outcome,
    outcome_tags,
)
from mlflow_optimization.features.outcomes.types import SolveOutcome

from .types import RecordOptions, RunReference, RunStatus, SolveRecord
from .ports import RunStore, RunStoreError


@dataclass(frozen=True)
class _RecordingContext:
    """The state captured at solve start."""

    parent: RunReference | None
    annotation: Annotation


class SolveRun:
    """The handle of one recorded solve."""

    def __init__(
        self,
        run: RunReference,
        runs: RunStore,
        artifacts: ArtifactStore,
        options: RecordOptions,
    ) -> None:
        """Initialize a handle for a recorded solve.

        Parameters
        ----------
        run : RunReference
            Reference to the recorded solve's run.
        runs : RunStore
            Store used to add metrics and tags to the run.
        artifacts : ArtifactStore
            Store used to attach files to the run.
        options : RecordOptions
            Options controlling recorded solver data and artifact logging.
        """
        self.run_id: str = run.run_id
        self._run: RunReference = run
        self._runs: RunStore = runs
        self._artifacts: ArtifactStore = artifacts
        self._options: RecordOptions = options

    def log_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add a user metric to this solve.

        Parameters
        ----------
        name : str
            User metric name; its recorded key is prefixed with ``user.``.
        value : float
            Metric value.
        unit : str, optional
            Unit tag stored as ``user.{name}.unit``.
        description : str, optional
            Description tag stored as ``user.{name}.description``.
        """
        try:
            entry: UserMetric | None = user_metric(
                name, value, unit=unit, description=description
            )
            if entry is not None:
                self._runs.log_metric(self._run, entry.metric_key, entry.value)
                if entry.tags:
                    self._runs.set_tags(self._run, entry.tags)
        except RunStoreError as error:
            warn(f"Could not log metric: {error}")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Add user tags to this solve.

        Parameters
        ----------
        tags : Mapping[str, str]
            Tags to add; keys beginning with ``optimization.`` or ``mlflow.``
            are ignored.
        """
        try:
            accepted: dict[str, str] = user_tags(tags)
            if accepted:
                self._runs.set_tags(self._run, accepted)
        except RunStoreError as error:
            warn(f"Could not set tags: {error}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Attach a local file to this solve.

        Parameters
        ----------
        local_path : str
            Path to an existing local file.
        artifact_path : str, optional
            Destination directory in run artifacts; if omitted, the file goes
            to the artifact root.
        """
        try:
            refusal: str | None = artifact_refusal(
                local_path, enabled=self._options.log_artifacts
            )
            if refusal is not None:
                warn(refusal)
                return
            self._artifacts.log_artifact(self.run_id, local_path, artifact_path)
        except ArtifactStoreError as error:
            warn(f"Could not log artifact: {error}")


_last_solve: ContextVar[SolveRun | None] = ContextVar("last_solve", default=None)


class RecordSolve:
    """The use case that stores each completed solve as one run."""

    def __init__(
        self, runs: RunStore, artifacts: ArtifactStore, options: RecordOptions
    ) -> None:
        self._runs: RunStore = runs
        self._artifacts: ArtifactStore = artifacts
        self.options: RecordOptions = options

    def begin(self) -> object:
        """The parent run and annotations in effect at solve start."""
        parent: RunReference | None
        try:
            parent = self._runs.active_run()
        except RunStoreError as error:
            warn(f"Could not get active run: {error}")
            parent = None
        return _RecordingContext(parent, current_annotation())

    def record(self, solve: CompletedSolve, context: object) -> None:
        """Store one completed solve."""
        if not isinstance(context, _RecordingContext):
            raise TypeError("Invalid recording context")
        observation: SolveObservation | None = solve.observation
        has_objective: bool | None = (
            observation.metrics.has_objective if observation is not None else None
        )
        outcome: SolveOutcome = classify_outcome(
            observation.outcome if observation else None,
            solve.error,
            has_objective=has_objective,
        )
        metrics: NormalizedMetrics = normalize_metrics(
            observation.metrics if observation else None,
            api=solve.api,
            has_solution=outcome.has_solution,
            wall_time_seconds=solve.wall_time_seconds,
            include_native=self.options.log_solver_metrics,
        )
        metadata: CollectedMetadata = collect_metadata(
            solve.api,
            observation.metadata if observation else None,
            include_parameters=self.options.log_solver_parameters,
        )
        tags: dict[str, str] = dict(annotation_tags(context.annotation))
        tags.update(outcome_tags(outcome))
        tags.update(metrics.tags)
        tags.update(metadata.tags)
        tags["optimization.solve_id"] = solve.solve_id
        tags["optimization.solve_index"] = str(solve.solve_index)
        if solve.error is not None:
            tags["optimization.error"] = type(solve.error).__name__
        status: RunStatus = RunStatus.FINISHED
        if isinstance(solve.error, KeyboardInterrupt):
            status = RunStatus.KILLED
        elif isinstance(solve.error, Exception):
            status = RunStatus.FAILED
        solver: str = metadata.tags.get("optimization.solver") or "unknown"
        record: SolveRecord = SolveRecord(
            name=f"{solve.api}/{solver} #{solve.solve_index}",
            status=status,
            start_time_ms=solve.start_time_ms,
            end_time_ms=solve.end_time_ms,
            tags=tags,
            metrics=metrics.values,
            params=metadata.params,
        )
        run: RunReference
        try:
            run = self._runs.save(record, context.parent)
        except RunStoreError as error:
            warn(f"Could not record solve: {error}")
            return
        _last_solve.set(SolveRun(run, self._runs, self._artifacts, self.options))


def last_solve() -> SolveRun | None:
    """The most recent solve recorded in the current thread or async context.

    Returns
    -------
    SolveRun or None
        None until a solve is recorded in the current context.
    """
    return _last_solve.get()
