"""MLflow implementation of the run storage."""

from __future__ import annotations

import base64
import zlib
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import mlflow
from mlflow.entities import Metric, Param, Run, RunTag
from mlflow.protos.service_pb2 import LogBatch
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.fluent import ActiveRun, _get_experiment_id
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.validation import (
    MAX_BATCH_LOG_REQUEST_SIZE,
    MAX_ENTITY_KEY_LENGTH,
    MAX_METRICS_PER_BATCH,
    MAX_PARAM_VAL_LENGTH,
    MAX_PARAMS_TAGS_PER_BATCH,
    MAX_TAG_VAL_LENGTH,
    path_not_unique,
    validate_param_and_metric_name,
)

from mlflow_optimization.common.diagnostics import warn
from mlflow_optimization.features.runs.ports import RunStoreError
from mlflow_optimization.features.runs.types import (
    HistoryPoint,
    RunReference,
    SolveRecord,
)

_ENCODED_KEY_PREFIX: str = "_mlflow_key_"
_BATCH_LOG_REQUEST_SIZE_SAFETY_MARGIN: int = MAX_BATCH_LOG_REQUEST_SIZE // 10
_MAX_BATCH_LOG_REQUEST_SIZE_BUDGET: int = (
    MAX_BATCH_LOG_REQUEST_SIZE - _BATCH_LOG_REQUEST_SIZE_SAFETY_MARGIN
)
_SOLVE_IDENTITY_TAG_KEYS: tuple[str, ...] = (
    "optimization.solve_id",
    "optimization.solve_index",
    "optimization.error",
)


class MlflowRunStore:
    """Solve runs stored through the MLflow tracking client."""

    def active_run(self) -> RunReference | None:
        active_mlflow_run: ActiveRun | None = _run_store_sdk_call(
            "reading the active run",
            mlflow.active_run,
        )
        if active_mlflow_run is None:
            return None
        active_run_id: str = _run_store_sdk_call(
            "reading the active run",
            lambda: active_mlflow_run.info.run_id,
        )
        active_experiment_id: str = _run_store_sdk_call(
            "reading the active run",
            lambda: active_mlflow_run.info.experiment_id,
        )
        return RunReference(
            run_id=active_run_id,
            experiment_id=active_experiment_id,
        )

    def start(
        self,
        name: str,
        start_time_ms: int,
        tags: Mapping[str, str],
        parent: RunReference | None,
    ) -> RunReference:
        operation: str = "starting the run"
        tracking_uri: str = _run_store_sdk_call(operation, mlflow.get_tracking_uri)
        tracking_client: MlflowClient = _run_store_sdk_call(
            operation,
            lambda: MlflowClient(tracking_uri=tracking_uri),
        )
        experiment_id: str
        run_name: str
        context_tag_values: dict[str, Any]
        creation_tag_values: dict[str, str]
        (
            experiment_id,
            run_name,
            context_tag_values,
            creation_tag_values,
        ) = self._prepare_run(name, tags, parent, operation)
        created_run: Run = self._create_run(
            tracking_client,
            experiment_id,
            start_time_ms,
            run_name,
            creation_tag_values,
            operation,
        )
        created_run_id: str = _run_store_sdk_call(
            operation,
            lambda: created_run.info.run_id,
        )
        try:
            self._log_tag_batches(
                tracking_client,
                created_run_id,
                _make_tag_entities(context_tag_values, operation),
                operation,
            )
        except RunStoreError as error:
            # NOTE: Keep the created run; finish() retries its tags after the solve.
            warn(f"Could not log tags at solve start: {error}")
        created_experiment_id: str = _run_store_sdk_call(
            operation,
            lambda: created_run.info.experiment_id,
        )
        return RunReference(
            run_id=created_run_id,
            experiment_id=created_experiment_id,
            tracking_uri=tracking_uri,
        )

    def log_history(
        self,
        run: RunReference,
        points: Sequence[HistoryPoint],
    ) -> None:
        operation: str = "logging run history"
        tracking_client: MlflowClient = _run_store_sdk_call(
            operation,
            lambda: MlflowClient(tracking_uri=run.tracking_uri),
        )
        metric_entities: list[Metric] = []
        for point in points:
            for metric_key, metric_value in point.values.items():
                encoded_metric_key: str | None = _encode_key(metric_key, operation)
                if encoded_metric_key is None:
                    continue
                metric_entities.append(
                    _run_store_sdk_call(
                        operation,
                        partial(
                            Metric,
                            key=encoded_metric_key,
                            value=metric_value,
                            timestamp=point.timestamp_ms,
                            step=point.step,
                        ),
                    )
                )
        self._log_metric_batches(
            tracking_client,
            run.run_id,
            metric_entities,
            operation,
        )

    def finish(
        self,
        run: RunReference,
        record: SolveRecord,
        *,
        final_step: int,
    ) -> None:
        operation: str = "finishing the run"
        tracking_client: MlflowClient = _run_store_sdk_call(
            operation,
            lambda: MlflowClient(tracking_uri=run.tracking_uri),
        )
        run_name: str = _truncate_tag_value(record.name)
        context_tag_values: dict[str, Any]
        try:
            context_tag_values = self._resolve_context_tag_values(
                run_name,
                record.tags,
                None,
                operation,
            )
            _run_store_sdk_call(
                operation,
                lambda: tracking_client.update_run(
                    run_id=run.run_id,
                    name=run_name,
                ),
            )
            self._log_solve_record(
                tracking_client,
                run.run_id,
                record,
                final_step,
                context_tag_values,
                operation,
            )
        finally:
            _run_store_sdk_call(
                operation,
                lambda: tracking_client.set_terminated(
                    run_id=run.run_id,
                    status=record.status.value,
                    end_time=record.end_time_ms,
                ),
            )

    def save(self, record: SolveRecord, parent: RunReference | None) -> RunReference:
        tracking_uri: str = _run_store_sdk_call(
            "saving the run", mlflow.get_tracking_uri
        )
        tracking_client: MlflowClient = _run_store_sdk_call(
            "saving the run",
            lambda: MlflowClient(tracking_uri=tracking_uri),
        )
        experiment_id: str
        run_name: str
        context_tag_values: dict[str, Any]
        creation_tag_values: dict[str, str]
        (
            experiment_id,
            run_name,
            context_tag_values,
            creation_tag_values,
        ) = self._prepare_run(record.name, record.tags, parent, "saving the run")
        created_run: Run = self._create_run(
            tracking_client,
            experiment_id,
            record.start_time_ms,
            run_name,
            creation_tag_values,
            "saving the run",
        )
        created_run_id: str = _run_store_sdk_call(
            "saving the run",
            lambda: created_run.info.run_id,
        )

        try:
            self._log_solve_record(
                tracking_client,
                created_run_id,
                record,
                0,
                context_tag_values,
                "saving the run",
            )
        finally:
            _run_store_sdk_call(
                "saving the run",
                lambda: tracking_client.set_terminated(
                    run_id=created_run_id,
                    status=record.status.value,
                    end_time=record.end_time_ms,
                ),
            )
        created_experiment_id: str = _run_store_sdk_call(
            "saving the run",
            lambda: created_run.info.experiment_id,
        )
        return RunReference(
            run_id=created_run_id,
            experiment_id=created_experiment_id,
        )

    def _prepare_run(
        self,
        name: str,
        tags: Mapping[str, str],
        parent: RunReference | None,
        operation: str,
    ) -> tuple[str, str, dict[str, Any], dict[str, str]]:
        """The experiment, name, and tags for a run."""
        experiment_id: str
        if parent is None:
            resolved_experiment_id: str | None = _run_store_sdk_call(
                operation,
                _get_experiment_id,
            )
            if resolved_experiment_id is None:
                raise RunStoreError(
                    f"Failed while {operation}: MLflow did not resolve "
                    "an experiment ID."
                )
            experiment_id = resolved_experiment_id
        else:
            experiment_id = parent.experiment_id

        run_name: str = _truncate_tag_value(name)
        context_tag_values: dict[str, Any] = self._resolve_context_tag_values(
            run_name,
            tags,
            parent,
            operation,
        )
        creation_tag_values: dict[str, str] = {}
        for identity_tag_key in _SOLVE_IDENTITY_TAG_KEYS:
            if identity_tag_key in context_tag_values:
                creation_tag_values[identity_tag_key] = _truncate_tag_value(
                    str(context_tag_values[identity_tag_key])
                )
        if parent is not None:
            creation_tag_values[MLFLOW_PARENT_RUN_ID] = parent.run_id
        return experiment_id, run_name, context_tag_values, creation_tag_values

    def _resolve_context_tag_values(
        self,
        run_name: str,
        tags: Mapping[str, str],
        parent: RunReference | None,
        operation: str,
    ) -> dict[str, Any]:
        """The resolved MLflow context and solve tags."""
        resolved_tag_values: dict[str, Any] = dict(tags)
        if parent is not None:
            resolved_tag_values[MLFLOW_PARENT_RUN_ID] = parent.run_id
        if run_name:
            resolved_tag_values[MLFLOW_RUN_NAME] = run_name
        return _run_store_sdk_call(
            operation,
            lambda: resolve_tags(resolved_tag_values),
        )

    def _create_run(
        self,
        tracking_client: MlflowClient,
        experiment_id: str,
        start_time_ms: int,
        run_name: str,
        creation_tag_values: dict[str, str],
        operation: str,
    ) -> Run:
        """A new MLflow run."""
        return _run_store_sdk_call(
            operation,
            lambda: tracking_client.create_run(
                experiment_id=experiment_id,
                start_time=start_time_ms,
                tags=creation_tag_values,
                run_name=run_name,
            ),
        )

    def _log_solve_record(
        self,
        tracking_client: MlflowClient,
        run_id: str,
        record: SolveRecord,
        final_step: int,
        context_tag_values: Mapping[str, Any],
        operation: str,
    ) -> None:
        """The final tags, parameters, and metrics of a solve."""
        metric_entities: list[Metric] = []
        for metric_key, metric_value in record.metrics.items():
            encoded_metric_key: str | None = _encode_key(metric_key, operation)
            if encoded_metric_key is None:
                continue
            metric_entities.append(
                _run_store_sdk_call(
                    operation,
                    partial(
                        Metric,
                        key=encoded_metric_key,
                        value=metric_value,
                        timestamp=record.end_time_ms,
                        step=final_step,
                    ),
                )
            )

        parameter_entities: list[Param] = []
        for parameter_key, parameter_value in record.params.items():
            encoded_parameter_key: str | None = _encode_key(parameter_key, operation)
            if encoded_parameter_key is None:
                continue
            parameter_entities.append(
                _run_store_sdk_call(
                    operation,
                    partial(
                        Param,
                        key=encoded_parameter_key,
                        value=parameter_value[:MAX_PARAM_VAL_LENGTH],
                    ),
                )
            )

        self._log_metric_batches(
            tracking_client,
            run_id,
            metric_entities,
            operation,
        )
        self._log_parameter_batches(
            tracking_client,
            run_id,
            parameter_entities,
            operation,
        )
        self._log_tag_batches(
            tracking_client,
            run_id,
            _make_tag_entities(context_tag_values, operation),
            operation,
        )

    def log_metric(self, run: RunReference, key: str, value: float) -> None:
        tracking_uri: str = _run_store_sdk_call(
            "logging a metric", mlflow.get_tracking_uri
        )
        tracking_client: MlflowClient = _run_store_sdk_call(
            "logging a metric",
            lambda: MlflowClient(tracking_uri=tracking_uri),
        )
        encoded_key: str | None = _encode_key(key, "logging a metric")
        if encoded_key is None:
            return
        _run_store_sdk_call(
            "logging a metric",
            lambda: tracking_client.log_metric(
                run_id=run.run_id,
                key=encoded_key,
                value=value,
                synchronous=True,
            ),
        )

    def set_tags(self, run: RunReference, tags: Mapping[str, str]) -> None:
        tracking_uri: str = _run_store_sdk_call("setting tags", mlflow.get_tracking_uri)
        tracking_client: MlflowClient = _run_store_sdk_call(
            "setting tags",
            lambda: MlflowClient(tracking_uri=tracking_uri),
        )
        self._log_tag_batches(
            tracking_client,
            run.run_id,
            _make_tag_entities(tags, "setting tags"),
            "setting tags",
        )

    def _log_metric_batches(
        self,
        tracking_client: MlflowClient,
        run_id: str,
        metric_entities: list[Metric],
        operation: str,
    ) -> None:
        current_batch: list[Metric] = []
        for metric_entity in metric_entities:
            candidate_batch: list[Metric] = [*current_batch, metric_entity]
            if current_batch and (
                len(current_batch) >= MAX_METRICS_PER_BATCH
                or _run_store_sdk_call(
                    operation,
                    partial(
                        _serialized_batch_request_size,
                        run_id,
                        candidate_batch,
                        [],
                        [],
                    ),
                )
                > _MAX_BATCH_LOG_REQUEST_SIZE_BUDGET
            ):
                _run_store_sdk_call(
                    operation,
                    partial(
                        tracking_client.log_batch,
                        run_id=run_id,
                        metrics=current_batch,
                        synchronous=True,
                    ),
                )
                current_batch = []
            current_batch.append(metric_entity)
        if current_batch:
            _run_store_sdk_call(
                operation,
                partial(
                    tracking_client.log_batch,
                    run_id=run_id,
                    metrics=current_batch,
                    synchronous=True,
                ),
            )

    def _log_parameter_batches(
        self,
        tracking_client: MlflowClient,
        run_id: str,
        parameter_entities: list[Param],
        operation: str,
    ) -> None:
        current_batch: list[Param] = []
        for parameter_entity in parameter_entities:
            candidate_batch: list[Param] = [*current_batch, parameter_entity]
            if current_batch and (
                len(current_batch) >= MAX_PARAMS_TAGS_PER_BATCH
                or _run_store_sdk_call(
                    operation,
                    partial(
                        _serialized_batch_request_size,
                        run_id,
                        [],
                        candidate_batch,
                        [],
                    ),
                )
                > _MAX_BATCH_LOG_REQUEST_SIZE_BUDGET
            ):
                _run_store_sdk_call(
                    operation,
                    partial(
                        tracking_client.log_batch,
                        run_id=run_id,
                        params=current_batch,
                        synchronous=True,
                    ),
                )
                current_batch = []
            current_batch.append(parameter_entity)
        if current_batch:
            _run_store_sdk_call(
                operation,
                partial(
                    tracking_client.log_batch,
                    run_id=run_id,
                    params=current_batch,
                    synchronous=True,
                ),
            )

    def _log_tag_batches(
        self,
        tracking_client: MlflowClient,
        run_id: str,
        tag_entities: list[RunTag],
        operation: str,
    ) -> None:
        current_batch: list[RunTag] = []
        for tag_entity in tag_entities:
            candidate_batch: list[RunTag] = [*current_batch, tag_entity]
            if current_batch and (
                len(current_batch) >= MAX_PARAMS_TAGS_PER_BATCH
                or _run_store_sdk_call(
                    operation,
                    partial(
                        _serialized_batch_request_size,
                        run_id,
                        [],
                        [],
                        candidate_batch,
                    ),
                )
                > _MAX_BATCH_LOG_REQUEST_SIZE_BUDGET
            ):
                _run_store_sdk_call(
                    operation,
                    partial(
                        tracking_client.log_batch,
                        run_id=run_id,
                        tags=current_batch,
                        synchronous=True,
                    ),
                )
                current_batch = []
            current_batch.append(tag_entity)
        if current_batch:
            _run_store_sdk_call(
                operation,
                partial(
                    tracking_client.log_batch,
                    run_id=run_id,
                    tags=current_batch,
                    synchronous=True,
                ),
            )


def _encode_key(key: str, operation: str = "saving the run") -> str | None:
    valid_key: bool = bool(
        _run_store_sdk_call(
            operation,
            lambda: validate_param_and_metric_name(key),
        )
    )
    unique_path: bool = not _run_store_sdk_call(
        operation,
        lambda: path_not_unique(key),
    )
    if (
        valid_key
        and unique_path
        and len(key) <= MAX_ENTITY_KEY_LENGTH
        and not key.startswith(_ENCODED_KEY_PREFIX)
    ):
        return key

    encoded_characters: list[str] = []
    for character in key:
        if character == "_":
            encoded_characters.append("__")
        elif character == "/":
            encoded_characters.append("_x2F_")
        elif _run_store_sdk_call(
            operation,
            partial(validate_param_and_metric_name, character),
        ):
            encoded_characters.append(character)
        else:
            encoded_characters.append(f"_x{ord(character):X}_")

    escaped_key: str = _ENCODED_KEY_PREFIX + "e_" + "".join(encoded_characters)
    if len(escaped_key) <= MAX_ENTITY_KEY_LENGTH:
        return escaped_key

    key_bytes: bytes = key.encode("utf-8", errors="surrogatepass")
    compressed_key_bytes: bytes = zlib.compress(key_bytes, level=9)
    base64_key_bytes: bytes = base64.urlsafe_b64encode(compressed_key_bytes)
    compressed_key: str = (
        _ENCODED_KEY_PREFIX + "z_" + base64_key_bytes.decode("ascii").rstrip("=")
    )
    if len(compressed_key) > MAX_ENTITY_KEY_LENGTH:
        warn(f"Skipping MLflow key {key!r} because it exceeds the tracker key limit.")
        return None
    return compressed_key


def _truncate_tag_value(value: str) -> str:
    return value[:MAX_TAG_VAL_LENGTH]


def _make_tag_entities(tag_values: Mapping[str, Any], operation: str) -> list[RunTag]:
    tag_entities: list[RunTag] = []
    for tag_key, tag_value in tag_values.items():
        encoded_tag_key: str | None = _encode_key(tag_key, operation)
        if encoded_tag_key is None:
            continue
        truncated_tag_value: str = _truncate_tag_value(str(tag_value))
        tag_entities.append(
            _run_store_sdk_call(
                operation,
                partial(
                    RunTag,
                    key=encoded_tag_key,
                    value=truncated_tag_value,
                ),
            )
        )
    return tag_entities


def _serialized_batch_request_size(
    run_id: str,
    metric_entities: list[Metric],
    parameter_entities: list[Param],
    tag_entities: list[RunTag],
) -> int:
    metric_protocol_buffers: list[Any] = []
    for metric_entity in metric_entities:
        metric_protocol_buffers.append(metric_entity.to_proto())
    parameter_protocol_buffers: list[Any] = []
    for parameter_entity in parameter_entities:
        parameter_protocol_buffers.append(parameter_entity.to_proto())
    tag_protocol_buffers: list[Any] = []
    for tag_entity in tag_entities:
        tag_protocol_buffers.append(tag_entity.to_proto())
    batch_message: LogBatch = LogBatch(
        run_id=run_id,
        metrics=metric_protocol_buffers,
        params=parameter_protocol_buffers,
        tags=tag_protocol_buffers,
    )
    serialized_request: str = message_to_json(batch_message)
    return len(serialized_request.encode("utf-8"))


def _run_store_sdk_call[ReturnType](
    operation: str,
    mlflow_call: Callable[[], ReturnType],
) -> ReturnType:
    """An MLflow SDK call converted to a run-store error."""
    try:
        return mlflow_call()
    except Exception as error:
        raise RunStoreError(
            f"Failed while {operation}: {type(error).__name__}: {error}"
        ) from error
