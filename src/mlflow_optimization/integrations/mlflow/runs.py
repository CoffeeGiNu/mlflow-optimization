"""MLflow implementation of the run storage."""

from __future__ import annotations

import base64
import zlib
from collections.abc import Callable, Mapping
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
from mlflow_optimization.features.runs.types import RunReference, SolveRecord

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

    def active_run(self: MlflowRunStore) -> RunReference | None:
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

    def save(
        self: MlflowRunStore, record: SolveRecord, parent: RunReference | None
    ) -> RunReference:
        tracking_uri: str = _run_store_sdk_call(
            "saving the run", mlflow.get_tracking_uri
        )
        tracking_client: MlflowClient = _run_store_sdk_call(
            "saving the run",
            lambda: MlflowClient(tracking_uri=tracking_uri),
        )
        experiment_id: str
        if parent is None:
            resolved_experiment_id: str | None = _run_store_sdk_call(
                "saving the run",
                _get_experiment_id,
            )
            if resolved_experiment_id is None:
                raise RunStoreError(
                    "Failed while saving the run: MLflow did not resolve an experiment ID."
                )
            experiment_id = resolved_experiment_id
        else:
            experiment_id = parent.experiment_id

        run_name: str = _truncate_tag_value(record.name)
        resolved_tag_values: dict[str, Any] = dict(record.tags)
        if parent is not None:
            resolved_tag_values[MLFLOW_PARENT_RUN_ID] = parent.run_id
        if run_name:
            resolved_tag_values[MLFLOW_RUN_NAME] = run_name
        context_tag_values: dict[str, Any] = _run_store_sdk_call(
            "saving the run",
            lambda: resolve_tags(resolved_tag_values),
        )

        creation_tag_values: dict[str, str] = {}
        for identity_tag_key in _SOLVE_IDENTITY_TAG_KEYS:
            if identity_tag_key in context_tag_values:
                creation_tag_values[identity_tag_key] = _truncate_tag_value(
                    str(context_tag_values[identity_tag_key])
                )
        if parent is not None:
            creation_tag_values[MLFLOW_PARENT_RUN_ID] = parent.run_id

        created_run: Run = _run_store_sdk_call(
            "saving the run",
            lambda: tracking_client.create_run(
                experiment_id=experiment_id,
                start_time=record.start_time_ms,
                tags=creation_tag_values,
                run_name=run_name,
            ),
        )
        created_run_id: str = _run_store_sdk_call(
            "saving the run",
            lambda: created_run.info.run_id,
        )

        try:
            metric_entities: list[Metric] = []
            for metric_key, metric_value in record.metrics.items():
                encoded_metric_key: str | None = _encode_key(
                    metric_key, "saving the run"
                )
                if encoded_metric_key is None:
                    continue
                metric_entities.append(
                    _run_store_sdk_call(
                        "saving the run",
                        partial(
                            Metric,
                            key=encoded_metric_key,
                            value=metric_value,
                            timestamp=record.end_time_ms,
                            step=0,
                        ),
                    )
                )

            parameter_entities: list[Param] = []
            for parameter_key, parameter_value in record.params.items():
                encoded_parameter_key: str | None = _encode_key(
                    parameter_key,
                    "saving the run",
                )
                if encoded_parameter_key is None:
                    continue
                parameter_entities.append(
                    _run_store_sdk_call(
                        "saving the run",
                        partial(
                            Param,
                            key=encoded_parameter_key,
                            value=parameter_value[:MAX_PARAM_VAL_LENGTH],
                        ),
                    )
                )

            self._log_metric_batches(tracking_client, created_run_id, metric_entities)
            self._log_parameter_batches(
                tracking_client,
                created_run_id,
                parameter_entities,
            )
            self._log_tag_batches(
                tracking_client,
                created_run_id,
                _make_tag_entities(context_tag_values, "saving the run"),
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

    def log_metric(
        self: MlflowRunStore, run: RunReference, key: str, value: float
    ) -> None:
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

    def set_tags(
        self: MlflowRunStore, run: RunReference, tags: Mapping[str, str]
    ) -> None:
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
        self: MlflowRunStore,
        tracking_client: MlflowClient,
        run_id: str,
        metric_entities: list[Metric],
    ) -> None:
        current_batch: list[Metric] = []
        for metric_entity in metric_entities:
            candidate_batch: list[Metric] = [*current_batch, metric_entity]
            if current_batch and (
                len(current_batch) >= MAX_METRICS_PER_BATCH
                or _run_store_sdk_call(
                    "saving the run",
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
                    "saving the run",
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
                "saving the run",
                partial(
                    tracking_client.log_batch,
                    run_id=run_id,
                    metrics=current_batch,
                    synchronous=True,
                ),
            )

    def _log_parameter_batches(
        self: MlflowRunStore,
        tracking_client: MlflowClient,
        run_id: str,
        parameter_entities: list[Param],
    ) -> None:
        current_batch: list[Param] = []
        for parameter_entity in parameter_entities:
            candidate_batch: list[Param] = [*current_batch, parameter_entity]
            if current_batch and (
                len(current_batch) >= MAX_PARAMS_TAGS_PER_BATCH
                or _run_store_sdk_call(
                    "saving the run",
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
                    "saving the run",
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
                "saving the run",
                partial(
                    tracking_client.log_batch,
                    run_id=run_id,
                    params=current_batch,
                    synchronous=True,
                ),
            )

    def _log_tag_batches(
        self: MlflowRunStore,
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
