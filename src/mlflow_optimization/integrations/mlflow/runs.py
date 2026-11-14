"""MLflow implementation of the run storage."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import mlflow
from mlflow.entities import Metric, Param, Run, RunTag
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.fluent import ActiveRun, _get_experiment_id
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from mlflow.utils.server_info import ServerInfoRequestError
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
from requests.exceptions import RequestException

from mlflow_optimization.features.runs.models import RunRef, SolveRecord
from mlflow_optimization.features.runs.ports import RunStoreError


class MlflowRunStore:
    """Solve runs stored through the MLflow tracking client."""

    def active_run(self: MlflowRunStore) -> RunRef | None:
        try:
            active_mlflow_run: ActiveRun | None = mlflow.active_run()
            if active_mlflow_run is None:
                return None
            return RunRef(
                run_id=active_mlflow_run.info.run_id,
                experiment_id=active_mlflow_run.info.experiment_id,
            )
        except (MlflowException, ServerInfoRequestError, RequestException, OSError) as error:
            raise RunStoreError("MLflow run storage failed.") from error

    def save(self: MlflowRunStore, record: SolveRecord, parent: RunRef | None) -> RunRef:
        try:
            return self._save(record, parent)
        except (MlflowException, ServerInfoRequestError, RequestException, OSError) as error:
            raise RunStoreError("MLflow run storage failed.") from error

    def _save(self: MlflowRunStore, record: SolveRecord, parent: RunRef | None) -> RunRef:
        tracking_client: MlflowClient = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        experiment_id: str
        if parent is None:
            resolved_experiment_id: str | None = _get_experiment_id()
            if resolved_experiment_id is None:
                raise RunStoreError("MLflow did not resolve an experiment ID.")
            experiment_id = resolved_experiment_id
        else:
            experiment_id = parent.experiment_id

        run_name: str = _truncate_tag_value(record.name)
        resolved_tag_values: dict[str, Any] = dict(record.tags)
        if parent is not None:
            resolved_tag_values[MLFLOW_PARENT_RUN_ID] = parent.run_id
        if run_name:
            resolved_tag_values[MLFLOW_RUN_NAME] = run_name
        context_tag_values: dict[str, Any] = resolve_tags(resolved_tag_values)

        created_run: Run = tracking_client.create_run(
            experiment_id=experiment_id,
            start_time=record.start_time_ms,
            run_name=run_name,
        )

        metric_entities: list[Metric] = []
        metric_key: str
        metric_value: float
        for metric_key, metric_value in record.metrics.items():
            metric_entities.append(
                Metric(
                    key=_sanitize_key(metric_key),
                    value=metric_value,
                    timestamp=record.end_time_ms,
                    step=0,
                )
            )

        parameter_entities: list[Param] = []
        parameter_key: str
        parameter_value: str
        for parameter_key, parameter_value in record.params.items():
            parameter_entities.append(
                Param(
                    key=_sanitize_key(parameter_key),
                    value=parameter_value[:MAX_PARAM_VAL_LENGTH],
                )
            )

        self._log_metric_batches(tracking_client, created_run.info.run_id, metric_entities)
        self._log_parameter_batches(tracking_client, created_run.info.run_id, parameter_entities)
        self._log_tag_batches(
            tracking_client,
            created_run.info.run_id,
            _make_tag_entities(context_tag_values),
        )
        tracking_client.set_terminated(
            run_id=created_run.info.run_id,
            status=record.status.value,
            end_time=record.end_time_ms,
        )
        return RunRef(
            run_id=created_run.info.run_id,
            experiment_id=created_run.info.experiment_id,
        )

    def log_metric(self: MlflowRunStore, run: RunRef, key: str, value: float) -> None:
        try:
            tracking_client: MlflowClient = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            tracking_client.log_metric(
                run_id=run.run_id,
                key=_sanitize_key(key),
                value=value,
                synchronous=True,
            )
        except (MlflowException, ServerInfoRequestError, RequestException, OSError) as error:
            raise RunStoreError("MLflow run storage failed.") from error

    def set_tags(self: MlflowRunStore, run: RunRef, tags: Mapping[str, str]) -> None:
        try:
            tracking_client: MlflowClient = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
            self._log_tag_batches(tracking_client, run.run_id, _make_tag_entities(tags))
        except (MlflowException, ServerInfoRequestError, RequestException, OSError) as error:
            raise RunStoreError("MLflow run storage failed.") from error

    def _log_metric_batches(
        self: MlflowRunStore,
        tracking_client: MlflowClient,
        run_id: str,
        metric_entities: list[Metric],
    ) -> None:
        current_batch: list[Metric] = []
        current_batch_size: int = _empty_batch_request_size(run_id)
        metric_entity: Metric
        for metric_entity in metric_entities:
            metric_payload: dict[str, str | float | int] = {
                "key": metric_entity.key,
                "value": metric_entity.value,
                "timestamp": metric_entity.timestamp,
                "step": metric_entity.step,
            }
            metric_payload_size: int = _serialized_item_size(metric_payload)
            added_size: int = metric_payload_size + (1 if current_batch else 0)
            if current_batch and (
                len(current_batch) >= MAX_METRICS_PER_BATCH
                or current_batch_size + added_size > MAX_BATCH_LOG_REQUEST_SIZE
            ):
                tracking_client.log_batch(
                    run_id=run_id,
                    metrics=current_batch,
                    synchronous=True,
                )
                current_batch = []
                current_batch_size = _empty_batch_request_size(run_id)
                added_size = metric_payload_size
            current_batch.append(metric_entity)
            current_batch_size += added_size
        if current_batch:
            tracking_client.log_batch(
                run_id=run_id,
                metrics=current_batch,
                synchronous=True,
            )

    def _log_parameter_batches(
        self: MlflowRunStore,
        tracking_client: MlflowClient,
        run_id: str,
        parameter_entities: list[Param],
    ) -> None:
        current_batch: list[Param] = []
        current_batch_size: int = _empty_batch_request_size(run_id)
        parameter_entity: Param
        for parameter_entity in parameter_entities:
            parameter_payload: dict[str, str | float | int] = {
                "key": parameter_entity.key,
                "value": parameter_entity.value,
            }
            parameter_payload_size: int = _serialized_item_size(parameter_payload)
            added_size: int = parameter_payload_size + (1 if current_batch else 0)
            if current_batch and (
                len(current_batch) >= MAX_PARAMS_TAGS_PER_BATCH
                or current_batch_size + added_size > MAX_BATCH_LOG_REQUEST_SIZE
            ):
                tracking_client.log_batch(
                    run_id=run_id,
                    params=current_batch,
                    synchronous=True,
                )
                current_batch = []
                current_batch_size = _empty_batch_request_size(run_id)
                added_size = parameter_payload_size
            current_batch.append(parameter_entity)
            current_batch_size += added_size
        if current_batch:
            tracking_client.log_batch(
                run_id=run_id,
                params=current_batch,
                synchronous=True,
            )

    def _log_tag_batches(
        self: MlflowRunStore,
        tracking_client: MlflowClient,
        run_id: str,
        tag_entities: list[RunTag],
    ) -> None:
        current_batch: list[RunTag] = []
        current_batch_size: int = _empty_batch_request_size(run_id)
        tag_entity: RunTag
        for tag_entity in tag_entities:
            tag_payload: dict[str, str | float | int] = {
                "key": tag_entity.key,
                "value": tag_entity.value,
            }
            tag_payload_size: int = _serialized_item_size(tag_payload)
            added_size: int = tag_payload_size + (1 if current_batch else 0)
            if current_batch and (
                len(current_batch) >= MAX_PARAMS_TAGS_PER_BATCH
                or current_batch_size + added_size > MAX_BATCH_LOG_REQUEST_SIZE
            ):
                tracking_client.log_batch(
                    run_id=run_id,
                    tags=current_batch,
                    synchronous=True,
                )
                current_batch = []
                current_batch_size = _empty_batch_request_size(run_id)
                added_size = tag_payload_size
            current_batch.append(tag_entity)
            current_batch_size += added_size
        if current_batch:
            tracking_client.log_batch(
                run_id=run_id,
                tags=current_batch,
                synchronous=True,
            )


def _sanitize_key(key: str) -> str:
    sanitized_characters: list[str] = []
    character: str
    for character in key:
        if validate_param_and_metric_name(character):
            sanitized_characters.append(character)
        else:
            sanitized_characters.append("_")

    safe_segments: list[str] = []
    segment: str
    for segment in "".join(sanitized_characters).split("/"):
        if not segment or segment == ".":
            continue
        if segment == "..":
            safe_segments.append("_")
        else:
            safe_segments.append(segment)

    sanitized_key: str = "/".join(safe_segments)[:MAX_ENTITY_KEY_LENGTH].rstrip("/")
    if not sanitized_key or path_not_unique(sanitized_key):
        return "_"
    return sanitized_key


def _truncate_tag_value(value: str) -> str:
    return value[:MAX_TAG_VAL_LENGTH]


def _make_tag_entities(tag_values: Mapping[str, Any]) -> list[RunTag]:
    tag_entities: list[RunTag] = []
    tag_key: str
    tag_value: Any
    for tag_key, tag_value in tag_values.items():
        tag_entities.append(
            RunTag(
                key=_sanitize_key(tag_key),
                value=_truncate_tag_value(str(tag_value)),
            )
        )
    return tag_entities


def _empty_batch_request_size(run_id: str) -> int:
    empty_batch: dict[str, str | list[str]] = {
        "run_id": run_id,
        "metrics": [],
        "params": [],
        "tags": [],
    }
    return len(json.dumps(empty_batch, ensure_ascii=True, separators=(",", ":")).encode("utf-8"))


def _serialized_item_size(item: dict[str, str | float | int]) -> int:
    serialized_item: str = json.dumps(item, ensure_ascii=True, separators=(",", ":"))
    return len(serialized_item.encode("utf-8"))
