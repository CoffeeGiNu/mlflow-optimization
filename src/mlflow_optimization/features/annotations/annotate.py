"""Scoping and naming rules for user annotations."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from math import isfinite

from mlflow_optimization.common.diagnostics import warn

from .types import Annotation, UserMetric


_annotation_context: ContextVar[Annotation] = ContextVar(
    "annotation_context", default=Annotation()
)


@contextmanager
def context(
    *,
    instance: str | None = None,
    description: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """A scope whose annotations apply to every solve started inside it."""
    enclosing_annotation: Annotation = current_annotation()
    merged_tags: dict[str, str] = dict(enclosing_annotation.tags)
    if tags is not None:
        merged_tags.update(user_tags(tags))
    merged_annotation: Annotation = Annotation(
        instance=instance if instance is not None else enclosing_annotation.instance,
        description=(
            description
            if description is not None
            else enclosing_annotation.description
        ),
        tags=merged_tags,
    )
    token: Token[Annotation] = _annotation_context.set(merged_annotation)
    try:
        yield
    finally:
        _annotation_context.reset(token)


def current_annotation() -> Annotation:
    """The merged annotations of the enclosing ``context`` scopes."""
    return _annotation_context.get()


def annotation_tags(annotation: Annotation) -> dict[str, str]:
    """The tags that record an annotation."""
    tags: dict[str, str] = user_tags(annotation.tags)
    if annotation.instance is not None:
        tags["optimization.instance"] = annotation.instance
    if annotation.description is not None:
        tags["mlflow.note.content"] = annotation.description
    return tags


def user_tags(tags: Mapping[str, str]) -> dict[str, str]:
    """The subset of user tags that may be recorded."""
    accepted_tags: dict[str, str] = {}
    key: str
    value: str
    for key, value in tags.items():
        if key.startswith(("optimization.", "mlflow.")):
            warn(f"Ignoring user tag with reserved key {key!r}.")
        else:
            accepted_tags[key] = value
    return accepted_tags


def user_metric(
    name: str, value: float, *, unit: str | None = None, description: str | None = None
) -> UserMetric | None:
    """The record of one user metric, or None when it may not be recorded."""
    if name.startswith(("optimization.", "mlflow.")):
        warn(f"Ignoring user metric with reserved name {name!r}.")
        return None
    if not isfinite(value):
        warn(f"Ignoring user metric {name!r} because its value is not finite.")
        return None
    metric_key: str = f"user.{name}"
    tags: dict[str, str] = {}
    if unit is not None:
        tags[f"{metric_key}.unit"] = unit
    if description is not None:
        tags[f"{metric_key}.description"] = description
    return UserMetric(metric_key=metric_key, value=value, tags=tags)
