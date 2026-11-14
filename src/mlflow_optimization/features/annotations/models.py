"""User-provided information attached to solves."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Annotation:
    """The user annotations in effect for a solve."""

    instance: str | None = None
    description: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KpiEntry:
    """One user KPI under final key names."""

    metric_key: str
    value: float
    tags: Mapping[str, str]
