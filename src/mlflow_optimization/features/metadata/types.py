"""Experiment conditions of a solve."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class ParameterOrigin(StrEnum):
    """Where a solver parameter value comes from."""

    EXPLICIT = "explicit"
    DEFAULT = "default"
    EFFECTIVE = "effective"


@dataclass(frozen=True)
class SolverParameter:
    """One solver parameter value and its origin."""

    name: str
    value: int | float | str | bool
    origin: ParameterOrigin = ParameterOrigin.EXPLICIT


@dataclass(frozen=True)
class ReportedMetadata:
    """The experiment conditions as an adapter read them from the solver."""

    solver: str | None = None
    api_version: str | None = None
    solver_version: str | None = None
    problem: Mapping[str, float | str | bool] = field(default_factory=dict)
    parameters: Sequence[SolverParameter] = ()
    wall_time_includes_problem_build: bool = False


@dataclass(frozen=True)
class CollectedMetadata:
    """The tags and params recorded for the experiment conditions, under final key names."""

    tags: Mapping[str, str]
    params: Mapping[str, str]
