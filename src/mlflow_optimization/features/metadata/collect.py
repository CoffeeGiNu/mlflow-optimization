"""Common rules for tracking the API, engine, problem and parameters."""

import re

from .types import (
    CollectedMetadata,
    ParameterOrigin,
    ReportedMetadata,
    SolverParameter,
)

SENSITIVE_PARAMETER_NAME_PARTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "key",
    "license",
    "user",
    "file",
    "dir",
    "directory",
    "path",
    "server",
    "host",
    "url",
    "log",
    "auth",
    "credential",
)


def _is_safe_parameter(parameter: SolverParameter) -> bool:
    """Whether a solver parameter is safe to record."""
    normalized_name: str = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", parameter.name
    )
    normalized_name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", normalized_name
    ).casefold()
    name_parts: list[str] = re.findall(r"[a-z0-9]+", normalized_name)
    for sensitive_part in SENSITIVE_PARAMETER_NAME_PARTS:
        for name_part in name_parts:
            if name_part == sensitive_part:
                return False
            if sensitive_part == "key" and sensitive_part in name_part:
                return False
            if sensitive_part == "dir" and name_part.endswith("dir"):
                return False
            if sensitive_part not in ("key", "log", "dir") and (
                name_part.startswith(sensitive_part)
                or name_part.endswith(sensitive_part)
            ):
                return False
    if isinstance(parameter.value, str):
        value: str = parameter.value
        if re.match(r"^[a-z][a-z0-9+.-]*:", value, re.IGNORECASE):
            return False
        if value.casefold().startswith("www."):
            return False
        if "/" in value or "\\" in value:
            return False
    return True


def _format_value(value: int | float | str | bool) -> str:
    """A string representation of a recorded value."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def collect_metadata(
    api: str, reported: ReportedMetadata | None, *, include_parameters: bool
) -> CollectedMetadata:
    """The recordable experiment conditions of a solve.

    ``reported`` is None when extraction failed.
    """
    tags: dict[str, str] = {"optimization.api": api}
    params: dict[str, str] = {}
    if reported is None:
        tags["optimization.solver"] = "unknown"
        return CollectedMetadata(tags=tags, params=params)

    tags["optimization.solver"] = reported.solver or "unknown"
    if reported.api_version is not None:
        tags["optimization.api_version"] = reported.api_version
    if reported.solver_version is not None:
        tags["optimization.solver_version"] = reported.solver_version
    tags["optimization.wall_time_includes_problem_build"] = _format_value(
        reported.wall_time_includes_problem_build
    )

    for problem_name, problem_value in reported.problem.items():
        params[f"problem.{problem_name}"] = _format_value(problem_value)

    if include_parameters:
        parameter_prefixes: dict[ParameterOrigin, str] = {
            ParameterOrigin.EXPLICIT: "solver_param.",
            ParameterOrigin.DEFAULT: "solver_param_default.",
            ParameterOrigin.EFFECTIVE: "solver_param_effective.",
        }
        for parameter in reported.parameters:
            if _is_safe_parameter(parameter):
                parameter_key: str = parameter_prefixes[parameter.origin] + parameter.name
                params[parameter_key] = _format_value(parameter.value)

    return CollectedMetadata(tags=tags, params=params)
