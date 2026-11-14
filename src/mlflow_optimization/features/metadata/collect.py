"""Common rules for tracking the API, engine, problem and parameters."""

from .types import (
    CollectedMetadata,
    ParameterOrigin,
    ReportedMetadata,
)


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
    collected_parameters: dict[str, str] = {}
    if reported is None:
        tags["optimization.solver"] = "unknown"
        return CollectedMetadata(tags=tags, params=collected_parameters)

    tags["optimization.solver"] = reported.solver or "unknown"
    if reported.api_version is not None:
        tags["optimization.api_version"] = reported.api_version
    if reported.solver_version is not None:
        tags["optimization.solver_version"] = reported.solver_version
    tags["optimization.wall_time_includes_problem_build"] = _format_value(
        reported.wall_time_includes_problem_build
    )

    for problem_name, problem_value in reported.problem.items():
        collected_parameters[f"problem.{problem_name}"] = _format_value(problem_value)

    if include_parameters:
        parameter_prefixes: dict[ParameterOrigin, str] = {
            ParameterOrigin.EXPLICIT: "solver_param.",
            ParameterOrigin.DEFAULT: "solver_param_default.",
            ParameterOrigin.EFFECTIVE: "solver_param_effective.",
        }
        for parameter in reported.parameters:
            parameter_key: str = (
                parameter_prefixes[parameter.origin] + parameter.name
            )
            collected_parameters[parameter_key] = _format_value(parameter.value)

    return CollectedMetadata(tags=tags, params=collected_parameters)
