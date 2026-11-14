"""Rules for which explicit artifact requests are honored."""

from pathlib import Path


def artifact_refusal(local_path: str, *, enabled: bool) -> str | None:
    """The reason an explicit artifact request is refused, or None when it is honored."""
    if not enabled:
        return "Artifact logging is disabled."
    try:
        is_file: bool = Path(local_path).is_file()
    except OSError:
        return f"Artifact path cannot be inspected: {local_path}"
    if not is_file:
        return f"Artifact path is not an existing file: {local_path}"
    return None
