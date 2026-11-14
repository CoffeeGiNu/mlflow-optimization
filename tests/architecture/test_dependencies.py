import ast
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
SOURCE_ROOT: Path = REPOSITORY_ROOT / "src" / "mlflow_optimization"
PACKAGE_NAME: str = "mlflow_optimization"
SOLVER_MODULES: frozenset[str] = frozenset(
    {"gurobipy", "pyscipopt", "pyomo", "ortools"}
)
FEATURE_IMPORT_RULES: dict[str, dict[str, frozenset[str] | None]] = {
    "annotations": {},
    "artifacts": {},
    "autolog": {
        "metadata": frozenset({"types"}),
        "metrics": frozenset({"types"}),
        "outcomes": frozenset({"types"}),
    },
    "metadata": {},
    "metrics": {},
    "outcomes": {},
    "runs": {
        "annotations": None,
        "artifacts": None,
        "autolog": frozenset({"types", "ports"}),
        "metadata": None,
        "metrics": None,
        "outcomes": None,
    },
}
INTEGRATION_SDK_PREFIXES: dict[str, frozenset[str]] = {
    "gurobipy": frozenset({"gurobipy"}),
    "mathopt": frozenset({"ortools"}),
    "pyomo": frozenset({"pyomo"}),
    "pyscipopt": frozenset({"pyscipopt"}),
    "cp_sat": frozenset({"ortools"}),
    "routing": frozenset({"ortools"}),
}
IMPORT_CONTAINERS: frozenset[str] = frozenset(
    {
        PACKAGE_NAME,
        f"{PACKAGE_NAME}.features",
        f"{PACKAGE_NAME}.integrations",
        f"{PACKAGE_NAME}.common",
        f"{PACKAGE_NAME}.features.annotations",
        f"{PACKAGE_NAME}.features.artifacts",
        f"{PACKAGE_NAME}.features.autolog",
        f"{PACKAGE_NAME}.features.metadata",
        f"{PACKAGE_NAME}.features.metrics",
        f"{PACKAGE_NAME}.features.outcomes",
        f"{PACKAGE_NAME}.features.runs",
    }
)
ORTOOLS_SUBMODULES: frozenset[str] = frozenset(
    {"math_opt", "sat", "constraint_solver"}
)


def _module_name(source_file: Path) -> str:
    relative_parts: tuple[str, ...] = source_file.relative_to(SOURCE_ROOT).with_suffix("").parts
    if relative_parts[-1] == "__init__":
        relative_parts = relative_parts[:-1]
    return ".".join((PACKAGE_NAME, *relative_parts))


def _package_name(module_name: str, source_file: Path) -> str:
    if source_file.name == "__init__.py":
        return module_name
    return module_name.rpartition(".")[0]


def _relative_import_base(package_name: str, import_node: ast.ImportFrom) -> str:
    package_parts: list[str] = package_name.split(".")
    parent_parts: list[str] = package_parts[: len(package_parts) - import_node.level + 1]
    if import_node.module is not None:
        parent_parts.extend(import_node.module.split("."))
    return ".".join(parent_parts)


def _imported_modules(
    package_name: str, import_node: ast.Import | ast.ImportFrom
) -> set[str]:
    imported_modules: set[str] = set()
    if isinstance(import_node, ast.Import):
        alias: ast.alias
        for alias in import_node.names:
            imported_modules.add(alias.name)
        return imported_modules

    base_module: str = import_node.module or ""
    if import_node.level > 0:
        base_module = _relative_import_base(package_name, import_node)
    if base_module in IMPORT_CONTAINERS:
        alias: ast.alias
        for alias in import_node.names:
            imported_modules.add(f"{base_module}.{alias.name}")
        return imported_modules
    if base_module == "ortools" and all(
        alias.name in ORTOOLS_SUBMODULES for alias in import_node.names
    ):
        alias: ast.alias
        for alias in import_node.names:
            imported_modules.add(f"{base_module}.{alias.name}")
        return imported_modules
    imported_modules.add(base_module)
    return imported_modules


def _is_standard_library_module(module_name: str) -> bool:
    module_root: str = module_name.partition(".")[0]
    return module_root in sys.stdlib_module_names or module_root in sys.builtin_module_names


def _is_package_module(module_name: str, package_name: str) -> bool:
    return module_name == package_name or module_name.startswith(f"{package_name}.")


def _feature_name(module_name: str) -> str | None:
    module_parts: list[str] = module_name.split(".")
    if len(module_parts) < 3 or module_parts[:2] != [PACKAGE_NAME, "features"]:
        return None
    return module_parts[2]


def _feature_module_name(module_name: str, feature_name: str) -> str:
    feature_prefix: str = f"{PACKAGE_NAME}.features.{feature_name}"
    return module_name.removeprefix(f"{feature_prefix}.") if module_name != feature_prefix else ""


def _integration_name(module_name: str) -> str | None:
    module_parts: list[str] = module_name.split(".")
    if len(module_parts) < 3 or module_parts[:2] != [PACKAGE_NAME, "integrations"]:
        return None
    return module_parts[2]


def _find_feature_cycle(feature_dependencies: dict[str, set[str]]) -> list[str] | None:
    visited_features: set[str] = set()
    active_features: list[str] = []

    def visit_feature(feature_name: str) -> list[str] | None:
        if feature_name in active_features:
            cycle_start: int = active_features.index(feature_name)
            return [*active_features[cycle_start:], feature_name]
        if feature_name in visited_features:
            return None
        active_features.append(feature_name)
        dependency: str
        for dependency in sorted(feature_dependencies[feature_name]):
            dependency_cycle: list[str] | None = visit_feature(dependency)
            if dependency_cycle is not None:
                return dependency_cycle
        active_features.pop()
        visited_features.add(feature_name)
        return None

    feature_name: str
    for feature_name in sorted(feature_dependencies):
        cycle: list[str] | None = visit_feature(feature_name)
        if cycle is not None:
            return cycle
    return None


def _check_feature_import(
    source_file: Path,
    source_feature: str,
    target_module: str,
    feature_dependencies: dict[str, set[str]],
) -> str | None:
    module_root: str = target_module.partition(".")[0]
    relative_path: str = str(source_file.relative_to(REPOSITORY_ROOT))
    if module_root in SOLVER_MODULES:
        return f"{relative_path}: feature '{source_feature}' imports solver SDK '{module_root}'"
    if _is_package_module(target_module, f"{PACKAGE_NAME}.integrations"):
        return f"{relative_path}: feature '{source_feature}' imports an integration"
    if _is_package_module(target_module, f"{PACKAGE_NAME}.common"):
        return None
    if _is_standard_library_module(target_module):
        return None

    target_feature: str | None = _feature_name(target_module)
    if target_feature == source_feature:
        return None
    if target_feature is None or target_feature not in FEATURE_IMPORT_RULES:
        return f"{relative_path}: feature '{source_feature}' imports unsupported module '{target_module}'"
    feature_dependencies[source_feature].add(target_feature)
    allowed_modules: frozenset[str] | None = FEATURE_IMPORT_RULES[source_feature].get(
        target_feature
    )
    if allowed_modules is None and target_feature in FEATURE_IMPORT_RULES[source_feature]:
        return None
    target_feature_module: str = _feature_module_name(target_module, target_feature)
    if allowed_modules is not None and target_feature_module in allowed_modules:
        return None
    return (
        f"{relative_path}: feature '{source_feature}' imports disallowed module "
        f"'{target_module}' from feature '{target_feature}'"
    )


def _check_integration_import(
    source_file: Path, source_integration: str, target_module: str
) -> str | None:
    relative_path: str = str(source_file.relative_to(REPOSITORY_ROOT))
    target_integration: str | None = _integration_name(target_module)
    module_root: str = target_module.partition(".")[0]
    if target_integration is not None:
        if target_integration == source_integration:
            return None
        return f"{relative_path}: integration '{source_integration}' imports integration '{target_integration}'"
    if _is_standard_library_module(target_module):
        return None
    if _is_package_module(target_module, f"{PACKAGE_NAME}.common"):
        return None
    target_feature: str | None = _feature_name(target_module)
    if source_integration == "mlflow":
        if module_root in SOLVER_MODULES:
            return f"{relative_path}: MLflow integration imports solver SDK '{module_root}'"
        if target_feature in {"runs", "artifacts"}:
            return None
        if target_feature is not None:
            return f"{relative_path}: MLflow integration imports disallowed feature '{target_feature}'"
        if module_root == "mlflow":
            return None
        return f"{relative_path}: MLflow integration imports unsupported module '{target_module}'"
    if source_integration not in INTEGRATION_SDK_PREFIXES:
        return f"{relative_path}: unknown integration '{source_integration}'"
    if target_feature is not None:
        feature_module: str = _feature_module_name(target_module, target_feature)
        if (
            target_feature in FEATURE_IMPORT_RULES
            and feature_module in {"types", "ports"}
        ):
            return None
        return f"{relative_path}: integration '{source_integration}' imports feature module '{target_module}'"
    allowed_sdk_prefix: str
    for allowed_sdk_prefix in INTEGRATION_SDK_PREFIXES[source_integration]:
        if _is_package_module(target_module, allowed_sdk_prefix):
            return None
    return f"{relative_path}: integration '{source_integration}' imports unsupported module '{target_module}'"


def _architecture_violations() -> list[str]:
    violations: list[str] = []
    feature_dependencies: dict[str, set[str]] = {}
    feature_name: str
    for feature_name in FEATURE_IMPORT_RULES:
        feature_dependencies[feature_name] = set()
    source_file: Path
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        module_name: str = _module_name(source_file)
        package_name: str = _package_name(module_name, source_file)
        source_feature: str | None = _feature_name(module_name)
        source_integration: str | None = _integration_name(module_name)
        if (
            source_integration is not None
            and source_integration != "mlflow"
            and source_integration not in INTEGRATION_SDK_PREFIXES
        ):
            violations.append(
                f"{source_file.relative_to(REPOSITORY_ROOT)}: unknown integration "
                f"'{source_integration}'"
            )
        syntax_tree: ast.Module = ast.parse(
            source_file.read_text(encoding="utf-8"), filename=str(source_file)
        )
        import_node: ast.AST
        for import_node in ast.walk(syntax_tree):
            if not isinstance(import_node, (ast.Import, ast.ImportFrom)):
                continue
            target_module: str
            for target_module in sorted(
                _imported_modules(package_name, import_node)
            ):
                if source_feature is not None:
                    feature_violation: str | None = _check_feature_import(
                        source_file,
                        source_feature,
                        target_module,
                        feature_dependencies,
                    )
                    if feature_violation is not None:
                        violations.append(feature_violation)
                elif _is_package_module(module_name, f"{PACKAGE_NAME}.common"):
                    if not _is_standard_library_module(target_module):
                        violations.append(
                            f"{source_file.relative_to(REPOSITORY_ROOT)}: common module imports "
                            f"non-standard-library module '{target_module}'"
                        )
                elif source_integration is not None:
                    integration_violation: str | None = _check_integration_import(
                        source_file, source_integration, target_module
                    )
                    if integration_violation is not None:
                        violations.append(integration_violation)

    feature_cycle: list[str] | None = _find_feature_cycle(feature_dependencies)
    if feature_cycle is not None:
        violations.append(f"feature dependency cycle: {' -> '.join(feature_cycle)}")
    return violations


def test_source_imports_follow_the_feature_and_integration_boundaries() -> None:
    violations: list[str] = _architecture_violations()
    assert violations == [], "\n".join(violations)


def test_importing_the_package_does_not_load_solver_sdks() -> None:
    environment: dict[str, str] = os.environ.copy()
    previous_python_path: str = environment.get("PYTHONPATH", "")
    python_path_entries: list[str] = [str(REPOSITORY_ROOT / "src")]
    if previous_python_path:
        python_path_entries.append(previous_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    script: str = (
        "import sys\n"
        "import mlflow_optimization\n"
        "solver_modules: tuple[str, ...] = ('gurobipy', 'pyscipopt', 'pyomo', 'ortools')\n"
        "loaded_modules: list[str] = []\n"
        "module_name: str\n"
        "solver_module: str\n"
        "for module_name in sys.modules:\n"
        "    for solver_module in solver_modules:\n"
        "        if module_name == solver_module or module_name.startswith(solver_module + '.'):\n"
        "            loaded_modules.append(module_name)\n"
        "loaded_modules = sorted(set(loaded_modules))\n"
        "if loaded_modules:\n"
        "    raise SystemExit(f'loaded solver SDK modules: {loaded_modules}')\n"
    )
    process: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
