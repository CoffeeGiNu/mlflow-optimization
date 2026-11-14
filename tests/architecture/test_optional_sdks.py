import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[2]
SOLVER_MODULE_NAMES: tuple[str, ...] = (
    "gurobipy",
    "pyscipopt",
    "pyomo",
    "ortools",
)
OPTIONAL_SDK_CASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("gurobipy",), ("gurobipy",)),
    (("pyscipopt",), ("pyscipopt",)),
    (("pyomo",), ("pyomo",)),
    (("ortools",), ("mathopt", "cp_sat", "routing")),
    (
        SOLVER_MODULE_NAMES,
        ("gurobipy", "pyscipopt", "pyomo", "mathopt", "cp_sat", "routing"),
    ),
)
OPTIONAL_SDK_SUBPROCESS_SCRIPT: str = """
import sys

blocked_modules: tuple[str, ...] = tuple(sys.argv[1].split(","))
blocked_apis: tuple[str, ...] = tuple(sys.argv[2].split(","))
tracking_uri: str = sys.argv[3]
blocked_module: str
for blocked_module in blocked_modules:
    sys.modules[blocked_module] = None

from typing import Any

import mlflow

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("optional-solver-sdk-test")
from mlflow_optimization import autolog

autolog()

solved_api: str | None = None
if "ortools" not in blocked_modules:
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        pass
    else:
        model: Any = cp_model.CpModel()
        variable: Any = model.new_bool_var("value")
        model.add(variable == 1)
        solver: Any = cp_model.CpSolver()
        solver.solve(model)
        solved_api = "cp_sat"

if solved_api is None and "pyscipopt" not in blocked_modules:
    try:
        from pyscipopt import Model
    except ImportError:
        pass
    else:
        model: Any = Model()
        model.hideOutput()
        variable: Any = model.addVar(name="value", vtype="B")
        model.setObjective(variable, "minimize")
        model.optimize()
        solved_api = "pyscipopt"

if solved_api is None and "pyomo" not in blocked_modules:
    try:
        import highspy
        import pyomo.environ as pyomo
    except ImportError:
        pass
    else:
        model: Any = pyomo.ConcreteModel()
        model.value = pyomo.Var(domain=pyomo.Binary)
        model.objective = pyomo.Objective(expr=model.value, sense=pyomo.minimize)
        solver: Any = pyomo.SolverFactory("highs")
        if solver.available(exception_flag=False):
            solver.solve(model)
            solved_api = "pyomo"

print(f"SOLVED_API={solved_api or ''}")

api_name: str
for api_name in blocked_apis:
    try:
        autolog(solvers=[api_name])
    except ImportError:
        continue
    raise AssertionError(f"Blocked solver API {api_name!r} did not raise ImportError.")

autolog(disable=True)
"""


@pytest.mark.parametrize(
    ("blocked_modules", "blocked_apis"),
    OPTIONAL_SDK_CASES,
    ids=("gurobipy", "pyscipopt", "pyomo", "ortools", "all-sdks"),
)
def test_optional_sdks_can_be_missing(
    blocked_modules: tuple[str, ...],
    blocked_apis: tuple[str, ...],
    tracking: Any,
) -> None:
    environment: dict[str, str] = os.environ.copy()
    previous_python_path: str = environment.get("PYTHONPATH", "")
    python_path_entries: list[str] = [str(REPOSITORY_ROOT / "src")]
    if previous_python_path:
        python_path_entries.append(previous_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    process: subprocess.CompletedProcess[str] = subprocess.run(
        [
            sys.executable,
            "-c",
            OPTIONAL_SDK_SUBPROCESS_SCRIPT,
            ",".join(blocked_modules),
            ",".join(blocked_apis),
            tracking.client.tracking_uri,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr

    solve_runs: list[Any] = tracking.solve_runs()
    if blocked_modules == SOLVER_MODULE_NAMES:
        assert solve_runs == []
    else:
        solved_api: str = ""
        for output_line in process.stdout.splitlines():
            if output_line.startswith("SOLVED_API="):
                solved_api = output_line.removeprefix("SOLVED_API=")
        assert solved_api
        assert len(solve_runs) == 1
        run_tags: dict[str, str] = solve_runs[0].data.tags
        assert run_tags["optimization.api"] == solved_api
