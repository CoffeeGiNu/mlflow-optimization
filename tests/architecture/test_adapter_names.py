"""Concrete adapter names are resolved lazily and exported by SDK packages."""

from importlib import import_module
from types import ModuleType

import pytest

from mlflow_optimization import _bootstrap


@pytest.mark.parametrize(
    ("api", "class_name"),
    [
        ("gurobipy", "GurobiAdapter"),
        ("pyscipopt", "PyScipOptAdapter"),
        ("pyomo", "PyomoAdapter"),
        ("mathopt", "MathOptAdapter"),
        ("cp_sat", "CpSatAdapter"),
        ("routing", "RoutingAdapter"),
    ],
)
def test_adapter_resolution_and_exports(api: str, class_name: str) -> None:
    module_name: str
    registered_class: str
    module_name, registered_class = _bootstrap.INTEGRATION_MODULES[api]
    assert registered_class == class_name
    module: ModuleType = import_module(module_name)
    adapter_class: type = getattr(module, class_name)
    assert type(_bootstrap._resolve_integrations([api])[0]) is adapter_class
    if api != "mathopt":
        package: ModuleType = import_module(f"mlflow_optimization.integrations.{api}")
        assert getattr(package, class_name) is adapter_class
        assert class_name in package.__all__
