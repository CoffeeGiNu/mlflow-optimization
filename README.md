# mlflow-optimization

Automatic MLflow logging of mathematical optimization solves.

Supported solvers: [gurobipy, PySCIPOpt, Pyomo, OR-Tools (MathOpt, CP-SAT, Routing)]

## install

```bash
pip install .
```

## usage

```python
import mlflow_optimization

mlflow_optimization.autolog()  # every solve is now logged as an MLflow run

with mlflow_optimization.context(instance="knapsack-50", tags={"team": "ops"}):
    model.optimize()

mlflow_optimization.last_solve().log_metric("gap_to_best_known", 0.02)
```

## License

Either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
