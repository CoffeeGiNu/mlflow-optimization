# mlflow-optimization

Automatic MLflow logging of mathematical optimization solves.

Supported solvers: [gurobipy, PySCIPOpt, Pyomo, OR-Tools (MathOpt, CP-SAT, Routing)]

## install

Log to a tracking server (`MLFLOW_TRACKING_URI`):

```bash
pip install mlflow-optimization
```

> **NOTE**
>
> `mlflow-optimization` installs `mlflow-skinny`, which lacks MLflow's default local store. To log without a tracking server, install `mlflow` as well:
>
> ```bash
> pip install mlflow-optimization mlflow
> ```

## usage

```python
import mlflow_optimization

mlflow_optimization.autolog()

with mlflow_optimization.context(instance="knapsack-50", tags={"team": "ops"}):
    model.optimize()

run = mlflow_optimization.last_solve()
if run is not None:
    run.log_metric("gap_to_best_known", 0.02)
```

## License

Either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.
