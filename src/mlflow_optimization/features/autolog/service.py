"""Activation of integrations and arbitration of observed solve calls."""

from collections.abc import Callable, Iterator, Sequence
from contextvars import ContextVar, Token
from dataclasses import replace
from itertools import count
from threading import Lock
from time import perf_counter, time
from typing import cast
from uuid import uuid4

from mlflow_optimization.common.diagnostics import warn
from mlflow_optimization.features.progress.types import ProgressHook, SampleSink

from .ports import SolverIntegration, SolveSink
from .types import CompletedSolve, SolveObservation

_solve_indices: Iterator[int] = count(1)
_solve_index_lock: Lock = Lock()


class _SolveScope:
    """The lifetime of an observed solve call."""

    def __init__(self) -> None:
        self.active: bool = True


_observing: ContextVar[_SolveScope | None] = ContextVar(
    "autolog_observing", default=None
)


class AutologManager:
    """The observer that turns outermost solve calls into completed solves."""

    def __init__(self, sink: SolveSink) -> None:
        self._sink: SolveSink = sink
        self._integrations: list[SolverIntegration] = []
        self._enabled: bool = False

    def enable(self, integrations: Sequence[SolverIntegration]) -> None:
        """Observe exactly the given integrations."""
        for integration in self._integrations[:]:
            if not any(integration is candidate for candidate in integrations):
                integration.uninstall()
                self._integrations.remove(integration)
        for integration in integrations:
            if not any(integration is installed for installed in self._integrations):
                integration.install(self)
                self._integrations.append(integration)
        self._enabled = True

    def disable(self) -> None:
        """Stop observing and uninstall every integration."""
        self._enabled = False
        for integration in self._integrations[:]:
            integration.uninstall()
            self._integrations.remove(integration)

    def observe[T](
        self,
        api: str,
        call: Callable[[ProgressHook | None], T],
        extract: Callable[[T | None, BaseException | None], SolveObservation],
    ) -> T:
        """The result of ``call(hook)``, observed as one solve unless nested or disabled."""
        parent: _SolveScope | None = _observing.get()
        if not self._enabled or (parent is not None and parent.active):
            return call(None)
        scope: _SolveScope = _SolveScope()
        token: Token[_SolveScope | None] = _observing.set(scope)
        try:
            failures: list[str] = []

            def guard(step: str, operation: Callable[[], object]) -> object:
                """The result of a guarded observation operation."""
                try:
                    return operation()
                # NOTE: A defect in logging must never change the result of the user's solve.
                except Exception as exception:  # noqa: BLE001
                    failures.append(
                        f"Solve {step} failed for {api}: "
                        f"{type(exception).__name__}: {exception}"
                    )
                    return None

            context: object = guard("begin", self._sink.begin)
            with _solve_index_lock:
                solve_index: int = next(_solve_indices)
            solve_id: str = uuid4().hex
            start_time_ms: int = int(time() * 1000)
            sample_sink: SampleSink | None = cast(
                SampleSink | None,
                guard(
                    "start",
                    lambda: self._sink.start(
                        context, api, solve_id, solve_index, start_time_ms
                    ),
                ),
            )
            progress_hook: ProgressHook | None = None
            if sample_sink is not None:
                sink: SampleSink = sample_sink

                def progress(incumbent: float | None, bound: float | None) -> None:
                    """A progress sample from the solve."""
                    sink(start, incumbent, bound)

                progress_hook = progress
            result: T | None = None

            def finish(solve: CompletedSolve) -> None:
                """Extract and record a completed solve."""
                observation: SolveObservation | None = None
                try:
                    observation = extract(result, solve.error)
                # NOTE: A defect in extraction must never change the result of the user's solve.
                except Exception as exception:  # noqa: BLE001
                    failures.append(
                        f"Solve extraction failed for {api}: "
                        f"{type(exception).__name__}: {exception}"
                    )
                self._sink.record(replace(solve, observation=observation), context)

            start: float = perf_counter()
            try:
                result = call(progress_hook)
            except BaseException as error:
                end: float = perf_counter()
                scope.active = False
                end_time_ms: int = int(time() * 1000)
                solve: CompletedSolve = CompletedSolve(
                    api,
                    solve_id,
                    solve_index,
                    start_time_ms,
                    end_time_ms,
                    end - start,
                    None,
                    error,
                )
                guard("recording", lambda: finish(solve))
                for failure in failures:
                    warn(failure)
                raise
            end = perf_counter()
            scope.active = False
            end_time_ms = int(time() * 1000)
            solve = CompletedSolve(
                api,
                solve_id,
                solve_index,
                start_time_ms,
                end_time_ms,
                end - start,
                None,
                None,
            )
            guard("recording", lambda: finish(solve))
            for failure in failures:
                warn(failure)
            return cast(T, result)
        finally:
            scope.active = False
            _observing.reset(token)
