"""Una sesión del solver Gurobi sobre un directorio de ejecución.

La sesión arranca en caliente desde la incumbente del checkpoint, escribe a
disco cada incumbente mejorada y la cota periódicamente, y se detiene por límite
de tiempo, brecha objetivo, Ctrl+C, SIGTERM o SIGHUP. Siempre se puede reanudar
después de detenerla.
"""

import signal
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import FrameType
from typing import Any

import gurobipy
from gurobipy import GRB

from set_cover.checkpoint import Session
from set_cover.model import add_set_cover
from set_cover.run import RunDirectory, utc_now

GAP_REACHED = "gap_reached"
OPTIMAL = "optimal"
TIME_LIMIT = "time_limit"
INTERRUPTED = "interrupted"

_STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)

type _SignalHandler = Callable[[int, FrameType | None], Any] | int | signal.Handlers | None


class ParameterError(ValueError):
    pass


@dataclass(frozen=True)
class SolveSettings:
    target_gap: float
    time_limit: float | None
    threads: int
    seed: int
    node_file_start_gb: float
    checkpoint_interval: float
    gurobi_params: dict[str, str]

    def describe(self) -> dict[str, str]:
        described = {
            "target_gap": str(self.target_gap),
            "time_limit": "none" if self.time_limit is None else str(self.time_limit),
            "threads": str(self.threads),
            "seed": str(self.seed),
            "node_file_start_gb": str(self.node_file_start_gb),
        }
        described.update({f"gurobi.{name}": value for name, value in self.gurobi_params.items()})
        return described


def solve(run: RunDirectory, settings: SolveSettings) -> Session | None:
    """Ejecuta una sesión; retorna ``None`` si el checkpoint ya cumple el objetivo."""
    gap = run.checkpoint.gap()
    if gap is not None and gap <= settings.target_gap:
        return None

    instance = run.instance
    env = gurobipy.Env(empty=True)
    env.setParam("LogFile", str(run.log_path))
    env.start()
    model = gurobipy.Model("weighted_set_cover", env=env)
    previous_handlers: dict[signal.Signals, _SignalHandler] = {}
    try:
        antennas = add_set_cover(model, instance, GRB.BINARY)
        _configure(model, run, settings)

        incumbent = run.checkpoint.incumbent
        if incumbent is not None:
            selected = set(incumbent.antenna_ids)
            for j, variable in antennas.items():
                variable.Start = 1.0 if j in selected else 0.0

        session = run.begin_session(settings.target_gap, settings.describe())
        monitor = _Monitor(run, session, antennas, settings)
        for stop_signal in _STOP_SIGNALS:
            previous_handlers[stop_signal] = signal.signal(stop_signal, monitor.request_stop)

        model.optimize(monitor)

        if model.SolCount > 0:
            run.offer_incumbent(_selected_antennas({j: v.X for j, v in antennas.items()}))
        session.best_bound = run.offer_bound(model.ObjBound)
        session.runtime_seconds = model.Runtime
        session.nodes = model.NodeCount
        session.status = _session_status(model.Status, monitor.gap_reached, run)
        session.ended_at = utc_now()
        run.save()
        return session
    finally:
        for stop_signal, handler in previous_handlers.items():
            signal.signal(stop_signal, handler)
        model.dispose()
        env.dispose()


def _configure(model: gurobipy.Model, run: RunDirectory, settings: SolveSettings) -> None:
    model.Params.MIPGap = settings.target_gap
    model.Params.Threads = settings.threads
    model.Params.Seed = settings.seed
    if settings.time_limit is not None:
        model.Params.TimeLimit = settings.time_limit
    run.node_file_dir.mkdir(exist_ok=True)
    model.Params.NodefileStart = settings.node_file_start_gb
    model.Params.NodefileDir = str(run.node_file_dir)
    for name, value in settings.gurobi_params.items():
        _set_param(model, name, value)


def _set_param(model: gurobipy.Model, name: str, value: str) -> None:
    """Asigna un parámetro desde texto, convertido al tipo que Gurobi declara para él."""
    info = model.getParamInfo(name)
    if info is None:
        raise ParameterError(f"unknown Gurobi parameter {name!r}")
    declared = info[1]
    try:
        if declared is int:
            model.setParam(name, int(value))
        elif declared is float:
            model.setParam(name, float(value))
        else:
            model.setParam(name, value)
    except ValueError:
        raise ParameterError(f"Gurobi parameter {name} expects {declared.__name__}") from None


def _selected_antennas(values: Mapping[int, float]) -> list[int]:
    """Ids de antena (base 1) cuya variable binaria está activa."""
    return sorted(j for j, value in values.items() if value > 0.5)


def _session_status(gurobi_status: int, gap_reached: bool, run: RunDirectory) -> str:
    gap = run.checkpoint.gap()
    if gap == 0.0:
        return OPTIMAL
    if gap_reached or gurobi_status == GRB.OPTIMAL:
        return GAP_REACHED
    if gurobi_status == GRB.TIME_LIMIT:
        return TIME_LIMIT
    if gurobi_status == GRB.INTERRUPTED:
        return INTERRUPTED
    return f"gurobi_status_{gurobi_status}"


class _Monitor:
    """Callback de Gurobi que guarda el progreso y aplica la brecha entre sesiones."""

    def __init__(
        self,
        run: RunDirectory,
        session: Session,
        antennas: gurobipy.tupledict[int, gurobipy.Var],
        settings: SolveSettings,
    ) -> None:
        self._run = run
        self._session = session
        self._antennas = antennas
        self._target_gap = settings.target_gap
        self._interval = settings.checkpoint_interval
        self._last_checkpoint = time.monotonic()
        self._stop_requested = False
        self.gap_reached = False

    def request_stop(self, signal_number: int, _frame: FrameType | None) -> None:
        print(
            f"\nReceived {signal.Signals(signal_number).name}; stopping after checkpoint.",
            file=sys.stderr,
            flush=True,
        )
        self._stop_requested = True

    def __call__(self, model: gurobipy.Model, where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            self._run.offer_incumbent(_selected_antennas(model.cbGetSolution(self._antennas)))
        elif where == GRB.Callback.MIP:
            if self._stop_requested:
                model.terminate()
                return
            self._track_bound(model)

    def _track_bound(self, model: gurobipy.Model) -> None:
        bound = self._run.offer_bound(float(model.cbGet(GRB.Callback.MIP_OBJBND)))
        if bound is not None:
            self._session.best_bound = bound
        now = time.monotonic()
        if now - self._last_checkpoint >= self._interval:
            self._session.runtime_seconds = float(model.cbGet(GRB.Callback.RUNTIME))
            self._session.nodes = float(model.cbGet(GRB.Callback.MIP_NODCNT))
            self._run.save()
            self._last_checkpoint = now
        # Gurobi solo conoce la cota de esta sesión; sesiones previas pueden cerrar la brecha antes.
        gap = self._run.checkpoint.gap()
        if gap is not None and gap <= self._target_gap:
            self.gap_reached = True
            model.terminate()
