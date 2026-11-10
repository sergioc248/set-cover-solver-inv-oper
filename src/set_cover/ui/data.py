"""Carga y transformación de resultados para la interfaz. Sin Streamlit: todo es probable."""

import csv
import io
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy
from numpy.typing import ArrayLike

from set_cover.checkpoint import Checkpoint, read_checkpoint
from set_cover.compare import MethodResult
from set_cover.ga.engine import GAResult, History
from set_cover.ga.jsonio import (
    RecordError,
    float_field,
    int_field,
    int_list_field,
    read_json_lines,
    str_field,
)
from set_cover.instance import Instance
from set_cover.run import checkpoint_path, is_locked
from set_cover.trajectory import TrajectoryPoint, parse_log, read_trajectory

LOG_NAME = "gurobi.log"
TRAJECTORY_NAME = "trajectory.csv"


@dataclass(frozen=True)
class Paths:
    """Ubicaciones de datos y resultados, relativas a la raíz del proyecto."""

    coverage: Path = Path("data/set_cover_500x500.csv")
    costs: Path = Path("data/antenna_costs_500.csv")
    configs: Path = Path("configs")
    calibration: Path = Path("results/ga/calibration")
    long_runs: Path = Path("results/ga/long_runs")
    saved_solver_runs: Path = Path("results/ga/comparison/gurobi")
    solver_runs: Path = Path("runs")


@dataclass(frozen=True)
class Spread:
    minimum: float
    maximum: float
    mean: float


@dataclass(frozen=True)
class InstanceStats:
    customers: int
    antennas: int
    links: int
    density: float
    total_cost: float
    cost: Spread
    antennas_per_customer: Spread
    customers_per_antenna: Spread


def instance_stats(instance: Instance) -> InstanceStats:
    per_customer = instance.coverage.sum(axis=1)
    per_antenna = instance.coverage.sum(axis=0)
    links = int(per_customer.sum())
    return InstanceStats(
        customers=instance.num_customers,
        antennas=instance.num_antennas,
        links=links,
        density=links / (instance.num_customers * instance.num_antennas),
        total_cost=float(instance.costs.sum()),
        cost=_spread(instance.costs),
        antennas_per_customer=_spread(per_customer),
        customers_per_antenna=_spread(per_antenna),
    )


def _spread(values: ArrayLike) -> Spread:
    array = numpy.asarray(values, dtype=numpy.float64)
    return Spread(float(array.min()), float(array.max()), float(array.mean()))


@dataclass(frozen=True)
class ExactRun:
    """Un directorio con ``checkpoint.json`` y, si existe, la trayectoria del solver."""

    directory: Path
    checkpoint: Checkpoint
    trajectory: list[TrajectoryPoint]
    running: bool

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def label(self) -> str:
        return f"Gurobi · {self.name}"


def discover_exact_runs(roots: Sequence[Path]) -> list[Path]:
    """Directorios con checkpoint directamente bajo cada raíz, ordenados por nombre."""
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(sorted(child for child in root.iterdir() if checkpoint_path(child).is_file()))
    return found


def load_exact_run(directory: Path) -> ExactRun:
    """Lee el checkpoint y la trayectoria: del log de Gurobi o de ``trajectory.csv``.

    El último punto se toma del checkpoint, que guarda la cota redondeada y es la
    fuente de verdad; el log solo muestra la cota sin redondear de cada sesión.
    """
    checkpoint = read_checkpoint(checkpoint_path(directory))
    log = directory / LOG_NAME
    saved = directory / TRAJECTORY_NAME
    if log.is_file():
        trajectory = parse_log(log.read_text(encoding="utf-8", errors="replace"))
    elif saved.is_file():
        trajectory = read_trajectory(saved)
    else:
        trajectory = []
    if checkpoint.best_bound is not None:
        last_seconds = trajectory[-1].seconds if trajectory else 0.0
        incumbent = None if checkpoint.incumbent is None else checkpoint.incumbent.cost
        trajectory = [
            *trajectory,
            TrajectoryPoint(
                max(last_seconds, checkpoint.total_runtime_seconds),
                incumbent,
                checkpoint.best_bound,
            ),
        ]
    return ExactRun(directory, checkpoint, trajectory, is_locked(directory))


def exact_method_result(instance: Instance, run: ExactRun) -> MethodResult | None:
    """Fila de comparación del solver, o ``None`` si aún no tiene incumbente."""
    incumbent = run.checkpoint.incumbent
    if incumbent is None:
        return None
    gap = run.checkpoint.gap()
    return MethodResult(
        method=run.label,
        cost=incumbent.cost,
        antenna_ids=incumbent.antenna_ids,
        covered_customers=covered_customers(instance, incumbent.antenna_ids),
        seconds=run.checkpoint.total_runtime_seconds,
        best_bound=run.checkpoint.best_bound,
        solver_gap=gap,
        proven_optimal=gap == 0.0,
        note=f"{len(run.checkpoint.sessions)} sesión(es); {run.directory}",
    )


def covered_customers(instance: Instance, antenna_ids: list[int]) -> int:
    return instance.num_customers - len(instance.uncovered_customers(antenna_ids))


@dataclass(frozen=True)
class GARunRecord:
    """Campos de un registro de ejecución del GA que muestra la interfaz."""

    seed: int
    config_id: str
    cost: float
    num_antennas: int
    antenna_ids: list[int]
    seconds: float
    evaluations: int
    generations: int
    seconds_to_best: float
    stop_reason: str


def read_ga_records(path: Path) -> list[GARunRecord]:
    records: list[GARunRecord] = []
    for record in read_json_lines(path):
        where = f"{path}"
        records.append(
            GARunRecord(
                seed=int_field(record, "seed", where),
                config_id=str_field(record, "config_id", where),
                cost=float_field(record, "best_cost", where),
                num_antennas=int_field(record, "num_antennas", where),
                antenna_ids=int_list_field(record, "antenna_ids", where),
                seconds=float_field(record, "elapsed_seconds", where),
                evaluations=int_field(record, "evaluations", where),
                generations=int_field(record, "generations", where),
                seconds_to_best=float_field(record, "seconds_to_best", where),
                stop_reason=str_field(record, "stop_reason", where),
            )
        )
    return records


def ga_record_result(
    instance: Instance, records: Sequence[GARunRecord], method: str
) -> MethodResult:
    """Mejor ejecución de un grupo de registros, como fila de comparación."""
    if not records:
        raise RecordError(f"{method}: no runs")
    best = min(records, key=lambda record: (record.cost, record.seed))
    _require_feasible(instance, best.antenna_ids, method)
    return MethodResult(
        method=method,
        cost=best.cost,
        antenna_ids=best.antenna_ids,
        covered_customers=instance.num_customers,
        seconds=best.seconds,
        best_bound=None,
        solver_gap=None,
        proven_optimal=False,
        note=f"mejor de {len(records)} ejecuciones (semilla {best.seed}); tiempo de esa ejecución",
    )


def ga_session_result(instance: Instance, result: GAResult, method: str) -> MethodResult:
    ids = list(result.evaluation.antenna_ids)
    _require_feasible(instance, ids, method)
    return MethodResult(
        method=method,
        cost=result.evaluation.cost,
        antenna_ids=ids,
        covered_customers=instance.num_customers,
        seconds=result.elapsed_seconds,
        best_bound=None,
        solver_gap=None,
        proven_optimal=False,
        note=f"semilla {result.seed}, configuración {result.config.config_id()}",
    )


def _require_feasible(instance: Instance, antenna_ids: list[int], method: str) -> None:
    uncovered = instance.uncovered_customers(antenna_ids)
    if uncovered:
        raise RecordError(f"{method}: solution leaves customers uncovered: {uncovered}")


@dataclass(frozen=True)
class Curve:
    """Mejor costo encontrado hasta cada instante de una ejecución."""

    label: str
    seconds: list[float]
    best: list[float]


def read_convergence(path: Path) -> list[Curve]:
    """Curvas por semilla desde ``final_convergence.csv``."""
    seconds: dict[str, list[float]] = {}
    best: dict[str, list[float]] = {}
    for row in read_csv_rows(path):
        seed = row["seed"]
        seconds.setdefault(seed, []).append(float(row["elapsed_seconds"]))
        best.setdefault(seed, []).append(float(row["best_so_far"]))
    return [Curve(f"semilla {seed}", seconds[seed], best[seed]) for seed in seconds]


def history_curve(history: History, label: str) -> Curve:
    return Curve(label, list(history.elapsed_seconds), list(history.best_so_far))


def median_curve(curves: Sequence[Curve], grid: Sequence[float]) -> list[float | None]:
    """Mediana entre curvas del mejor costo en cada instante de ``grid``.

    Cada curva es escalonada; antes de su primer punto no tiene valor. Una curva que
    terminó conserva su último valor. La mediana existe solo cuando todas tienen valor.
    """
    medians: list[float | None] = []
    for instant in grid:
        values = [_value_at(curve, instant) for curve in curves]
        present = [value for value in values if value is not None]
        medians.append(
            statistics.median(present) if present and len(present) == len(values) else None
        )
    return medians


def _value_at(curve: Curve, instant: float) -> float | None:
    position = int(numpy.searchsorted(numpy.asarray(curve.seconds), instant, side="right"))
    return None if position == 0 else curve.best[position - 1]


def median_series(
    curves: Sequence[Curve], points: int = 200
) -> tuple[list[float], list[float | None]]:
    """Mediana entre curvas sobre una rejilla logarítmica que cubre todas las ejecuciones."""
    starts = [curve.seconds[0] for curve in curves if curve.seconds]
    stops = [curve.seconds[-1] for curve in curves if curve.seconds]
    if not starts:
        return [], []
    start = max(min(starts), 1e-3)
    grid = log_grid(start, max(*stops, start * 2), points)
    return grid, median_curve(curves, grid)


def log_grid(start: float, stop: float, points: int) -> list[float]:
    if start <= 0.0 or stop <= start:
        raise ValueError("log_grid needs 0 < start < stop")
    return [float(value) for value in numpy.geomspace(start, stop, points)]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def number(row: Mapping[str, str], key: str) -> float | None:
    """Valor numérico de una celda CSV; vacío o no numérico da ``None``."""
    text = row.get(key, "")
    try:
        return float(text)
    except ValueError:
        return None


def jaccard(first: Sequence[int], second: Sequence[int]) -> float:
    union = set(first) | set(second)
    return 1.0 if not union else len(set(first) & set(second)) / len(union)


def duration(seconds: float) -> str:
    """Duración legible: ``4,0 s``, ``1 min 52 s`` o ``92 h 43 min``."""
    if seconds < 60:
        return f"{seconds:.1f} s".replace(".", ",")
    whole = round(seconds)
    if whole < 3600:
        return f"{whole // 60} min {whole % 60:02d} s"
    return f"{whole // 3600} h {whole % 3600 // 60:02d} min"
