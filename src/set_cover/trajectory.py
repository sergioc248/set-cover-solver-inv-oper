"""Trayectoria del solver exacto en el tiempo, extraída de ``gurobi.log``.

El checkpoint solo guarda el estado final. El log tiene una línea de progreso por
intervalo con incumbente, cota y tiempo de la sesión; aquí se acumulan las sesiones
en un solo eje de tiempo. La cota de una sesión reanudada reinicia desde la
relajación LP, pero la cota probada en una sesión previa sigue siendo válida, así
que la trayectoria guarda el máximo acumulado de la cota y el mínimo de la incumbente.
"""

import csv
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

from set_cover.ga.jsonio import write_text_atomic

_SESSION_HEADER = "Gurobi Optimizer version"
_EXPLORED = re.compile(r"^Explored .* in ([0-9.]+) seconds")
_NODE_TIME = re.compile(r"^(\d+)s$")
_CSV_HEADER = ("seconds", "incumbent", "bound")


class TrajectoryError(ValueError):
    pass


@dataclass(frozen=True)
class TrajectoryPoint:
    """Estado acumulado: segundos totales de solver, mejor incumbente y mejor cota."""

    seconds: float
    incumbent: float | None
    bound: float


def parse_log(text: str) -> list[TrajectoryPoint]:
    """Puntos de progreso de todas las sesiones de un log, en orden y acumulados."""
    raw: list[tuple[float, float | None, float]] = []
    offset = 0.0
    session_end = 0.0
    for line in text.splitlines():
        if line.startswith(_SESSION_HEADER):
            offset += session_end
            session_end = 0.0
            continue
        explored = _EXPLORED.match(line)
        if explored is not None:
            session_end = float(explored.group(1))
            continue
        progress = _progress(line)
        if progress is not None:
            seconds, incumbent, bound = progress
            session_end = max(session_end, seconds)
            raw.append((offset + seconds, incumbent, bound))
    return _accumulate(raw)


def _progress(line: str) -> tuple[float, float | None, float] | None:
    """Tiempo, incumbente y cota de una línea del log de nodos, o ``None`` si no es una."""
    tokens = line.split()
    if len(tokens) < 6:
        return None
    time_match = _NODE_TIME.match(tokens[-1])
    gap = tokens[-3]
    if time_match is None or not (gap == "-" or gap.endswith("%")):
        return None
    bound = _number(tokens[-4])
    incumbent_token = tokens[-5]
    incumbent = None if incumbent_token == "-" else _number(incumbent_token)
    if bound is None or (incumbent_token != "-" and incumbent is None):
        return None
    return float(time_match.group(1)), incumbent, bound


def _number(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _accumulate(raw: list[tuple[float, float | None, float]]) -> list[TrajectoryPoint]:
    """Mínimo acumulado de la incumbente y máximo de la cota; descarta puntos sin cambio."""
    points: list[TrajectoryPoint] = []
    best_incumbent: float | None = None
    best_bound = -math.inf
    for seconds, incumbent, bound in raw:
        if incumbent is not None and (best_incumbent is None or incumbent < best_incumbent):
            best_incumbent = incumbent
        best_bound = max(best_bound, bound)
        point = TrajectoryPoint(seconds, best_incumbent, best_bound)
        if points and _same_state(points[-1], point):
            continue
        points.append(point)
    if raw and points[-1].seconds != raw[-1][0]:
        # Conserva el último instante para que la curva llegue hasta el final de la ejecución.
        last = points[-1]
        points.append(TrajectoryPoint(raw[-1][0], last.incumbent, last.bound))
    return points


def _same_state(previous: TrajectoryPoint, current: TrajectoryPoint) -> bool:
    """La cota se compara con resolución de una unidad: los costos de la instancia son enteros."""
    return previous.incumbent == current.incumbent and current.bound - previous.bound < 1.0


def trajectory_csv(points: list[TrajectoryPoint]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_HEADER)
    for point in points:
        incumbent = "" if point.incumbent is None else repr(point.incumbent)
        writer.writerow([repr(point.seconds), incumbent, repr(point.bound)])
    return buffer.getvalue()


def write_trajectory(path: Path, points: list[TrajectoryPoint]) -> None:
    write_text_atomic(path, trajectory_csv(points))


def read_trajectory(path: Path) -> list[TrajectoryPoint]:
    reader = csv.reader(io.StringIO(path.read_text(encoding="utf-8")))
    header = next(reader, None)
    if header is None or tuple(header) != _CSV_HEADER:
        raise TrajectoryError(f"{path}: header MUST be {','.join(_CSV_HEADER)}")
    points: list[TrajectoryPoint] = []
    for line_number, row in enumerate(reader, start=2):
        if len(row) != len(_CSV_HEADER):
            raise TrajectoryError(f"{path}:{line_number}: expected {len(_CSV_HEADER)} columns")
        seconds, incumbent, bound = row
        try:
            points.append(
                TrajectoryPoint(
                    float(seconds), None if incumbent == "" else float(incumbent), float(bound)
                )
            )
        except ValueError:
            raise TrajectoryError(f"{path}:{line_number}: values MUST be numbers") from None
    return points
