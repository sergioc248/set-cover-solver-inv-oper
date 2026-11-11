"""Estado compartido de la interfaz: instancia, cachés de resultados y sesiones del usuario."""

import dataclasses
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import streamlit
from plotly import graph_objects

from set_cover.checkpoint import CheckpointError
from set_cover.ga.calibration import RunSummary
from set_cover.ga.jsonio import RecordError, read_json_lines, to_mapping
from set_cover.ga.problem import CoverProblem, UncoverableError
from set_cover.instance import Instance, InstanceError, load_instance
from set_cover.run import checkpoint_path, is_locked
from set_cover.trajectory import TrajectoryError
from set_cover.ui import session
from set_cover.ui.data import (
    LOG_NAME,
    TRAJECTORY_NAME,
    Curve,
    ExactRun,
    GARunRecord,
    Paths,
    discover_exact_runs,
    display_rows,
    load_exact_run,
    read_convergence,
    read_csv_rows,
    read_ga_records,
)
from set_cover.ui.solver_jobs import SolverJobs

PATHS = Paths()

# Errores de lectura de archivos de resultados que la interfaz muestra en lugar de fallar.
RESULT_ERRORS = (
    OSError,
    ValueError,
    RecordError,
    CheckpointError,
    TrajectoryError,
    json.JSONDecodeError,
)


def fail(message: str) -> NoReturn:
    streamlit.error(message)
    streamlit.stop()


@streamlit.cache_resource(show_spinner="Cargando la instancia…")
def _problem(coverage: str, costs: str) -> tuple[Instance, CoverProblem]:
    instance = load_instance(Path(coverage), Path(costs))
    return instance, CoverProblem.from_instance(instance)


def current_problem() -> tuple[Instance, CoverProblem]:
    try:
        return _problem(str(PATHS.coverage), str(PATHS.costs))
    except (InstanceError, UncoverableError, OSError):
        fail(
            f"No se pudo cargar la instancia ({PATHS.coverage}, {PATHS.costs}): "
            f"{sys.exception()}. Ejecute la interfaz desde la raíz del proyecto."
        )


@streamlit.cache_resource
def solver_jobs() -> SolverJobs:
    return SolverJobs()


def stamp(*paths: Path) -> tuple[int, ...]:
    """Huella de modificación de los archivos; invalida las cachés cuando cambian."""
    return tuple(path.stat().st_mtime_ns if path.exists() else -1 for path in paths)


@streamlit.cache_data(show_spinner=False)
def _csv_rows(path: str, version: tuple[int, ...]) -> list[dict[str, str]]:
    return read_csv_rows(Path(path))


def csv_rows(path: Path) -> list[dict[str, str]]:
    return _csv_rows(str(path), stamp(path))


@streamlit.cache_data(show_spinner=False)
def _json_object(path: str, version: tuple[int, ...]) -> dict[str, object]:
    return to_mapping(json.loads(Path(path).read_text(encoding="utf-8")), path)


def json_object(path: Path) -> dict[str, object]:
    return _json_object(str(path), stamp(path))


@streamlit.cache_data(show_spinner="Leyendo ejecuciones…")
def _ga_records(path: str, version: tuple[int, ...]) -> list[GARunRecord]:
    return read_ga_records(Path(path))


def ga_records(path: Path) -> list[GARunRecord]:
    return _ga_records(str(path), stamp(path))


@streamlit.cache_data(show_spinner=False)
def _convergence(path: str, version: tuple[int, ...]) -> list[Curve]:
    return read_convergence(Path(path))


def convergence(path: Path) -> list[Curve]:
    return _convergence(str(path), stamp(path))


@streamlit.cache_data(show_spinner="Leyendo la calibración…")
def _stage_costs(path: str, version: tuple[int, ...]) -> dict[str, list[float]]:
    costs: dict[str, list[float]] = {}
    for record in read_json_lines(Path(path)):
        summary = RunSummary.from_record(record)
        costs.setdefault(summary.config_id, []).append(summary.best_cost)
    return costs


def stage_costs(path: Path) -> dict[str, list[float]]:
    """Costos de cada ejecución de una etapa, agrupados por configuración."""
    return _stage_costs(str(path), stamp(path))


@streamlit.cache_data(show_spinner="Leyendo el solver…")
def _exact_run(directory: str, version: tuple[int, ...]) -> ExactRun:
    return load_exact_run(Path(directory))


def exact_run(directory: Path) -> ExactRun:
    files = (checkpoint_path(directory), directory / LOG_NAME, directory / TRAJECTORY_NAME)
    cached = _exact_run(str(directory), stamp(*files))
    # El lock cambia sin tocar los archivos: se consulta siempre.
    return dataclasses.replace(cached, running=is_locked(directory))


def saved_exact_runs() -> list[Path]:
    return discover_exact_runs([PATHS.saved_solver_runs])


def other_exact_runs() -> list[Path]:
    """Ejecuciones en ``runs/``: de sesiones anteriores, de esta o lanzadas desde la terminal."""
    return discover_exact_runs([PATHS.solver_runs])


def exact_run_choices(include_other: bool) -> dict[str, Path]:
    """Ejecuciones del solver para elegir, con etiquetas que dicen de dónde vienen.

    Siempre las guardadas del proyecto y las de esta sesión; las demás de ``runs/`` solo si
    ``include_other``.
    """
    choices = {f"Gurobi · {path.name} (guardada)": path for path in saved_exact_runs()}
    mine = [path for path in session.solver_runs() if checkpoint_path(path).is_file()]
    choices.update({f"Gurobi · {path.name} (esta sesión)": path for path in mine})
    if include_other:
        choices.update(
            {
                f"Gurobi · {path.name} (runs/)": path
                for path in other_exact_runs()
                if path not in mine
            }
        )
    return choices


def default_exact_choices(labels: Iterable[str]) -> list[str]:
    """Las guardadas y las de esta sesión."""
    return [label for label in labels if not label.endswith("(runs/)")]


def options_key(prefix: str, options: Iterable[str]) -> str:
    """Clave de widget que cambia cuando cambian las opciones.

    Streamlit conserva la selección de un widget con clave fija aunque aparezcan opciones
    nuevas; con esta clave, una ejecución nueva vuelve a aplicar la selección por defecto.
    """
    digest = hashlib.sha1("\n".join(options).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def chart(figure: graph_objects.Figure, key: str, how_to_read: str) -> None:
    """Gráfico con su guía de lectura debajo."""
    streamlit.plotly_chart(figure, key=key)
    streamlit.caption(f":material/visibility: **Cómo leerlo.** {how_to_read}")


def show_table(
    rows: Sequence[Mapping[str, object]], decimals: Mapping[str, int | None] | None = None
) -> None:
    """Tabla con números en formato español, alineados a la derecha, y ``—`` si faltan."""
    formatted, numeric = display_rows(rows, decimals or {})
    streamlit.dataframe(
        formatted,
        hide_index=True,
        column_config={
            column: streamlit.column_config.TextColumn(alignment="right") for column in numeric
        },
    )
