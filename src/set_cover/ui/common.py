"""Estado compartido de la interfaz: instancia, cachés de resultados y sesiones del usuario."""

import dataclasses
import json
import sys
from pathlib import Path
from typing import NoReturn

import streamlit

from set_cover.checkpoint import CheckpointError
from set_cover.ga.calibration import RunSummary
from set_cover.ga.engine import GAResult
from set_cover.ga.jsonio import RecordError, read_json_lines, to_mapping
from set_cover.ga.problem import CoverProblem, UncoverableError
from set_cover.instance import Instance, InstanceError, load_instance
from set_cover.run import checkpoint_path, is_locked
from set_cover.trajectory import TrajectoryError
from set_cover.ui.data import (
    LOG_NAME,
    TRAJECTORY_NAME,
    Curve,
    ExactRun,
    GARunRecord,
    Paths,
    discover_exact_runs,
    load_exact_run,
    read_convergence,
    read_csv_rows,
    read_ga_records,
)
from set_cover.ui.solver_jobs import SolverJobs

PATHS = Paths()
GA_RUNS_KEY = "ga_runs"

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


def exact_run_directories() -> list[Path]:
    return discover_exact_runs([PATHS.saved_solver_runs, PATHS.solver_runs])


def session_ga_runs() -> list[GAResult]:
    """Ejecuciones del GA hechas en esta sesión del navegador, en orden."""
    stored = streamlit.session_state.get(GA_RUNS_KEY)
    if not isinstance(stored, list):
        return []
    return [run for run in stored if isinstance(run, GAResult)]


def add_session_ga_run(result: GAResult) -> None:
    streamlit.session_state[GA_RUNS_KEY] = [*session_ga_runs(), result]


def session_run_label(index: int, result: GAResult) -> str:
    return f"GA sesión #{index + 1} (semilla {result.seed})"
