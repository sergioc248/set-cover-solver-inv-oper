"""Lo que el usuario hizo en esta sesión del navegador. «Restablecer» lo olvida todo."""

from dataclasses import dataclass
from pathlib import Path

import streamlit

from set_cover.ga.calibration import CalibrationSettings
from set_cover.ga.engine import GAResult
from set_cover.ui.data import GAGroup, result_group

_GA_BATCHES = "session-ga-batches"
_CALIBRATIONS = "session-calibrations"
_SOLVER_RUNS = "session-solver-runs"


@dataclass(frozen=True)
class GABatch:
    """Una o más semillas ejecutadas con la misma configuración desde la interfaz."""

    number: int
    source: str
    results: tuple[GAResult, ...]

    @property
    def label(self) -> str:
        seeds = len(self.results)
        return f"Lote {self.number} · {self.source} · {seeds} semilla{'s' if seeds > 1 else ''}"

    def group(self) -> GAGroup:
        return result_group(f"GA · lote {self.number} ({self.source})", self.results)


@dataclass(frozen=True)
class SessionCalibration:
    """Calibración lanzada desde la interfaz; sus archivos viven en ``directory``."""

    directory: Path
    settings: CalibrationSettings

    @property
    def label(self) -> str:
        return f"Calibración de la sesión: {self.directory.name}"


def ga_batches() -> list[GABatch]:
    stored = streamlit.session_state.get(_GA_BATCHES)
    if not isinstance(stored, list):
        return []
    return [batch for batch in stored if isinstance(batch, GABatch)]


def add_ga_batch(source: str, results: list[GAResult]) -> GABatch:
    batches = ga_batches()
    batch = GABatch(len(batches) + 1, source, tuple(results))
    streamlit.session_state[_GA_BATCHES] = [*batches, batch]
    return batch


def calibrations() -> list[SessionCalibration]:
    stored = streamlit.session_state.get(_CALIBRATIONS)
    if not isinstance(stored, list):
        return []
    return [item for item in stored if isinstance(item, SessionCalibration)]


def add_calibration(calibration: SessionCalibration) -> None:
    known = [item for item in calibrations() if item.directory != calibration.directory]
    streamlit.session_state[_CALIBRATIONS] = [*known, calibration]


def solver_runs() -> list[Path]:
    """Directorios del solver iniciados o reanudados en esta sesión."""
    stored = streamlit.session_state.get(_SOLVER_RUNS)
    if not isinstance(stored, list):
        return []
    return [path for path in stored if isinstance(path, Path)]


def add_solver_run(directory: Path) -> None:
    if directory not in solver_runs():
        streamlit.session_state[_SOLVER_RUNS] = [*solver_runs(), directory]


def counts() -> tuple[int, int, int]:
    """Lotes del GA, calibraciones y ejecuciones del solver de esta sesión."""
    return len(ga_batches()), len(calibrations()), len(solver_runs())


def reset() -> None:
    """Olvida todo lo hecho en la sesión, incluidos los valores de los formularios."""
    streamlit.session_state.clear()
