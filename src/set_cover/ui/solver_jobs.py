"""Trabajos largos lanzados desde la interfaz, como procesos independientes.

Las sesiones del solver (``set-cover solve``) y las calibraciones (``set-cover-ga
calibrate``) corren en su propio proceso y grupo de sesión: sobreviven a recargas de la
página y al cierre del navegador, y guardan su avance en su directorio. Detener una sesión
del solver envía SIGTERM, que guarda el checkpoint; detener una calibración envía SIGINT a su
grupo, que termina las ejecuciones en curso y las registra.
"""

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from set_cover.ga.calibration import CalibrationSettings, Stage
from set_cover.run import is_locked

OUTPUT_NAME = "solver-output.log"
PID_NAME = "solver.pid"


class SolverJobError(RuntimeError):
    pass


@dataclass(frozen=True)
class SolverRequest:
    run_dir: Path
    coverage: Path
    costs: Path
    target_gap: float
    time_limit: float | None
    threads: int
    seed: int


def solver_command(request: SolverRequest) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "set_cover",
        "solve",
        "--run-dir",
        str(request.run_dir),
        "--coverage",
        str(request.coverage),
        "--costs",
        str(request.costs),
        "--gap",
        repr(request.target_gap),
        "--threads",
        str(request.threads),
        "--seed",
        str(request.seed),
        # La interfaz lee el checkpoint mientras corre; un intervalo corto lo mantiene al día.
        "--checkpoint-interval",
        "2",
    ]
    if request.time_limit is not None:
        command += ["--time-limit", repr(request.time_limit)]
    return command


SEARCH_SPACE_NAME = "search_space.json"


def calibration_command(
    directory: Path,
    settings: CalibrationSettings,
    coverage: Path,
    costs: Path,
    stages: list[Stage],
) -> list[str]:
    """``set-cover-ga calibrate`` con ``settings``; lee el espacio de ``SEARCH_SPACE_NAME``."""
    command = [
        sys.executable,
        "-m",
        "set_cover.ga",
        "calibrate",
        "--coverage",
        str(coverage),
        "--costs",
        str(costs),
        "--output",
        str(directory),
        "--search-space",
        str(directory / SEARCH_SPACE_NAME),
        "--workers",
        str(settings.workers),
        "--budget",
        str(settings.evaluation_budget),
        "--explore-configs",
        str(settings.explore_configs),
        "--explore-seeds",
        str(settings.explore_seeds),
        "--refine-top",
        str(settings.refine_top),
        "--refine-neighbors",
        str(settings.refine_neighbors),
        "--refine-seeds",
        str(settings.refine_seeds),
        "--final-seeds",
        str(settings.final_seeds),
        "--similarity",
        repr(settings.similarity_tolerance),
        "--sampling-seed",
        str(settings.sampling_seed),
    ]
    for stage in stages:
        command += ["--stage", stage.value]
    return command


class SolverJobs:
    """Procesos lanzados por este servidor. Una instancia por proceso de Streamlit."""

    def __init__(self) -> None:
        self._processes: dict[Path, subprocess.Popen[bytes]] = {}
        self._stop_signals: dict[Path, signal.Signals] = {}

    def start(
        self, run_dir: Path, command: list[str], stop_signal: signal.Signals = signal.SIGTERM
    ) -> None:
        """Lanza ``command``; ``stop`` le enviará ``stop_signal`` a todo su grupo de procesos."""
        run_dir = run_dir.resolve()
        if self.running(run_dir):
            raise SolverJobError(f"a solver session is already running in {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        output = (run_dir / OUTPUT_NAME).open("ab")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            # El hijo heredó su propio descriptor; el del servidor ya no hace falta.
            output.close()
        self._processes[run_dir] = process
        self._stop_signals[run_dir] = stop_signal
        (run_dir / PID_NAME).write_text(f"{process.pid}\n", encoding="utf-8")

    def running(self, run_dir: Path) -> bool:
        """Hay un proceso vivo de esta interfaz, o cualquier solver tiene el lock del directorio."""
        process = self._processes.get(run_dir.resolve())
        if process is not None and process.poll() is None:
            return True
        return is_locked(run_dir)

    def exit_code(self, run_dir: Path) -> int | None:
        """Código de salida del último proceso lanzado aquí, o ``None`` si sigue o no existe."""
        process = self._processes.get(run_dir.resolve())
        return None if process is None else process.poll()

    def stop(self, run_dir: Path) -> bool:
        """Pide al solver que guarde y termine; retorna si había algo que detener."""
        resolved = run_dir.resolve()
        process = self._processes.get(resolved)
        if process is not None and process.poll() is None:
            # El proceso encabeza su propio grupo (start_new_session): su pid es el del grupo.
            os.killpg(process.pid, self._stop_signals.get(resolved, signal.SIGTERM))
            return True
        # Proceso lanzado por un servidor anterior: solo se detiene si aún tiene el lock.
        pid = _read_pid(run_dir)
        if pid is not None and is_locked(run_dir):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return False
            return True
        return False


def read_output(run_dir: Path, max_lines: int = 40) -> str:
    path = run_dir / OUTPUT_NAME
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _read_pid(run_dir: Path) -> int | None:
    path = run_dir / PID_NAME
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
