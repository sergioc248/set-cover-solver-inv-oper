"""Sesiones del solver exacto lanzadas desde la interfaz, como procesos independientes.

Cada sesión es ``set-cover solve`` en su propio proceso y grupo de sesión: sobrevive a
recargas de la página y al cierre del navegador, y conserva el checkpoint y el lock del
directorio de ejecución. Detenerla envía SIGTERM, que el solver atiende guardando el
checkpoint antes de salir.
"""

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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


class SolverJobs:
    """Procesos lanzados por este servidor. Una instancia por proceso de Streamlit."""

    def __init__(self) -> None:
        self._processes: dict[Path, subprocess.Popen[bytes]] = {}

    def start(self, run_dir: Path, command: list[str]) -> None:
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
        process = self._processes.get(run_dir.resolve())
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
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
