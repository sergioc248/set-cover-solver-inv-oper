"""Directorio de ejecución: un checkpoint, un log de Gurobi y un proceso del solver a la vez."""

import fcntl
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from set_cover.checkpoint import (
    Checkpoint,
    CheckpointError,
    Incumbent,
    Session,
    read_checkpoint,
    write_checkpoint,
)
from set_cover.instance import Instance

RUNNING = "running"
ABORTED = "aborted"

# Las cotas a esta distancia por debajo de un entero se tratan como ese entero.
_BOUND_ROUNDING_TOLERANCE = 1e-6


class RunLockedError(RuntimeError):
    pass


def checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint.json"


def is_locked(run_dir: Path) -> bool:
    """Indica si un proceso del solver tiene tomado ``run_dir`` en este momento."""
    lock_path = run_dir / "lock"
    if not lock_path.exists():
        return False
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(descriptor)
    return False


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunDirectory:
    """Dueño exclusivo del checkpoint de un directorio de ejecución para un proceso.

    Cada cambio de estado se escribe a disco de inmediato, así que una terminación
    forzada pierde como máximo el progreso de la cota desde la última actualización
    periódica.
    """

    def __init__(self, run_dir: Path, instance: Instance) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.instance = instance
        self._lock_descriptor = os.open(run_dir / "lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock_descriptor)
            raise RunLockedError(f"another solver process is using {run_dir}") from None
        try:
            self.checkpoint = self._load_or_create()
        except BaseException:
            self.close()
            raise

    @property
    def log_path(self) -> Path:
        return self.run_dir / "gurobi.log"

    @property
    def node_file_dir(self) -> Path:
        return self.run_dir / "nodefiles"

    def close(self) -> None:
        if self._lock_descriptor >= 0:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1

    def save(self) -> None:
        write_checkpoint(checkpoint_path(self.run_dir), self.checkpoint)

    def offer_incumbent(self, antenna_ids: list[int]) -> bool:
        """Persiste ``antenna_ids`` si es más barata que la incumbente."""
        uncovered = self.instance.uncovered_customers(antenna_ids)
        if uncovered:
            raise ValueError(f"solution leaves customers uncovered: {uncovered}")
        cost = self.instance.cost_of(antenna_ids)
        current = self.checkpoint.incumbent
        if current is not None and cost >= current.cost:
            return False
        self.checkpoint.incumbent = Incumbent(antenna_ids=sorted(antenna_ids), cost=cost)
        self.save()
        return True

    def offer_bound(self, bound: float) -> float | None:
        """Registra una cota inferior probada; retorna la cota reforzada, si es finita."""
        if not math.isfinite(bound):
            return None
        if self.instance.integral_costs:
            bound = float(math.ceil(bound - _BOUND_ROUNDING_TOLERANCE))
        if self.checkpoint.best_bound is None or bound > self.checkpoint.best_bound:
            self.checkpoint.best_bound = bound
        return bound

    def begin_session(self, target_gap: float, settings: dict[str, str]) -> Session:
        session = Session(
            started_at=utc_now(),
            ended_at=None,
            status=RUNNING,
            runtime_seconds=0.0,
            nodes=0.0,
            target_gap=target_gap,
            best_bound=None,
            settings=settings,
        )
        self.checkpoint.sessions.append(session)
        self.save()
        return session

    def _load_or_create(self) -> Checkpoint:
        path = checkpoint_path(self.run_dir)
        fingerprint = self.instance.fingerprint()
        if not path.exists():
            checkpoint = Checkpoint(instance_fingerprint=fingerprint)
            write_checkpoint(path, checkpoint)
            return checkpoint
        checkpoint = read_checkpoint(path)
        if checkpoint.instance_fingerprint != fingerprint:
            raise CheckpointError(f"{path} belongs to a different instance; use another --run-dir")
        incumbent = checkpoint.incumbent
        if incumbent is not None and self.instance.uncovered_customers(incumbent.antenna_ids):
            raise CheckpointError(f"{path}: stored incumbent does not cover every customer")
        # El lock es nuestro, así que toda sesión aún marcada "running" murió sin limpiar.
        for session in checkpoint.sessions:
            if session.status == RUNNING:
                session.status = ABORTED
        write_checkpoint(path, checkpoint)
        return checkpoint
