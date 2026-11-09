"""Estado persistente del solver que sobrevive a interrupciones entre sesiones.

Gurobi no puede persistir su árbol de branch-and-bound, así que una sesión
reanudada reinicia la búsqueda. Se conservan la mejor solución incumbente (que
se reinyecta como MIP start) y la mejor cota inferior probada (válida para
siempre sobre la misma instancia).
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    pass


@dataclass
class Incumbent:
    antenna_ids: list[int]
    cost: float


@dataclass
class Session:
    started_at: str
    ended_at: str | None
    status: str
    runtime_seconds: float
    nodes: float
    target_gap: float
    best_bound: float | None
    settings: dict[str, str]


@dataclass
class Checkpoint:
    instance_fingerprint: str
    incumbent: Incumbent | None = None
    best_bound: float | None = None
    sessions: list[Session] = field(default_factory=list)

    @property
    def total_runtime_seconds(self) -> float:
        return sum(session.runtime_seconds for session in self.sessions)

    def gap(self) -> float | None:
        """Brecha relativa entre la incumbente y la mejor cota, como Gurobi define MIPGap."""
        if self.incumbent is None or self.best_bound is None:
            return None
        if self.incumbent.cost == 0:
            return 0.0
        return max(0.0, (self.incumbent.cost - self.best_bound) / abs(self.incumbent.cost))

    def to_json(self) -> str:
        return json.dumps({"schema_version": SCHEMA_VERSION, **asdict(self)}, indent=2) + "\n"

    @staticmethod
    def from_json(text: str) -> "Checkpoint":
        raw = _mapping(json.loads(text), "checkpoint")
        version = _field(raw, "schema_version", "checkpoint")
        if version != SCHEMA_VERSION:
            raise CheckpointError(f"unsupported checkpoint schema version {version!r}")
        incumbent_raw = _field(raw, "incumbent", "checkpoint")
        incumbent = None
        if incumbent_raw is not None:
            data = _mapping(incumbent_raw, "incumbent")
            incumbent = Incumbent(
                antenna_ids=_int_list(_field(data, "antenna_ids", "incumbent"), "antenna_ids"),
                cost=_number(_field(data, "cost", "incumbent"), "incumbent.cost"),
            )
        sessions_raw = _field(raw, "sessions", "checkpoint")
        if not isinstance(sessions_raw, list):
            raise CheckpointError("checkpoint.sessions MUST be a list")
        return Checkpoint(
            instance_fingerprint=_string(
                _field(raw, "instance_fingerprint", "checkpoint"), "instance_fingerprint"
            ),
            incumbent=incumbent,
            best_bound=_optional_number(_field(raw, "best_bound", "checkpoint"), "best_bound"),
            sessions=[_session(item) for item in sessions_raw],
        )


def read_checkpoint(path: Path) -> Checkpoint:
    return Checkpoint.from_json(path.read_text(encoding="utf-8"))


def write_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Reemplaza ``path`` atómicamente para que un fallo nunca deje un checkpoint truncado."""
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(descriptor, checkpoint.to_json().encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _session(value: object) -> Session:
    data = _mapping(value, "session")
    settings = _mapping(_field(data, "settings", "session"), "session.settings")
    ended_at = _field(data, "ended_at", "session")
    return Session(
        started_at=_string(_field(data, "started_at", "session"), "session.started_at"),
        ended_at=None if ended_at is None else _string(ended_at, "session.ended_at"),
        status=_string(_field(data, "status", "session"), "session.status"),
        runtime_seconds=_number(
            _field(data, "runtime_seconds", "session"), "session.runtime_seconds"
        ),
        nodes=_number(_field(data, "nodes", "session"), "session.nodes"),
        target_gap=_number(_field(data, "target_gap", "session"), "session.target_gap"),
        best_bound=_optional_number(_field(data, "best_bound", "session"), "session.best_bound"),
        settings={key: _string(item, f"session.settings.{key}") for key, item in settings.items()},
    )


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CheckpointError(f"{where} MUST be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise CheckpointError(f"{where} keys MUST be strings")
        result[key] = item
    return result


def _field(data: dict[str, object], key: str, where: str) -> object:
    if key not in data:
        raise CheckpointError(f"{where}.{key} is missing")
    return data[key]


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise CheckpointError(f"{where} MUST be a string")
    return value


def _number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointError(f"{where} MUST be a number")
    return float(value)


def _optional_number(value: object, where: str) -> float | None:
    return None if value is None else _number(value, where)


def _int_list(value: object, where: str) -> list[int]:
    if not isinstance(value, list):
        raise CheckpointError(f"{where} MUST be a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise CheckpointError(f"{where} MUST contain integers")
        result.append(item)
    return result
