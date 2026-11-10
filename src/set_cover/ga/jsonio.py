"""Lectura validada de JSON y escritura duradera de archivos de resultados."""

import json
import os
from pathlib import Path


class RecordError(ValueError):
    pass


def to_mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RecordError(f"{where} MUST be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RecordError(f"{where} keys MUST be strings")
        result[key] = item
    return result


def field(data: dict[str, object], key: str, where: str) -> object:
    if key not in data:
        raise RecordError(f"{where}.{key} is missing")
    return data[key]


def int_field(data: dict[str, object], key: str, where: str) -> int:
    value = field(data, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordError(f"{where}.{key} MUST be an integer")
    return value


def float_field(data: dict[str, object], key: str, where: str) -> float:
    value = field(data, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecordError(f"{where}.{key} MUST be a number")
    return float(value)


def optional_float_field(data: dict[str, object], key: str, where: str) -> float | None:
    if field(data, key, where) is None:
        return None
    return float_field(data, key, where)


def str_field(data: dict[str, object], key: str, where: str) -> str:
    value = field(data, key, where)
    if not isinstance(value, str):
        raise RecordError(f"{where}.{key} MUST be a string")
    return value


def bool_field(data: dict[str, object], key: str, where: str) -> bool:
    value = field(data, key, where)
    if not isinstance(value, bool):
        raise RecordError(f"{where}.{key} MUST be a boolean")
    return value


def int_list_field(data: dict[str, object], key: str, where: str) -> list[int]:
    value = field(data, key, where)
    if not isinstance(value, list):
        raise RecordError(f"{where}.{key} MUST be a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RecordError(f"{where}.{key} MUST contain integers")
        result.append(item)
    return result


def float_list_field(data: dict[str, object], key: str, where: str) -> list[float]:
    value = field(data, key, where)
    if not isinstance(value, list):
        raise RecordError(f"{where}.{key} MUST be a list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RecordError(f"{where}.{key} MUST contain numbers")
        result.append(float(item))
    return result


def write_text_atomic(path: Path, text: str) -> None:
    """Reemplaza ``path`` atómicamente para que una interrupción nunca deje un archivo truncado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(descriptor, text.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json_lines(path: Path) -> list[dict[str, object]]:
    """Lee un archivo JSON Lines; descarta solo una última línea truncada por una interrupción."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, object]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise RecordError(f"{path}:{number}: invalid JSON") from None
        records.append(to_mapping(parsed, f"{path}:{number}"))
    return records


class JsonLinesWriter:
    """Agrega una línea por registro y la sincroniza a disco antes de retornar."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_truncated_tail(path)
        self._descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    @staticmethod
    def _repair_truncated_tail(path: Path) -> None:
        if not path.exists():
            return
        content = path.read_bytes()
        if content and not content.endswith(b"\n"):
            cut = content.rfind(b"\n") + 1
            descriptor = os.open(path, os.O_WRONLY)
            try:
                os.ftruncate(descriptor, cut)
            finally:
                os.close(descriptor)

    def append(self, record: dict[str, object]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        os.write(self._descriptor, line.encode("utf-8"))
        os.fsync(self._descriptor)

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
