"""Carga y validación de la instancia de set cover ponderado."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy
from numpy.typing import NDArray

_COSTS_HEADER = ("antenna_index_1based", "cost")


class InstanceError(ValueError):
    pass


@dataclass(frozen=True)
class Instance:
    """Matriz de cobertura ``a[i, j]`` (cliente i, antena j) y costos de las antenas.

    Internamente las antenas se indexan desde 0; en toda salida al usuario
    (``antenna_ids``) se indexan desde 1, igual que el archivo de costos.
    """

    coverage: NDArray[numpy.int8]
    costs: NDArray[numpy.float64]

    @property
    def num_customers(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def num_antennas(self) -> int:
        return int(self.coverage.shape[1])

    @property
    def integral_costs(self) -> bool:
        return bool(numpy.all(self.costs == numpy.round(self.costs)))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(numpy.asarray(self.coverage.shape, dtype=numpy.int64).tobytes())
        digest.update(numpy.ascontiguousarray(self.coverage).tobytes())
        digest.update(numpy.ascontiguousarray(self.costs).tobytes())
        return digest.hexdigest()

    def cost_of(self, antenna_ids: list[int]) -> float:
        return float(self.costs[numpy.asarray(antenna_ids, dtype=numpy.int64) - 1].sum())

    def uncovered_customers(self, antenna_ids: list[int]) -> list[int]:
        """Clientes (base 1) no cubiertos por las antenas dadas (base 1)."""
        columns = numpy.asarray(antenna_ids, dtype=numpy.int64) - 1
        covered = self.coverage[:, columns].sum(axis=1) >= 1
        return (numpy.flatnonzero(~covered) + 1).tolist()


def load_instance(coverage_path: Path, costs_path: Path) -> Instance:
    coverage = _load_coverage(coverage_path)
    costs = _load_costs(costs_path)
    if costs.shape[0] != coverage.shape[1]:
        raise InstanceError(
            f"{costs.shape[0]} costs for {coverage.shape[1]} antennas in {coverage_path}"
        )
    uncoverable = numpy.flatnonzero(coverage.sum(axis=1) == 0)
    if uncoverable.size:
        raise InstanceError(f"customers with no covering antenna: {(uncoverable + 1).tolist()}")
    return Instance(coverage=coverage, costs=costs)


def _load_coverage(path: Path) -> NDArray[numpy.int8]:
    """Lee el CSV cliente x antena cuya fila de encabezado numera las antenas 0..n-1."""
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) < 2:
        raise InstanceError(f"{path}: expected a header row and at least one customer row")
    header = [cell.strip() for cell in lines[0].split(",")]
    if header != [str(j) for j in range(len(header))]:
        raise InstanceError(f"{path}: header MUST label antennas 0..{len(header) - 1}")
    rows: list[list[int]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(header):
            raise InstanceError(
                f"{path}:{line_number}: {len(cells)} columns, header has {len(header)}"
            )
        if any(cell not in ("0", "1") for cell in cells):
            raise InstanceError(f"{path}:{line_number}: coverage values MUST be 0 or 1")
        rows.append([int(cell) for cell in cells])
    return numpy.asarray(rows, dtype=numpy.int8)


def _load_costs(path: Path) -> NDArray[numpy.float64]:
    """Lee el CSV ``antenna_index_1based,cost``; los ids DEBEN ir de 1..n en orden."""
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or [cell.strip() for cell in lines[0].split(",")] != list(_COSTS_HEADER):
        raise InstanceError(f"{path}: header MUST be {','.join(_COSTS_HEADER)}")
    values: list[float] = []
    for line_number, line in enumerate(lines[1:], start=2):
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 2:
            raise InstanceError(f"{path}:{line_number}: expected 2 columns, found {len(cells)}")
        antenna, cost_text = cells
        if antenna != str(len(values) + 1):
            raise InstanceError(f"{path}:{line_number}: expected antenna id {len(values) + 1}")
        try:
            cost = float(cost_text)
        except ValueError:
            raise InstanceError(
                f"{path}:{line_number}: cost is not a number: {cost_text!r}"
            ) from None
        if not numpy.isfinite(cost) or cost < 0:
            raise InstanceError(f"{path}:{line_number}: cost is invalid: {cost_text!r}")
        values.append(cost)
    return numpy.asarray(values, dtype=numpy.float64)
