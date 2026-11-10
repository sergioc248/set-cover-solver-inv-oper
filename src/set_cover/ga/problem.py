"""Instancia precalculada para el algoritmo genético y evaluación exacta de cromosomas.

Un cromosoma es un vector booleano de longitud ``n`` (una posición por antena,
indexada desde 0). ``genes[j] = True`` significa que se instala la antena ``j + 1``
del archivo de costos.
"""

from dataclasses import dataclass

import numpy
from numpy.typing import NDArray

from set_cover.instance import Instance

type Genes = NDArray[numpy.bool_]
type Counts = NDArray[numpy.int32]

# Piso del costo al calcular la razón cobertura/costo, para que una antena de costo 0 no divida
# por cero. Solo afecta el orden de preferencia, nunca el costo reportado.
_MIN_COST_FOR_RATIO = 1e-12


class UncoverableError(ValueError):
    """Algún cliente no lo cubre ninguna antena, así que ninguna solución es factible."""

    def __init__(self, customers: list[int]) -> None:
        super().__init__(f"customers with no covering antenna: {customers}")
        self.customers = customers


@dataclass(frozen=True, eq=False)
class CoverProblem:
    """Matriz de cobertura en las formas que usan los operadores, calculadas una sola vez.

    ``coverage[i, j]`` vale True si la antena ``j`` cubre al cliente ``i`` (base 0).
    """

    coverage: NDArray[numpy.bool_]
    costs: NDArray[numpy.float64]
    # a[i, j] en float32: ``by_customer @ x`` da cuántas antenas de x cubren a cada cliente.
    by_customer: NDArray[numpy.float32]
    # a[i, j] transpuesta: ``by_antenna @ u`` da cuántos clientes de u cubre cada antena.
    by_antenna: NDArray[numpy.float32]
    # Clientes que cubre cada antena; evita recorrer la matriz al actualizar conteos.
    customers_of: tuple[NDArray[numpy.intp], ...]
    # Clientes que cubre cada antena, y el inverso del costo usado en la razón voraz.
    coverage_sizes: NDArray[numpy.float64]
    ratio_weights: NDArray[numpy.float64]

    @property
    def num_customers(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def num_antennas(self) -> int:
        return int(self.coverage.shape[1])

    @staticmethod
    def from_instance(instance: Instance) -> "CoverProblem":
        return CoverProblem.build(instance.coverage, instance.costs)

    @staticmethod
    def build(
        coverage: NDArray[numpy.integer] | NDArray[numpy.bool_], costs: NDArray[numpy.floating]
    ) -> "CoverProblem":
        matrix = numpy.asarray(coverage)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("coverage MUST be a non-empty customers x antennas matrix")
        if not numpy.isin(matrix, (0, 1)).all():
            raise ValueError("coverage values MUST be 0 or 1")
        cost_vector = numpy.asarray(costs, dtype=numpy.float64)
        if cost_vector.shape != (matrix.shape[1],):
            raise ValueError(f"{cost_vector.size} costs for {matrix.shape[1]} antennas")
        if not numpy.isfinite(cost_vector).all() or (cost_vector < 0).any():
            raise ValueError("costs MUST be finite and non-negative")
        covered = matrix.astype(numpy.bool_)
        uncoverable = numpy.flatnonzero(~covered.any(axis=1))
        if uncoverable.size:
            raise UncoverableError((uncoverable + 1).tolist())
        by_customer = numpy.ascontiguousarray(covered, dtype=numpy.float32)
        return CoverProblem(
            coverage=covered,
            costs=cost_vector,
            by_customer=by_customer,
            by_antenna=numpy.ascontiguousarray(by_customer.T),
            customers_of=tuple(numpy.flatnonzero(column) for column in covered.T),
            coverage_sizes=covered.sum(axis=0).astype(numpy.float64),
            ratio_weights=1.0 / numpy.maximum(cost_vector, _MIN_COST_FOR_RATIO),
        )

    def empty_genes(self) -> Genes:
        return numpy.zeros(self.num_antennas, dtype=numpy.bool_)

    def coverage_counts(self, genes: Genes) -> Counts:
        """Cuántas antenas seleccionadas cubren a cada cliente."""
        return (self.by_customer @ genes.astype(numpy.float32)).astype(numpy.int32)

    def cost(self, genes: Genes) -> float:
        return float(self.costs[genes].sum())


@dataclass(frozen=True)
class Evaluation:
    """Métricas de un cromosoma calculadas desde la matriz original, sin estado incremental."""

    cost: float
    antenna_ids: tuple[int, ...]
    covered: int
    uncovered: tuple[int, ...]

    @property
    def num_antennas(self) -> int:
        return len(self.antenna_ids)

    @property
    def feasible(self) -> bool:
        return not self.uncovered


def evaluate(problem: CoverProblem, genes: Genes) -> Evaluation:
    """Evalúa ``genes``; antenas y clientes del resultado se numeran desde 1."""
    if genes.dtype != numpy.bool_ or genes.shape != (problem.num_antennas,):
        raise ValueError(
            f"chromosome MUST be a boolean vector of length {problem.num_antennas}, "
            f"got {genes.dtype} {genes.shape}"
        )
    is_covered = problem.coverage[:, genes].any(axis=1)
    return Evaluation(
        cost=problem.cost(genes),
        antenna_ids=tuple((numpy.flatnonzero(genes) + 1).tolist()),
        covered=int(is_covered.sum()),
        uncovered=tuple((numpy.flatnonzero(~is_covered) + 1).tolist()),
    )


def genes_from_ids(problem: CoverProblem, antenna_ids: list[int]) -> Genes:
    """Cromosoma con las antenas dadas (base 1) activadas."""
    genes = problem.empty_genes()
    indices = numpy.asarray(antenna_ids, dtype=numpy.intp) - 1
    if indices.size and (indices.min() < 0 or indices.max() >= problem.num_antennas):
        raise ValueError(f"antenna ids MUST be in 1..{problem.num_antennas}")
    genes[indices] = True
    return genes
