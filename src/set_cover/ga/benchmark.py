"""Mide el costo de las operaciones del algoritmo genético antes de decidir optimizaciones.

Compara tres formas de verificar cobertura (conteos por producto matriz-vector,
máscara booleana y bitsets de 64 bits) y mide reparación, eliminación de
redundancias y la evaluación completa de un descendiente.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy
from numpy.typing import NDArray

from set_cover.ga.config import PruneOrder
from set_cover.ga.operators import bit_flip, uniform_crossover
from set_cover.ga.problem import CoverProblem, Genes
from set_cover.ga.repair import greedy_complete, remove_redundant, repair_and_prune


@dataclass(frozen=True)
class Measurement:
    name: str
    microseconds: float


def _time(name: str, operation: Callable[[], object], repeat: int) -> Measurement:
    operation()
    started = time.perf_counter()
    for _ in range(repeat):
        operation()
    return Measurement(name, (time.perf_counter() - started) / repeat * 1e6)


def _bitsets(problem: CoverProblem) -> NDArray[numpy.uint64]:
    """Un bitset de clientes por antena, empaquetado en palabras de 64 bits."""
    words = -(-problem.num_customers // 64)
    padded = numpy.zeros((problem.num_antennas, words * 64), dtype=numpy.bool_)
    padded[:, : problem.num_customers] = problem.coverage.T
    bits = padded.reshape(problem.num_antennas, words, 64).astype(numpy.uint64)
    shifts = numpy.arange(64, dtype=numpy.uint64)
    return numpy.asarray(numpy.bitwise_or.reduce(bits << shifts, axis=2), dtype=numpy.uint64)


def run_benchmark(problem: CoverProblem, repeat: int, seed: int) -> list[Measurement]:
    rng = numpy.random.default_rng(seed)
    solutions: list[Genes] = []
    for _ in range(20):
        genes = problem.empty_genes()
        repair_and_prune(problem, genes, PruneOrder.COST_DESC, rng, rcl_alpha=0.3)
        solutions.append(genes)
    solution = solutions[0]
    bitsets = _bitsets(problem)
    full = numpy.bitwise_or.reduce(bitsets, axis=0)

    def feasible_by_counts() -> bool:
        return bool((problem.coverage_counts(solution) > 0).all())

    def feasible_by_mask() -> bool:
        return bool(problem.coverage[:, solution].any(axis=1).all())

    def feasible_by_bitset() -> bool:
        return bool(numpy.array_equal(numpy.bitwise_or.reduce(bitsets[solution], axis=0), full))

    def greedy_from_empty() -> None:
        genes = problem.empty_genes()
        greedy_complete(problem, genes, problem.coverage_counts(genes), rng)

    def child() -> Genes:
        first, second = solutions[int(rng.integers(len(solutions)))], solutions[0]
        offspring, _ = uniform_crossover(first, second, rng)
        bit_flip(offspring, 0.005, rng)
        return offspring

    def repair_child() -> None:
        genes = child()
        greedy_complete(problem, genes, problem.coverage_counts(genes), rng)

    def prune_child() -> None:
        genes = child()
        counts = problem.coverage_counts(genes)
        greedy_complete(problem, genes, counts, rng)
        remove_redundant(problem, genes, counts, PruneOrder.COST_DESC, rng)

    def full_offspring() -> None:
        repair_and_prune(problem, child(), PruneOrder.COST_DESC, rng)

    return [
        _time("coverage counts (float32 matrix-vector)", feasible_by_counts, repeat),
        _time("feasibility (boolean mask)", feasible_by_mask, repeat),
        _time("feasibility (64-bit bitsets)", feasible_by_bitset, repeat),
        _time("crossover + mutation", child, repeat),
        _time("greedy construction from empty", greedy_from_empty, repeat),
        _time("child: crossover + mutation + repair", repair_child, repeat),
        _time("child: ... + redundancy removal", prune_child, repeat),
        _time("child: full evaluation (repair_and_prune)", full_offspring, repeat),
    ]
