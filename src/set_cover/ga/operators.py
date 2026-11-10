"""Selección, cruce y mutación. Ningún operador modifica a los padres."""

import numpy
from numpy.typing import NDArray

from set_cover.ga.config import CrossoverKind, MutationKind
from set_cover.ga.problem import Genes


def tournament(fitness: NDArray[numpy.float64], size: int, rng: numpy.random.Generator) -> int:
    """Índice del individuo más barato entre ``size`` sorteados con reemplazo."""
    contenders = rng.integers(0, fitness.size, size)
    return int(contenders[numpy.argmin(fitness[contenders])])


def uniform_crossover(
    first: Genes, second: Genes, rng: numpy.random.Generator
) -> tuple[Genes, Genes]:
    """Cada gen viene de un padre con probabilidad 1/2; el segundo hijo es el complemento."""
    mask = rng.random(first.size) < 0.5
    return numpy.where(mask, first, second), numpy.where(mask, second, first)


def one_point_crossover(
    first: Genes, second: Genes, rng: numpy.random.Generator
) -> tuple[Genes, Genes]:
    """Intercambia las colas de los padres a partir de un punto de corte en 1..n-1."""
    if first.size < 2:
        return first.copy(), second.copy()
    point = int(rng.integers(1, first.size))
    return (
        numpy.concatenate((first[:point], second[point:])),
        numpy.concatenate((second[:point], first[point:])),
    )


def crossover(
    kind: CrossoverKind, first: Genes, second: Genes, rng: numpy.random.Generator
) -> tuple[Genes, Genes]:
    if kind is CrossoverKind.UNIFORM:
        return uniform_crossover(first, second, rng)
    return one_point_crossover(first, second, rng)


def bit_flip(genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    """Invierte cada gen con probabilidad ``rate``, en el lugar; retorna cuántos invirtió."""
    flips = rng.random(genes.size) < rate
    genes ^= flips
    return int(flips.sum())


def _move_count(genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    return int(rng.binomial(genes.size, rate))


def add_mutation(genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    """Activa antenas inactivas elegidas al azar, en el lugar; retorna cuántas activó."""
    inactive = numpy.flatnonzero(~genes)
    count = min(_move_count(genes, rate, rng), inactive.size)
    genes[rng.choice(inactive, count, replace=False)] = True
    return count


def remove_mutation(genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    """Desactiva antenas activas elegidas al azar, en el lugar; retorna cuántas desactivó.

    La reparación posterior debe cubrir a los clientes que queden sin antena con otras.
    """
    active = numpy.flatnonzero(genes)
    count = min(_move_count(genes, rate, rng), active.size)
    genes[rng.choice(active, count, replace=False)] = False
    return count


def swap_mutation(genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    """Cambia antenas activas por inactivas, una por una, en el lugar; retorna los cambios."""
    active = numpy.flatnonzero(genes)
    inactive = numpy.flatnonzero(~genes)
    count = min(_move_count(genes, rate, rng), active.size, inactive.size)
    genes[rng.choice(active, count, replace=False)] = False
    genes[rng.choice(inactive, count, replace=False)] = True
    return count


_SIMPLE_MUTATIONS = (
    MutationKind.BIT_FLIP,
    MutationKind.ADD,
    MutationKind.REMOVE,
    MutationKind.SWAP,
)


def mutate(kind: MutationKind, genes: Genes, rate: float, rng: numpy.random.Generator) -> int:
    """Aplica el operador ``kind`` en el lugar; retorna cuántos genes cambió."""
    if kind is MutationKind.MIXED:
        kind = _SIMPLE_MUTATIONS[int(rng.integers(len(_SIMPLE_MUTATIONS)))]
    if kind is MutationKind.ADD:
        return add_mutation(genes, rate, rng)
    if kind is MutationKind.REMOVE:
        return remove_mutation(genes, rate, rng)
    if kind is MutationKind.SWAP:
        return 2 * swap_mutation(genes, rate, rng)
    return bit_flip(genes, rate, rng)
