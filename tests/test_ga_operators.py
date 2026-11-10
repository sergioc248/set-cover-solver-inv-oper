import numpy
import pytest

from set_cover.ga.config import MutationKind
from set_cover.ga.operators import (
    add_mutation,
    bit_flip,
    mutate,
    one_point_crossover,
    remove_mutation,
    swap_mutation,
    tournament,
    uniform_crossover,
)


def _parents() -> tuple[numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(1)
    return rng.random(500) < 0.1, rng.random(500) < 0.1


def test_uniform_crossover_mixes_genes_without_touching_parents() -> None:
    first, second = _parents()
    originals = first.copy(), second.copy()

    child_a, child_b = uniform_crossover(first, second, numpy.random.default_rng(0))

    assert child_a.shape == child_b.shape == (500,)
    assert child_a.dtype == numpy.bool_
    from_first = child_a == first
    assert (from_first | (child_a == second)).all()
    # Donde un hijo toma al primer padre, el otro toma al segundo.
    differ = first != second
    assert (child_b[differ & from_first] == second[differ & from_first]).all()
    assert (first == originals[0]).all()
    assert (second == originals[1]).all()


def test_one_point_crossover_swaps_tails() -> None:
    first = numpy.zeros(500, dtype=numpy.bool_)
    second = numpy.ones(500, dtype=numpy.bool_)

    child_a, child_b = one_point_crossover(first, second, numpy.random.default_rng(0))

    point = int(numpy.argmax(child_a))
    assert 1 <= point < 500
    assert not child_a[:point].any()
    assert child_a[point:].all()
    assert (child_b == ~child_a).all()


def test_bit_flip_rate_limits() -> None:
    genes, _ = _parents()
    unchanged = genes.copy()
    assert bit_flip(unchanged, 0.0, numpy.random.default_rng(0)) == 0
    assert (unchanged == genes).all()

    flipped = genes.copy()
    assert bit_flip(flipped, 1.0, numpy.random.default_rng(0)) == 500
    assert (flipped == ~genes).all()


def test_add_mutation_only_activates() -> None:
    genes, _ = _parents()
    mutated = genes.copy()

    count = add_mutation(mutated, 0.02, numpy.random.default_rng(0))

    assert count > 0
    assert (mutated[genes]).all()
    assert int(mutated.sum()) == int(genes.sum()) + count


def test_remove_mutation_only_deactivates_and_stops_at_empty() -> None:
    genes, _ = _parents()
    mutated = genes.copy()

    count = remove_mutation(mutated, 0.02, numpy.random.default_rng(0))

    assert count > 0
    assert not (mutated[~genes]).any()
    assert int(mutated.sum()) == int(genes.sum()) - count

    emptied = genes.copy()
    assert remove_mutation(emptied, 1.0, numpy.random.default_rng(0)) == int(genes.sum())
    assert not emptied.any()


def test_swap_mutation_keeps_the_number_of_active_antennas() -> None:
    genes, _ = _parents()
    mutated = genes.copy()

    count = swap_mutation(mutated, 0.02, numpy.random.default_rng(0))

    assert count > 0
    assert int(mutated.sum()) == int(genes.sum())
    assert int((mutated & ~genes).sum()) == count
    assert int((genes & ~mutated).sum()) == count


@pytest.mark.parametrize("kind", list(MutationKind))
def test_mutate_reports_the_changed_genes(kind: MutationKind) -> None:
    genes, _ = _parents()
    rng = numpy.random.default_rng(3)
    for _ in range(20):
        mutated = genes.copy()
        assert mutate(kind, mutated, 0.01, rng) == int((mutated != genes).sum())
    unchanged = genes.copy()
    assert mutate(kind, unchanged, 0.0, rng) == 0
    assert (unchanged == genes).all()


def test_tournament_returns_the_cheapest_contender() -> None:
    fitness = numpy.asarray([5.0, 1.0, 3.0, 4.0])
    rng = numpy.random.default_rng(0)
    winners = [tournament(fitness, 4, rng) for _ in range(200)]
    # Con 4 sorteos sobre 4 individuos, el más barato gana casi siempre.
    assert winners.count(1) > 120
    assert all(0 <= winner < 4 for winner in winners)
    assert {tournament(fitness, 1, rng) for _ in range(200)} == {0, 1, 2, 3}
