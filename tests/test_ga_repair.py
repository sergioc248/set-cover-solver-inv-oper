import numpy
import pytest

from set_cover.ga.config import PruneOrder
from set_cover.ga.problem import CoverProblem, UncoverableError, evaluate
from set_cover.ga.repair import greedy_complete, remove_redundant, repair_and_prune
from set_cover.instance import Instance
from tests.ga_support import random_problem, real_instance


@pytest.fixture(scope="module")
def problem() -> CoverProblem:
    return CoverProblem.from_instance(real_instance())


@pytest.mark.parametrize("seed", range(25))
def test_repair_makes_random_chromosomes_feasible(problem: CoverProblem, seed: int) -> None:
    rng = numpy.random.default_rng(seed)
    genes = rng.random(500) < rng.choice([0.0, 0.01, 0.05])
    instance = real_instance()

    repair_and_prune(problem, genes, PruneOrder.COST_DESC, rng)

    assert evaluate(problem, genes).feasible
    assert instance.uncovered_customers((numpy.flatnonzero(genes) + 1).tolist()) == []


@pytest.mark.parametrize("order", list(PruneOrder))
@pytest.mark.parametrize("seed", range(10))
def test_pruning_keeps_coverage_and_never_raises_cost(
    problem: CoverProblem, order: PruneOrder, seed: int
) -> None:
    rng = numpy.random.default_rng(seed)
    genes = rng.random(500) < 0.2
    counts = problem.coverage_counts(genes)
    greedy_complete(problem, genes, counts, rng)
    before = problem.cost(genes)

    removed = remove_redundant(problem, genes, counts, order, rng)

    assert removed > 0
    assert problem.cost(genes) <= before
    assert evaluate(problem, genes).feasible
    assert counts.tolist() == problem.coverage_counts(genes).tolist()
    for antenna in numpy.flatnonzero(genes):
        without = genes.copy()
        without[antenna] = False
        assert not evaluate(problem, without).feasible, "a redundant antenna survived"


def test_greedy_prefers_best_coverage_to_cost_ratio(small_instance: Instance) -> None:
    problem = CoverProblem.from_instance(small_instance)
    genes = problem.empty_genes()

    cost = repair_and_prune(problem, genes, PruneOrder.COST_DESC, numpy.random.default_rng(0))

    assert (numpy.flatnonzero(genes) + 1).tolist() == [2, 3]
    assert cost == 5.0


def test_repair_is_deterministic_for_a_seed(problem: CoverProblem) -> None:
    results = []
    for _ in range(2):
        rng = numpy.random.default_rng(3)
        genes = problem.empty_genes()
        repair_and_prune(problem, genes, PruneOrder.RANDOM, rng, rcl_alpha=0.5)
        results.append(genes.tobytes())
    assert results[0] == results[1]


def test_randomized_greedy_produces_distinct_feasible_solutions(problem: CoverProblem) -> None:
    rng = numpy.random.default_rng(11)
    solutions = set()
    for _ in range(10):
        genes = problem.empty_genes()
        repair_and_prune(problem, genes, PruneOrder.COST_DESC, rng, rcl_alpha=0.3)
        assert evaluate(problem, genes).feasible
        solutions.add(genes.tobytes())
    assert len(solutions) > 1


def test_repair_reports_uncoverable_customers_instead_of_looping() -> None:
    valid = random_problem(0, customers=3, antennas=3, density=0.5)
    coverage = valid.coverage.copy()
    coverage[2, :] = False
    broken = CoverProblem(
        coverage=coverage,
        costs=valid.costs,
        by_customer=coverage.astype(numpy.float32),
        by_antenna=numpy.ascontiguousarray(coverage.T.astype(numpy.float32)),
        customers_of=tuple(numpy.flatnonzero(column) for column in coverage.T),
        coverage_sizes=coverage.sum(axis=0).astype(numpy.float64),
        ratio_weights=1.0 / valid.costs,
    )
    genes = broken.empty_genes()

    with pytest.raises(UncoverableError, match=r"\[3\]"):
        greedy_complete(broken, genes, broken.coverage_counts(genes), numpy.random.default_rng(0))
