import numpy
import pytest

from set_cover.ga.problem import CoverProblem, UncoverableError, evaluate, genes_from_ids
from set_cover.instance import Instance
from tests.ga_support import DATA, real_instance


def test_real_instance_has_500_customers_and_antennas() -> None:
    problem = CoverProblem.from_instance(real_instance())

    assert problem.num_customers == 500
    assert problem.num_antennas == 500
    assert problem.empty_genes().shape == (500,)
    assert problem.empty_genes().dtype == numpy.bool_
    assert problem.coverage.any(axis=1).all()


def test_costs_match_the_one_based_cost_file_unchanged() -> None:
    instance = real_instance()
    problem = CoverProblem.from_instance(instance)
    rows = (DATA / "antenna_costs_500.csv").read_text(encoding="utf-8").splitlines()[1:]
    expected = [float(row.split(",")[1]) for row in rows]

    assert problem.costs.tolist() == expected
    # La columna 0 del CSV de cobertura es la antena 1 del archivo de costos.
    assert problem.costs[0] == 3771.0
    assert evaluate(problem, genes_from_ids(problem, [1])).cost == 3771.0
    assert float(problem.costs.sum()) == 1_516_821.0


def test_evaluation_counts_cost_and_coverage(small_instance: Instance) -> None:
    problem = CoverProblem.from_instance(small_instance)

    feasible = evaluate(problem, genes_from_ids(problem, [2, 3]))
    partial = evaluate(problem, genes_from_ids(problem, [1]))

    assert (feasible.cost, feasible.covered, feasible.feasible) == (5.0, 4, True)
    assert feasible.antenna_ids == (2, 3)
    assert (partial.cost, partial.covered, partial.uncovered) == (4.0, 2, (3, 4))
    assert not partial.feasible


def test_evaluation_matches_a_direct_loop_on_the_real_instance() -> None:
    instance = real_instance()
    problem = CoverProblem.from_instance(instance)
    rng = numpy.random.default_rng(7)
    for _ in range(20):
        genes = rng.random(500) < 0.03
        evaluation = evaluate(problem, genes)
        selected = [j for j in range(500) if genes[j]]
        covered = sum(any(instance.coverage[i, j] == 1 for j in selected) for i in range(500))
        assert evaluation.covered == covered
        assert evaluation.cost == pytest.approx(sum(instance.costs[j] for j in selected))
        assert list(evaluation.uncovered) == instance.uncovered_customers([j + 1 for j in selected])


def test_coverage_counts_match_matrix_sums(small_instance: Instance) -> None:
    problem = CoverProblem.from_instance(small_instance)
    genes = genes_from_ids(problem, [1, 2, 3])

    assert problem.coverage_counts(genes).tolist() == [2, 2, 1, 1]


@pytest.mark.parametrize(
    "genes",
    [numpy.zeros(3, dtype=numpy.bool_), numpy.zeros(4, dtype=numpy.int8)],
)
def test_rejects_malformed_chromosomes(small_instance: Instance, genes: numpy.ndarray) -> None:
    with pytest.raises(ValueError, match="boolean vector of length 4"):
        evaluate(CoverProblem.from_instance(small_instance), genes)


def test_detects_uncoverable_customers() -> None:
    with pytest.raises(UncoverableError, match=r"\[2\]"):
        CoverProblem.build(
            numpy.asarray([[1, 0], [0, 0]], dtype=numpy.int8), numpy.asarray([1.0, 1.0])
        )


def test_rejects_out_of_range_antenna_ids(small_instance: Instance) -> None:
    with pytest.raises(ValueError, match=r"1\.\.4"):
        genes_from_ids(CoverProblem.from_instance(small_instance), [0])
