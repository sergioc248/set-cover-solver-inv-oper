import pytest

from set_cover.ga.config import CrossoverKind, GAConfig, MutationKind, PruneOrder
from set_cover.ga.engine import GAResult, StopReason, run_ga
from set_cover.ga.problem import CoverProblem, evaluate
from tests.ga_support import brute_force_optimum, random_problem, real_instance

_SHORT = GAConfig(population_size=20, max_generations=30, stall_generations=None)


@pytest.fixture(scope="module")
def real_problem() -> CoverProblem:
    return CoverProblem.from_instance(real_instance())


def _comparable(result: GAResult) -> dict[str, object]:
    record = result.to_record()
    record.pop("elapsed_seconds")
    record.pop("seconds_to_best")
    history = record.pop("history")
    assert isinstance(history, dict)
    history.pop("elapsed_seconds")
    record["history"] = history
    return record


@pytest.mark.parametrize("seed", range(6))
def test_small_instances_are_feasible_and_never_below_the_optimum(seed: int) -> None:
    problem = random_problem(seed, customers=10, antennas=12, density=0.25)
    optimum = brute_force_optimum(problem)

    result = run_ga(problem, _SHORT, seed)

    assert result.evaluation.feasible
    assert result.evaluation.cost >= optimum
    assert result.evaluation.cost == problem.cost(result.best_genes)


def test_finds_the_optimum_on_most_small_instances() -> None:
    hits = 0
    for seed in range(10):
        problem = random_problem(100 + seed, customers=10, antennas=12, density=0.25)
        hits += run_ga(problem, _SHORT, seed).evaluation.cost == brute_force_optimum(problem)
    # Es una metaheurística: no se exige el óptimo siempre.
    assert hits >= 8


def test_same_seed_reproduces_the_run(real_problem: CoverProblem) -> None:
    config = GAConfig(population_size=30, max_evaluations=1500, stall_generations=None)

    first = run_ga(real_problem, config, 42)
    second = run_ga(real_problem, config, 42)
    other = run_ga(real_problem, config, 43)

    assert _comparable(first) == _comparable(second)
    assert _comparable(first) != _comparable(other)


@pytest.mark.parametrize("elite_count", [1, 2, 5])
def test_elitism_keeps_the_best_solution(real_problem: CoverProblem, elite_count: int) -> None:
    config = GAConfig(
        population_size=30,
        elite_count=elite_count,
        mutation_rate=0.05,
        max_generations=40,
        stall_generations=None,
    )
    history = run_ga(real_problem, config, 5).history

    assert history.population_best == history.best_so_far
    assert all(a >= b for a, b in zip(history.best_so_far, history.best_so_far[1:], strict=False))


def test_reported_solutions_satisfy_all_500_constraints(real_problem: CoverProblem) -> None:
    instance = real_instance()
    for seed, crossover in enumerate(CrossoverKind):
        for order in PruneOrder:
            config = _SHORT.replace(crossover=crossover.value, prune_order=order.value)
            result = run_ga(real_problem, config, seed)
            ids = list(result.evaluation.antenna_ids)
            assert result.evaluation.covered == 500
            assert instance.uncovered_customers(ids) == []
            assert instance.cost_of(ids) == result.evaluation.cost
            assert evaluate(real_problem, result.best_genes).feasible


@pytest.mark.parametrize("mutation", list(MutationKind))
def test_every_mutation_yields_feasible_solutions(
    real_problem: CoverProblem, mutation: MutationKind
) -> None:
    config = _SHORT.replace(mutation=mutation.value, mutation_rate=0.03)
    result = run_ga(real_problem, config, 4)
    assert result.evaluation.feasible
    assert real_instance().uncovered_customers(list(result.evaluation.antenna_ids)) == []
    assert run_ga(real_problem, config, 4).evaluation.cost == result.evaluation.cost


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"max_generations": 3}, StopReason.MAX_GENERATIONS),
        ({"max_generations": None, "max_evaluations": 75}, StopReason.MAX_EVALUATIONS),
        ({"max_generations": None, "stall_generations": 2}, StopReason.STALL),
        ({"max_generations": None, "time_limit": 0.2}, StopReason.TIME_LIMIT),
    ],
)
def test_stopping_criteria(
    real_problem: CoverProblem, changes: dict[str, object], reason: StopReason
) -> None:
    result = run_ga(real_problem, _SHORT.replace(**changes), 1)

    assert result.stop_reason is reason
    assert result.evaluation.feasible
    if reason is StopReason.MAX_EVALUATIONS:
        assert result.evaluations == 75
        assert result.history.evaluations[-1] == 75
    if reason is StopReason.MAX_GENERATIONS:
        assert result.generations == 3
        assert result.history.generation == [0, 1, 2, 3]


def test_initial_population_is_diverse(real_problem: CoverProblem) -> None:
    config = GAConfig(population_size=50, max_generations=0, greedy_fraction=0.5)
    result = run_ga(real_problem, config, 0)

    assert result.history.unique_individuals[0] == 50
    assert result.evaluations >= 50


def test_history_marks_improving_generations(real_problem: CoverProblem) -> None:
    history = run_ga(real_problem, _SHORT, 2).history
    best = history.best_so_far

    assert history.improved[0]
    for generation in range(1, len(best)):
        assert history.improved[generation] == (best[generation] < best[generation - 1])
