import itertools
from pathlib import Path

import numpy

from set_cover.ga.problem import CoverProblem
from set_cover.instance import Instance, load_instance

DATA = Path(__file__).resolve().parents[1] / "data"


def real_instance() -> Instance:
    return load_instance(DATA / "set_cover_500x500.csv", DATA / "antenna_costs_500.csv")


def random_problem(seed: int, customers: int, antennas: int, density: float) -> CoverProblem:
    """Instancia aleatoria pequeña en la que todo cliente tiene al menos una antena."""
    rng = numpy.random.default_rng(seed)
    coverage = rng.random((customers, antennas)) < density
    for customer in numpy.flatnonzero(~coverage.any(axis=1)):
        coverage[customer, rng.integers(antennas)] = True
    costs = rng.integers(1, 50, antennas).astype(numpy.float64)
    return CoverProblem.build(coverage, costs)


def brute_force_optimum(problem: CoverProblem) -> float:
    best = numpy.inf
    for size in range(1, problem.num_antennas + 1):
        for subset in itertools.combinations(range(problem.num_antennas), size):
            columns = list(subset)
            if problem.coverage[:, columns].any(axis=1).all():
                best = min(best, float(problem.costs[columns].sum()))
    return best
