import pytest

from set_cover.instance import Instance
from set_cover.relaxation import solve_relaxation
from tests.conftest import InstanceFactory

# Tres clientes y tres antenas que cubren cada par: el LP elige 0.5 de cada una.
TRIANGLE_COVERAGE = [
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1],
]


def test_triangle_relaxation_is_fractional(make_instance: InstanceFactory) -> None:
    result = solve_relaxation(make_instance(TRIANGLE_COVERAGE, [1.0, 1.0, 1.0]), threads=1)

    assert result.objective == pytest.approx(1.5)
    assert result.integer_bound == 2.0
    assert result.at_one == []
    assert result.fractional == pytest.approx({1: 0.5, 2: 0.5, 3: 0.5})


def test_relaxation_bounds_the_integer_optimum(small_instance: Instance) -> None:
    result = solve_relaxation(small_instance, threads=1)

    assert result.objective <= 5.0 + 1e-9
    assert result.integer_bound <= 5.0


def test_keeps_fractional_bound_for_fractional_costs(make_instance: InstanceFactory) -> None:
    result = solve_relaxation(make_instance(TRIANGLE_COVERAGE, [1.5, 1.5, 1.5]), threads=1)

    assert result.integer_bound == pytest.approx(2.25)
