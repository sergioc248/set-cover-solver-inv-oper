from collections.abc import Callable
from pathlib import Path

import pytest

from set_cover.instance import Instance, load_instance

type InstanceFactory = Callable[[list[list[int]], list[float]], Instance]

# Clientes 1..4. La cobertura más barata son las antenas {2, 3} con costo 5.
SMALL_COVERAGE = [
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
]
SMALL_COSTS = [4.0, 2.0, 3.0, 4.0]


def write_coverage(path: Path, coverage: list[list[int]]) -> None:
    header = ",".join(str(j) for j in range(len(coverage[0])))
    rows = [",".join(str(value) for value in row) for row in coverage]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def write_costs(path: Path, costs: list[object]) -> None:
    rows = [f"{antenna},{cost}" for antenna, cost in enumerate(costs, start=1)]
    path.write_text("\n".join(["antenna_index_1based,cost", *rows]) + "\n", encoding="utf-8")


@pytest.fixture
def make_instance(tmp_path: Path) -> InstanceFactory:
    def factory(coverage: list[list[int]], costs: list[float]) -> Instance:
        write_coverage(tmp_path / "coverage.csv", coverage)
        write_costs(tmp_path / "costs.csv", list(costs))
        return load_instance(tmp_path / "coverage.csv", tmp_path / "costs.csv")

    return factory


@pytest.fixture
def small_instance(make_instance: InstanceFactory) -> Instance:
    return make_instance(SMALL_COVERAGE, SMALL_COSTS)
