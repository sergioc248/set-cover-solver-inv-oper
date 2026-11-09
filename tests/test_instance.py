from pathlib import Path

import pytest

from set_cover.instance import Instance, InstanceError, load_instance
from tests.conftest import SMALL_COSTS, SMALL_COVERAGE, write_costs, write_coverage


def test_loads_coverage_and_costs(small_instance: Instance) -> None:
    assert small_instance.num_customers == 4
    assert small_instance.num_antennas == 4
    assert small_instance.costs.tolist() == SMALL_COSTS
    assert small_instance.integral_costs


def test_cost_and_coverage_use_one_based_antenna_ids(small_instance: Instance) -> None:
    assert small_instance.cost_of([2, 3]) == 5.0
    assert small_instance.uncovered_customers([2, 3]) == []
    assert small_instance.uncovered_customers([1]) == [3, 4]


def test_fingerprint_changes_with_costs(small_instance: Instance, tmp_path: Path) -> None:
    write_costs(tmp_path / "other.csv", [4, 2, 3, 5])
    other = load_instance(tmp_path / "coverage.csv", tmp_path / "other.csv")
    assert small_instance.fingerprint() != other.fingerprint()


@pytest.mark.parametrize(
    ("coverage_text", "message"),
    [
        ("1,0\n1,0\n0,1\n", "header"),
        ("0,1\n1,2\n0,1\n", "0 or 1"),
        ("0,1\n1\n0,1\n", "columns"),
        ("0,1\n", "at least one customer"),
    ],
)
def test_rejects_malformed_coverage(tmp_path: Path, coverage_text: str, message: str) -> None:
    (tmp_path / "coverage.csv").write_text(coverage_text, encoding="utf-8")
    write_costs(tmp_path / "costs.csv", [1, 1])
    with pytest.raises(InstanceError, match=message):
        load_instance(tmp_path / "coverage.csv", tmp_path / "costs.csv")


@pytest.mark.parametrize(
    ("costs", "message"),
    [
        ([1, 2, 3], "3 costs for 4 antennas"),
        ([1, 2, "x", 4], "not a number"),
        ([1, 2, -3, 4], "invalid"),
    ],
)
def test_rejects_malformed_costs(tmp_path: Path, costs: list[object], message: str) -> None:
    write_coverage(tmp_path / "coverage.csv", SMALL_COVERAGE)
    write_costs(tmp_path / "costs.csv", costs)
    with pytest.raises(InstanceError, match=message):
        load_instance(tmp_path / "coverage.csv", tmp_path / "costs.csv")


@pytest.mark.parametrize(
    ("costs_text", "message"),
    [
        ("antenna,cost\n1,1\n", "header"),
        ("antenna_index_1based,cost\n2,1\n", "expected antenna id 1"),
        ("antenna_index_1based,cost\n1,1,1\n", "2 columns"),
        ("antenna_index_1based,cost\n1,nan\n", "invalid"),
    ],
)
def test_rejects_malformed_cost_rows(tmp_path: Path, costs_text: str, message: str) -> None:
    write_coverage(tmp_path / "coverage.csv", [[1]])
    (tmp_path / "costs.csv").write_text(costs_text, encoding="utf-8")
    with pytest.raises(InstanceError, match=message):
        load_instance(tmp_path / "coverage.csv", tmp_path / "costs.csv")


def test_rejects_customer_without_covering_antenna(tmp_path: Path) -> None:
    write_coverage(tmp_path / "coverage.csv", [[1, 0], [0, 0]])
    write_costs(tmp_path / "costs.csv", [1, 1])
    with pytest.raises(InstanceError, match=r"no covering antenna: \[2\]"):
        load_instance(tmp_path / "coverage.csv", tmp_path / "costs.csv")
