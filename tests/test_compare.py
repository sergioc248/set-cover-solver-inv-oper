import pytest

from set_cover.checkpoint import Checkpoint, Incumbent
from set_cover.compare import Comparison, ComparisonError, MethodResult, solver_result
from set_cover.instance import Instance


def _result(method: str, cost: float, bound: float | None, optimal: bool) -> MethodResult:
    return MethodResult(
        method=method,
        cost=cost,
        antenna_ids=[2, 3],
        covered_customers=4,
        seconds=1.0,
        best_bound=bound,
        solver_gap=None,
        proven_optimal=optimal,
        note="",
    )


def test_gap_against_a_proven_optimum() -> None:
    comparison = Comparison([_result("GA", 110.0, None, False), _result("MIP", 100.0, 100.0, True)])
    rows = {row["method"]: row for row in comparison.rows()}

    assert rows["GA"]["gap_vs_optimum_pct"] == pytest.approx(10.0)
    assert rows["MIP"]["gap_vs_optimum_pct"] == 0.0


def test_without_optimality_reports_best_known_and_bound_separately() -> None:
    comparison = Comparison([_result("GA", 95.0, None, False), _result("MIP", 100.0, 80.0, False)])
    rows = {row["method"]: row for row in comparison.rows()}

    assert comparison.optimum is None
    assert comparison.best_known == 95.0
    assert rows["GA"]["gap_vs_optimum_pct"] is None
    assert rows["MIP"]["diff_vs_best_known_pct"] == pytest.approx(100 * 5 / 95)
    assert rows["GA"]["gap_upper_bound_pct"] == pytest.approx(100 * 15 / 80)
    assert "| GA |" in comparison.to_markdown()


def test_solver_result_verifies_the_incumbent(small_instance: Instance) -> None:
    good = Checkpoint("x", incumbent=Incumbent([2, 3], 5.0), best_bound=5.0)
    assert solver_result(small_instance, good, "MIP", "").proven_optimal

    with pytest.raises(ComparisonError, match="uncovered"):
        solver_result(small_instance, Checkpoint("x", incumbent=Incumbent([1], 4.0)), "MIP", "")
    with pytest.raises(ComparisonError, match="reported cost"):
        solver_result(small_instance, Checkpoint("x", incumbent=Incumbent([2, 3], 9.0)), "MIP", "")
