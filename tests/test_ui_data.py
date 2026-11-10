from pathlib import Path

import pytest

from set_cover.checkpoint import Checkpoint, Incumbent, write_checkpoint
from set_cover.ga.config import GAConfig
from set_cover.ga.engine import run_ga
from set_cover.ga.jsonio import RecordError
from set_cover.ga.problem import CoverProblem
from set_cover.trajectory import TrajectoryPoint, write_trajectory
from set_cover.ui.data import (
    Curve,
    GARunRecord,
    discover_exact_runs,
    duration,
    exact_method_result,
    ga_record_result,
    ga_session_result,
    instance_stats,
    jaccard,
    load_exact_run,
    median_curve,
    median_series,
    read_convergence,
    read_ga_records,
)
from tests.ga_support import real_instance

_SAVED = Path("results/ga/comparison/gurobi")


def test_instance_stats_match_the_verified_data() -> None:
    stats = instance_stats(real_instance())

    assert (stats.customers, stats.antennas, stats.links) == (500, 500, 24_971)
    assert (stats.cost.minimum, stats.cost.maximum) == (2000.0, 3998.0)
    assert stats.total_cost == 1_516_821.0
    assert (stats.customers_per_antenna.minimum, stats.customers_per_antenna.maximum) == (33, 66)
    assert (stats.antennas_per_customer.minimum, stats.antennas_per_customer.maximum) == (32, 70)


def _run_dir(tmp_path: Path, name: str, bound: float | None, ids: list[int] | None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    instance = real_instance()
    incumbent = None if ids is None else Incumbent(ids, instance.cost_of(ids))
    write_checkpoint(
        directory / "checkpoint.json",
        Checkpoint(instance.fingerprint(), incumbent=incumbent, best_bound=bound),
    )
    return directory


def test_discovers_only_directories_with_a_checkpoint(tmp_path: Path) -> None:
    _run_dir(tmp_path, "b", None, None)
    _run_dir(tmp_path, "a", None, None)
    (tmp_path / "empty").mkdir()

    found = discover_exact_runs([tmp_path, tmp_path / "missing"])

    assert [path.name for path in found] == ["a", "b"]


def test_exact_run_ends_at_the_checkpoint_state(tmp_path: Path) -> None:
    ids = list(range(1, 501))
    directory = _run_dir(tmp_path, "run", 30000.0, ids)
    write_trajectory(
        directory / "trajectory.csv",
        [TrajectoryPoint(0.0, None, 27000.0), TrajectoryPoint(5.0, 60000.0, 29999.5)],
    )

    run = load_exact_run(directory)

    assert not run.running
    assert run.trajectory[-1].bound == 30000.0
    assert run.trajectory[-1].incumbent == real_instance().cost_of(ids)
    assert run.trajectory[-1].seconds == 5.0
    result = exact_method_result(real_instance(), run)
    assert result is not None
    assert result.covered_customers == 500
    assert result.best_bound == 30000.0


def test_exact_run_without_incumbent_has_no_comparison_row(tmp_path: Path) -> None:
    run = load_exact_run(_run_dir(tmp_path, "run", 28000.0, None))
    assert exact_method_result(real_instance(), run) is None


def test_saved_solver_runs_load_with_their_trajectories() -> None:
    runs = [load_exact_run(directory) for directory in discover_exact_runs([_SAVED])]

    assert {run.name for run in runs} == {"4s", "112s", "92h"}
    longest = next(run for run in runs if run.name == "92h")
    assert longest.checkpoint.best_bound == 38875.0
    assert longest.trajectory[-1].bound == 38875.0
    assert len(longest.trajectory) > 100


def _record(seed: int, cost: float, ids: list[int]) -> GARunRecord:
    return GARunRecord(seed, "id", cost, len(ids), ids, 1.0, 10, 1, 0.5, "stall")


def test_ga_record_result_picks_the_cheapest_feasible_run() -> None:
    instance = real_instance()
    everything = list(range(1, 501))
    records = [
        _record(1, instance.cost_of(everything), everything),
        _record(2, 10.0, everything),
    ]

    result = ga_record_result(instance, records, "GA")

    assert result.cost == 10.0
    assert "semilla 2" in result.note
    with pytest.raises(RecordError, match="uncovered"):
        ga_record_result(instance, [_record(3, 1.0, [1])], "GA")
    with pytest.raises(RecordError, match="no runs"):
        ga_record_result(instance, [], "GA")


def test_ga_session_result_reports_the_run() -> None:
    instance = real_instance()
    config = GAConfig(population_size=10, max_generations=2, stall_generations=None)
    result = run_ga(CoverProblem.from_instance(instance), config, 1)

    row = ga_session_result(instance, result, "GA sesión")

    assert row.cost == result.evaluation.cost
    assert row.covered_customers == 500
    assert row.best_bound is None


def test_reads_the_saved_final_evaluation() -> None:
    records = read_ga_records(Path("results/ga/calibration/runs_final.jsonl"))
    curves = read_convergence(Path("results/ga/calibration/final_convergence.csv"))

    assert len(records) == 30
    assert min(record.cost for record in records) == 48538.0
    assert len(curves) == 30
    assert all(curve.best == sorted(curve.best, reverse=True) for curve in curves)


def test_median_curve_is_stepwise_and_needs_every_curve() -> None:
    curves = [
        Curve("a", [1.0, 3.0], [10.0, 6.0]),
        Curve("b", [2.0, 4.0], [8.0, 4.0]),
        Curve("c", [1.5], [9.0]),
    ]

    assert median_curve(curves, [0.5, 1.5, 2.0, 3.5, 10.0]) == [None, None, 9.0, 8.0, 6.0]
    grid, values = median_series(curves, points=5)
    assert grid[0] == 1.0
    assert grid[-1] == 4.0
    assert values[-1] == 6.0
    assert median_series([], points=5) == ([], [])


def test_jaccard_and_duration() -> None:
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    assert jaccard([], []) == 1.0
    assert duration(4.0) == "4,0 s"
    assert duration(112.0) == "1 min 52 s"
    assert duration(333_822.0) == "92 h 43 min"
