from pathlib import Path

from set_cover.instance import Instance
from set_cover.run import RunDirectory
from set_cover.solver import OPTIMAL, SolveSettings, solve


def _settings(target_gap: float) -> SolveSettings:
    return SolveSettings(
        target_gap=target_gap,
        time_limit=30.0,
        threads=1,
        seed=0,
        node_file_start_gb=1.0,
        checkpoint_interval=1.0,
        gurobi_params={"OutputFlag": "0"},
    )


def test_proves_optimal_cover(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    session = solve(run, _settings(0.0))
    run.close()

    assert session is not None
    assert session.status == OPTIMAL
    assert session.ended_at is not None
    assert run.checkpoint.incumbent is not None
    assert run.checkpoint.incumbent.antenna_ids == [2, 3]
    assert run.checkpoint.incumbent.cost == 5.0
    assert run.checkpoint.best_bound == 5.0


def test_skips_solving_when_checkpoint_meets_target(
    small_instance: Instance, tmp_path: Path
) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    run.offer_incumbent([2, 3])
    run.offer_bound(4.6)
    assert solve(run, _settings(0.0)) is None
    run.close()


def test_resumes_from_checkpointed_incumbent(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    run.offer_incumbent([1, 2, 3, 4])
    run.offer_bound(4.0)
    run.close()

    resumed = RunDirectory(tmp_path / "run", small_instance)
    session = solve(resumed, _settings(0.0))
    resumed.close()

    assert session is not None
    assert session.status == OPTIMAL
    assert resumed.checkpoint.incumbent is not None
    assert resumed.checkpoint.incumbent.cost == 5.0
    assert resumed.checkpoint.best_bound == 5.0
    assert len(resumed.checkpoint.sessions) == 1
