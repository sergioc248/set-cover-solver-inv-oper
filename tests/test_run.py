from pathlib import Path

import pytest

from set_cover.checkpoint import Checkpoint, CheckpointError, Incumbent, read_checkpoint
from set_cover.instance import Instance
from set_cover.run import (
    ABORTED,
    RUNNING,
    RunDirectory,
    RunLockedError,
    checkpoint_path,
    is_locked,
)
from tests.conftest import InstanceFactory


def test_checkpoint_round_trips(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    run.offer_incumbent([3, 2])
    run.offer_bound(4.2)
    run.begin_session(0.01, {"threads": "2"})
    run.close()

    stored = read_checkpoint(checkpoint_path(tmp_path / "run"))
    assert stored == run.checkpoint
    assert stored == Checkpoint.from_json(stored.to_json())


def test_keeps_only_cheaper_incumbents(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    assert run.offer_incumbent([1, 2, 3, 4])
    assert run.offer_incumbent([2, 3])
    assert not run.offer_incumbent([1, 4])
    assert run.checkpoint.incumbent is not None
    assert run.checkpoint.incumbent.antenna_ids == [2, 3]
    run.close()


def test_rejects_non_covering_incumbent(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    with pytest.raises(ValueError, match="uncovered"):
        run.offer_incumbent([1])
    run.close()


def test_rounds_bounds_up_for_integral_costs(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    assert run.offer_bound(4.2) == 5.0
    assert run.offer_bound(3.9999999) == 4.0
    assert run.checkpoint.best_bound == 5.0
    assert run.offer_bound(float("-inf")) is None
    run.close()


def test_keeps_fractional_bounds_for_fractional_costs(
    make_instance: InstanceFactory, tmp_path: Path
) -> None:
    instance = make_instance([[1]], [1.5])
    run = RunDirectory(tmp_path / "run", instance)
    assert run.offer_bound(1.2) == 1.2
    run.close()


def test_run_directory_is_exclusive(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    assert is_locked(tmp_path / "run")
    with pytest.raises(RunLockedError):
        RunDirectory(tmp_path / "run", small_instance)
    run.close()
    assert not is_locked(tmp_path / "run")


def test_marks_sessions_of_dead_processes_aborted(small_instance: Instance, tmp_path: Path) -> None:
    run = RunDirectory(tmp_path / "run", small_instance)
    run.begin_session(0.01, {})
    run.close()
    assert read_checkpoint(checkpoint_path(tmp_path / "run")).sessions[0].status == RUNNING

    reopened = RunDirectory(tmp_path / "run", small_instance)
    assert reopened.checkpoint.sessions[0].status == ABORTED
    reopened.close()


def test_refuses_checkpoint_of_another_instance(
    make_instance: InstanceFactory, small_instance: Instance, tmp_path: Path
) -> None:
    RunDirectory(tmp_path / "run", small_instance).close()
    other = make_instance([[1, 1], [1, 0]], [1, 1])
    with pytest.raises(CheckpointError, match="different instance"):
        RunDirectory(tmp_path / "run", other)
    assert not is_locked(tmp_path / "run")


def test_rejects_unknown_schema_version() -> None:
    with pytest.raises(CheckpointError, match="schema version"):
        Checkpoint.from_json('{"schema_version": 99}')


def test_gap_matches_gurobi_definition() -> None:
    checkpoint = Checkpoint(instance_fingerprint="x", best_bound=90.0)
    assert checkpoint.gap() is None
    checkpoint.incumbent = Incumbent(antenna_ids=[1], cost=100.0)
    assert checkpoint.gap() == pytest.approx(0.1)
