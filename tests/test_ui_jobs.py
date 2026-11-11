import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from set_cover.ga.calibration import CalibrationSettings, Stage
from set_cover.ui.launcher import APP, command
from set_cover.ui.solver_jobs import (
    PID_NAME,
    SEARCH_SPACE_NAME,
    SolverJobError,
    SolverJobs,
    SolverRequest,
    calibration_command,
    read_output,
    solver_command,
)

# Proceso que termina limpio con SIGTERM, como el solver al recibirlo.
_SLEEPER = [
    sys.executable,
    "-c",
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
    "print('ready', flush=True)\n"
    "time.sleep(30)\n",
]


def test_solver_command_runs_the_existing_cli(tmp_path: Path) -> None:
    request = SolverRequest(
        run_dir=tmp_path / "run",
        coverage=Path("c.csv"),
        costs=Path("k.csv"),
        target_gap=0.0,
        time_limit=60.0,
        threads=4,
        seed=7,
    )

    command_line = solver_command(request)

    assert command_line[:4] == [sys.executable, "-m", "set_cover", "solve"]
    joined = " ".join(command_line)
    assert f"--run-dir {tmp_path / 'run'}" in joined
    assert "--gap 0.0" in joined
    assert "--time-limit 60.0" in joined
    assert "--threads 4" in joined
    assert "--seed 7" in joined
    unlimited = solver_command(SolverRequest(tmp_path, Path("c"), Path("k"), 0.01, None, 0, 0))
    assert "--time-limit" not in unlimited


def _wait(condition: Callable[[], bool], seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_start_monitor_and_stop_a_session(tmp_path: Path) -> None:
    jobs = SolverJobs()
    run_dir = tmp_path / "run"

    jobs.start(run_dir, _SLEEPER)
    _wait(lambda: "ready" in read_output(run_dir))

    assert jobs.running(run_dir)
    assert (run_dir / PID_NAME).read_text(encoding="utf-8").strip().isdigit()
    with pytest.raises(SolverJobError, match="already running"):
        jobs.start(run_dir, _SLEEPER)
    assert jobs.stop(run_dir)
    _wait(lambda: not jobs.running(run_dir))
    assert jobs.exit_code(run_dir) == 0
    assert not jobs.stop(run_dir)


def test_reports_a_failed_session(tmp_path: Path) -> None:
    jobs = SolverJobs()
    run_dir = tmp_path / "run"

    jobs.start(run_dir, [sys.executable, "-c", "import sys; print('no license'); sys.exit(3)"])
    _wait(lambda: not jobs.running(run_dir))

    assert jobs.exit_code(run_dir) == 3
    assert "no license" in read_output(run_dir)
    assert jobs.exit_code(tmp_path / "unknown") is None


def test_launcher_runs_the_app_with_extra_arguments() -> None:
    assert APP.is_file()
    assert command(["--server.port", "8600"]) == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        "8600",
    ]


_INTERRUPTIBLE = [
    sys.executable,
    "-c",
    "import sys, time\n"
    "print('ready', flush=True)\n"
    "try:\n"
    "    time.sleep(30)\n"
    "except KeyboardInterrupt:\n"
    "    print('saved', flush=True)\n"
    "    sys.exit(130)\n",
]


def test_stop_sends_the_signal_chosen_at_start(tmp_path: Path) -> None:
    jobs = SolverJobs()
    run_dir = tmp_path / "calibration"

    jobs.start(run_dir, _INTERRUPTIBLE, signal.SIGINT)
    _wait(lambda: "ready" in read_output(run_dir))
    assert jobs.stop(run_dir)
    _wait(lambda: not jobs.running(run_dir))

    assert jobs.exit_code(run_dir) == 130
    assert "saved" in read_output(run_dir)


def test_calibration_command_runs_the_ga_cli(tmp_path: Path) -> None:
    settings = CalibrationSettings(explore_configs=7, refine_seeds=3, workers=2)

    line = calibration_command(
        tmp_path, settings, Path("c.csv"), Path("k.csv"), [Stage.EXPLORE, Stage.FINAL]
    )

    assert line[:4] == [sys.executable, "-m", "set_cover.ga", "calibrate"]
    joined = " ".join(line)
    assert f"--output {tmp_path}" in joined
    assert f"--search-space {tmp_path / SEARCH_SPACE_NAME}" in joined
    assert "--explore-configs 7" in joined
    assert "--refine-seeds 3" in joined
    assert "--workers 2" in joined
    assert "--stage explore --stage final" in joined
