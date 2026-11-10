from pathlib import Path

import pytest

from set_cover.trajectory import (
    TrajectoryError,
    TrajectoryPoint,
    parse_log,
    read_trajectory,
    write_trajectory,
)

# Dos sesiones: la segunda reinicia su reloj y su cota (desde la LP) al reanudar.
_LOG = """\
Gurobi Optimizer version 13.0.3 build v13.0.3rc0 (linux64 - "Arch Linux")
Presolve time: 0.01s
    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 27859.9266    0  199          - 27859.9266      -     -    0s
H    0     0                    86788.000000 27859.9266  67.9%     -    0s
     0     0 28000.5000    0  190 86788.0000 28000.5000  67.7%     -    1s
H12908  9721                    50678.000000 30648.6191  39.5%   121   26s

Explored 13000 nodes (900 simplex iterations) in 30.00 seconds (5.0 work units)
Best objective 5.067800000000e+04, best bound 3.064900000000e+04, gap 39.5226%
Gurobi Optimizer version 13.0.3 build v13.0.3rc0 (linux64 - "Arch Linux")
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time
     0     0 27859.9266    0  199 50678.0000 27859.9266  45.0%     -    0s
 340925887 246385427 48860.4505   84   69 49017.0000 31000.0000  36.8%   106   100s
"""


def test_parses_sessions_on_one_cumulative_clock() -> None:
    points = parse_log(_LOG)

    assert points[0] == TrajectoryPoint(0.0, None, 27859.9266)
    assert points[1] == TrajectoryPoint(0.0, 86788.0, 27859.9266)
    assert TrajectoryPoint(26.0, 50678.0, 30648.6191) in points
    # La segunda sesión empieza en 30 s, donde terminó la primera según "Explored".
    assert points[-1] == TrajectoryPoint(130.0, 49017.0, 31000.0)


def test_bound_never_decreases_and_incumbent_never_increases() -> None:
    points = parse_log(_LOG)

    bounds = [point.bound for point in points]
    incumbents = [point.incumbent for point in points if point.incumbent is not None]
    assert bounds == sorted(bounds)
    assert incumbents == sorted(incumbents, reverse=True)
    # El reinicio de la cota a 27.859 en la segunda sesión no se refleja en la trayectoria.
    assert all(point.bound >= 30648.6191 for point in points if point.seconds >= 26.0)


def test_ignores_lines_that_are_not_progress() -> None:
    assert (
        parse_log("Presolve time: 0.01s\nRoot relaxation: objective 2.78e+04, 0.05 seconds\n") == []
    )
    assert parse_log("") == []


def test_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    points = parse_log(_LOG)

    write_trajectory(path, points)

    assert read_trajectory(path) == points


def test_rejects_malformed_csv(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("time,cost\n1,2\n", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="header"):
        read_trajectory(path)
    path.write_text("seconds,incumbent,bound\n1,x,2\n", encoding="utf-8")
    with pytest.raises(TrajectoryError, match="numbers"):
        read_trajectory(path)


def test_saved_trajectories_match_their_checkpoints() -> None:
    root = Path("results/ga/comparison/gurobi")
    for name, incumbent in (("4s", 52554.0), ("112s", 50048.0), ("92h", 49017.0)):
        points = read_trajectory(root / name / "trajectory.csv")
        assert points[-1].incumbent == incumbent
