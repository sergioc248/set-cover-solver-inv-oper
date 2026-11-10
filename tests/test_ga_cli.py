import json
from pathlib import Path

import pytest

from set_cover.ga.cli import main
from set_cover.instance import Instance


def test_run_prints_and_saves_a_feasible_solution(
    small_instance: Instance, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    del small_instance  # escribe coverage.csv y costs.csv en tmp_path
    output = tmp_path / "result.json"
    code = main(
        [
            "run",
            "--coverage",
            str(tmp_path / "coverage.csv"),
            "--costs",
            str(tmp_path / "costs.csv"),
            "--population-size",
            "6",
            "--tournament-size",
            "2",
            "--max-generations",
            "5",
            "--output",
            str(output),
            "--convergence",
            str(tmp_path / "convergence.csv"),
            "--json",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert code == 0
    assert printed["best_cost"] == 5.0
    assert printed["antenna_ids"] == [2, 3]
    assert json.loads(output.read_text(encoding="utf-8"))["feasible"]
    assert (tmp_path / "convergence.csv").exists()


def test_reports_configuration_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"population_size": 1}', encoding="utf-8")

    assert main(["run", "--config", str(config)]) == 1
    assert "population_size" in capsys.readouterr().err
