import json
from pathlib import Path

import numpy
import pytest

from set_cover.ga.calibration import (
    REFINEMENT_GRID,
    SEARCH_SPACE,
    Calibration,
    CalibrationError,
    CalibrationSettings,
    ConfigSummary,
    Stage,
    neighbors,
    rank,
    read_search_space,
    refinement_grid,
    sample_configurations,
    space_size,
    stage_seeds,
    validate_search_space,
)
from set_cover.ga.config import GAConfig
from set_cover.ga.jsonio import read_json_lines
from tests.ga_support import random_problem


def _summary(
    population: int, median: float, std: float, seconds: float, feasible: float = 1.0
) -> ConfigSummary:
    return ConfigSummary(
        config=GAConfig(population_size=population),
        runs=5,
        feasible_rate=feasible,
        median_cost=median,
        mean_cost=median,
        std_cost=std,
        best_cost=median,
        worst_cost=median,
        mean_seconds=seconds,
        mean_evaluations=1000.0,
        mean_generations=10.0,
        median_evaluations_to_best=500.0,
        median_seconds_to_best=1.0,
    )


def test_ranking_follows_the_selection_criteria() -> None:
    cheapest_but_unstable = _summary(10, 1000.0, 50.0, 1.0)
    similar_and_stable = _summary(20, 1001.0, 0.0, 2.0)
    similar_stable_faster = _summary(30, 1001.5, 0.0, 1.5)
    clearly_worse = _summary(40, 1100.0, 0.0, 0.1)
    infeasible = _summary(50, 900.0, 0.0, 0.1, feasible=0.8)

    ranked = rank(
        [
            clearly_worse,
            infeasible,
            cheapest_but_unstable,
            similar_and_stable,
            similar_stable_faster,
        ],
        tolerance=0.002,
    )

    assert [s.config.population_size for s in ranked] == [30, 20, 10, 40, 50]


def test_without_tolerance_the_median_decides() -> None:
    ranked = rank([_summary(20, 1001.0, 0.0, 1.0), _summary(10, 1000.0, 50.0, 9.0)], 0.0)
    assert ranked[0].config.population_size == 10


def test_stage_seeds_never_overlap() -> None:
    seeds = [set(stage_seeds(stage, 999)) for stage in Stage]
    for index, first in enumerate(seeds):
        for second in seeds[index + 1 :]:
            assert not first & second


def test_sampled_configurations_are_distinct_and_inside_the_space() -> None:
    configs = sample_configurations(GAConfig(), 40, numpy.random.default_rng(0))

    assert len({config.config_id() for config in configs}) == 40
    for config in configs:
        values = config.to_dict()
        for name, options in SEARCH_SPACE.items():
            assert values[name] in options
            assert values[name] in REFINEMENT_GRID[name]


def test_neighbors_change_exactly_one_parameter() -> None:
    config = GAConfig(population_size=100, mutation_rate=0.005)
    original = config.to_dict()

    for neighbor in neighbors(config):
        changed = [k for k, v in neighbor.to_dict().items() if original[k] != v]
        assert len(changed) == 1


def _settings() -> CalibrationSettings:
    return CalibrationSettings(
        evaluation_budget=600,
        explore_configs=4,
        explore_seeds=2,
        refine_top=2,
        refine_neighbors=1,
        refine_seeds=2,
        time_budget_seconds=0.05,
        time_top=1,
        time_seeds=1,
        final_seeds=3,
        workers=1,
    )


def test_calibration_runs_all_stages_and_resumes(tmp_path: Path) -> None:
    problem = random_problem(0, customers=15, antennas=20, density=0.2)
    calibration = Calibration(problem, _settings(), tmp_path, lambda _: None)

    calibration.run([Stage.EXPLORE])
    first = read_json_lines(calibration.runs_path(Stage.EXPLORE))
    calibration.run([Stage.EXPLORE])

    assert len(first) == 8
    assert read_json_lines(calibration.runs_path(Stage.EXPLORE)) == first

    calibration.run([Stage.REFINE, Stage.TIME, Stage.FINAL])
    refined = calibration.ranked(Stage.REFINE)
    selected = json.loads(calibration.selected_config_path.read_text(encoding="utf-8"))
    statistics = json.loads((tmp_path / "final_statistics.json").read_text(encoding="utf-8"))

    assert selected == refined[0].config.to_dict()
    assert refined == rank(refined, _settings().similarity_tolerance)
    assert statistics["runs"] == 3
    assert statistics["feasible_rate"] == 1.0
    assert set(statistics["seeds"]).isdisjoint(stage_seeds(Stage.REFINE, 2))
    assert (
        (tmp_path / "final_convergence.csv")
        .read_text(encoding="utf-8")
        .startswith("config_id,seed,generation")
    )
    assert calibration.summary_path(Stage.TIME).exists()
    effects = (tmp_path / "parameter_effects_explore.csv").read_text(encoding="utf-8")
    assert effects.splitlines()[0].startswith("parameter,value,configs,runs")
    assert {line.split(",")[0] for line in effects.splitlines()[1:]} == set(SEARCH_SPACE)


def test_refuses_to_resume_with_different_settings(tmp_path: Path) -> None:
    problem = random_problem(0, customers=15, antennas=20, density=0.2)
    Calibration(problem, _settings(), tmp_path, lambda _: None).run([Stage.EXPLORE])
    changed = CalibrationSettings(
        evaluation_budget=700, explore_configs=4, explore_seeds=2, workers=1
    )

    with pytest.raises(CalibrationError, match="different settings"):
        Calibration(problem, changed, tmp_path, lambda _: None).run([Stage.EXPLORE])


def test_refinement_requires_a_complete_exploration(tmp_path: Path) -> None:
    problem = random_problem(0, customers=15, antennas=20, density=0.2)

    with pytest.raises(CalibrationError, match="not been planned"):
        Calibration(problem, _settings(), tmp_path, lambda _: None).run([Stage.REFINE])


def _narrow_space() -> dict[str, tuple[int | float | str, ...]]:
    space = {name: values[:1] for name, values in SEARCH_SPACE.items()}
    space["mutation"] = ("remove", "swap")
    space["mutation_rate"] = (0.01, 0.03)
    return space


def test_search_space_must_cover_every_parameter_with_grid_values() -> None:
    validate_search_space(SEARCH_SPACE)
    incomplete = {name: values for name, values in SEARCH_SPACE.items() if name != "mutation"}
    with pytest.raises(CalibrationError, match="exactly"):
        validate_search_space(incomplete)
    with pytest.raises(CalibrationError, match="at least one"):
        validate_search_space({**SEARCH_SPACE, "mutation": ()})
    with pytest.raises(CalibrationError, match="refinement grid"):
        validate_search_space({**SEARCH_SPACE, "population_size": (64,)})
    with pytest.raises(CalibrationError, match="combinations"):
        CalibrationSettings(explore_configs=5, search_space=_narrow_space())


def test_refinement_grid_stays_inside_the_chosen_values() -> None:
    grid = refinement_grid(_narrow_space())

    assert grid["mutation"] == ("remove", "swap")
    assert grid["mutation_rate"] == (0.01, 0.015, 0.02, 0.03)
    assert grid["population_size"] == (50,)
    for config in neighbors(GAConfig(mutation_rate=0.02, population_size=50), grid):
        assert config.mutation_rate in grid["mutation_rate"]
        assert config.population_size == 50


def test_sampling_uses_the_given_space() -> None:
    space = _narrow_space()
    configs = sample_configurations(
        GAConfig(), space_size(space), numpy.random.default_rng(0), space
    )

    assert len({config.config_id() for config in configs}) == 4
    assert {config.mutation.value for config in configs} == {"remove", "swap"}
    with pytest.raises(CalibrationError, match="only 4"):
        sample_configurations(GAConfig(), 5, numpy.random.default_rng(0), space)


def test_custom_calibration_reports_progress_and_stays_in_its_space(tmp_path: Path) -> None:
    problem = random_problem(0, customers=15, antennas=20, density=0.2)
    space = _narrow_space()
    settings = CalibrationSettings(
        evaluation_budget=600,
        explore_configs=3,
        explore_seeds=2,
        refine_top=2,
        refine_neighbors=2,
        refine_seeds=2,
        final_seeds=2,
        search_space=space,
    )
    progress: list[tuple[Stage, int, int]] = []

    def record(stage: Stage, finished: int, total: int) -> None:
        progress.append((stage, finished, total))

    calibration = Calibration(problem, settings, tmp_path, lambda _: None, record)
    calibration.run([Stage.EXPLORE, Stage.REFINE, Stage.FINAL])

    explore = [item for item in progress if item[0] is Stage.EXPLORE]
    assert explore[0] == (Stage.EXPLORE, 0, 6)
    assert explore[-1] == (Stage.EXPLORE, 6, 6)
    assert progress[-1][0] is Stage.FINAL
    assert progress[-1][1] == progress[-1][2] == 2
    allowed = refinement_grid(space)
    for summary in calibration.ranked(Stage.REFINE):
        assert summary.config.mutation.value in allowed["mutation"]
        assert summary.config.mutation_rate in allowed["mutation_rate"]


def test_reads_a_search_space_file(tmp_path: Path) -> None:
    path = tmp_path / "space.json"
    path.write_text(json.dumps(_narrow_space()), encoding="utf-8")

    assert read_search_space(path) == _narrow_space()
    path.write_text(json.dumps({**_narrow_space(), "mutation": "swap"}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="list"):
        read_search_space(path)
    path.write_text(json.dumps({**_narrow_space(), "elite_count": [True]}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="invalid"):
        read_search_space(path)
