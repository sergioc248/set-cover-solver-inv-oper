"""Calibración reproducible y reanudable de los hiperparámetros del algoritmo genético.

Etapas (ver ``docs/genetic_algorithm.md``):

- ``explore``: configuraciones muestreadas al azar del espacio inicial, pocas semillas.
- ``refine``: las mejores de ``explore`` y vecinas suyas, con más semillas nuevas.
- ``time``: las mejores de ``refine`` con un presupuesto de tiempo fijo, solo informativo.
- ``final``: la configuración seleccionada en ``refine`` con semillas nunca usadas.

Cada etapa escribe su plan (configuraciones y semillas) y agrega una línea por
ejecución terminada. Al reanudar, las ejecuciones ya registradas no se repiten.
"""

import csv
import io
import json
import multiprocessing
import os
import signal
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy

from set_cover.ga.config import CrossoverKind, GAConfig, MutationKind, PruneOrder
from set_cover.ga.engine import run_ga
from set_cover.ga.jsonio import (
    JsonLinesWriter,
    RecordError,
    bool_field,
    float_field,
    float_list_field,
    int_field,
    read_json_lines,
    str_field,
    to_mapping,
    write_json,
    write_text_atomic,
)
from set_cover.ga.problem import CoverProblem

type ParameterValue = int | float | str


class CalibrationError(RuntimeError):
    pass


class Stage(StrEnum):
    EXPLORE = "explore"
    REFINE = "refine"
    TIME = "time"
    FINAL = "final"


# Valores candidatos iniciales; ``prune_order`` compara los órdenes de eliminación de redundancias.
SEARCH_SPACE: dict[str, tuple[ParameterValue, ...]] = {
    "population_size": (50, 100, 200),
    "crossover_rate": (0.6, 0.8, 0.9),
    "mutation_rate": (0.002, 0.005, 0.01, 0.02, 0.03, 0.05),
    "mutation": tuple(kind.value for kind in MutationKind),
    "tournament_size": (2, 3, 5),
    "elite_count": (1, 2, 5),
    "crossover": (CrossoverKind.UNIFORM.value, CrossoverKind.ONE_POINT.value),
    "greedy_fraction": (0.25, 0.5, 0.75),
    "stall_generations": (100, 200, 300),
    "prune_order": (
        PruneOrder.COST_DESC.value,
        PruneOrder.RATIO_DESC.value,
        PruneOrder.RANDOM.value,
    ),
}

# Rejilla más fina para el refinamiento: los vecinos de un valor son sus adyacentes aquí.
# Contiene todos los valores de ``SEARCH_SPACE``.
REFINEMENT_GRID: dict[str, tuple[ParameterValue, ...]] = {
    "population_size": (30, 50, 75, 100, 150, 200, 300),
    "crossover_rate": (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
    "mutation_rate": (
        0.001,
        0.002,
        0.0035,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.04,
        0.05,
        0.065,
        0.08,
    ),
    "mutation": SEARCH_SPACE["mutation"],
    "tournament_size": (2, 3, 4, 5, 7),
    "elite_count": (1, 2, 3, 5, 8),
    "crossover": SEARCH_SPACE["crossover"],
    "greedy_fraction": (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
    "stall_generations": (50, 100, 150, 200, 250, 300, 400),
    "prune_order": SEARCH_SPACE["prune_order"],
}

_CATEGORICAL = frozenset({"crossover", "mutation", "prune_order"})

type SearchSpace = Mapping[str, Sequence[ParameterValue]]
# Avance de una etapa: (etapa, ejecuciones terminadas, total de la etapa).
type TrialProgress = Callable[["Stage", int, int], None]

_BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_SEED_BASE = {Stage.EXPLORE: 1000, Stage.REFINE: 2000, Stage.TIME: 3000, Stage.FINAL: 9000}
_MAX_SEEDS_PER_STAGE = 1000


@dataclass(frozen=True)
class CalibrationSettings:
    evaluation_budget: int = 100_000
    explore_configs: int = 40
    explore_seeds: int = 5
    refine_top: int = 6
    refine_neighbors: int = 4
    refine_seeds: int = 15
    time_budget_seconds: float = 60.0
    time_top: int = 3
    time_seeds: int = 10
    final_seeds: int = 30
    # Configuraciones cuya mediana está a esta fracción de la mejor se consideran similares.
    similarity_tolerance: float = 0.002
    sampling_seed: int = 20261109
    workers: int = 1
    search_space: SearchSpace = field(default_factory=lambda: SEARCH_SPACE)

    def __post_init__(self) -> None:
        seed_counts = {
            "explore_seeds": self.explore_seeds,
            "refine_seeds": self.refine_seeds,
            "time_seeds": self.time_seeds,
            "final_seeds": self.final_seeds,
        }
        for name, count in seed_counts.items():
            if not 1 <= count < _MAX_SEEDS_PER_STAGE:
                raise CalibrationError(f"{name} MUST be in [1, {_MAX_SEEDS_PER_STAGE})")
        if self.explore_configs < 1 or self.refine_top < 1 or self.time_top < 1:
            raise CalibrationError("configuration counts MUST be positive")
        if self.refine_neighbors < 0:
            raise CalibrationError("refine_neighbors MUST be non-negative")
        if self.workers < 1:
            raise CalibrationError("workers MUST be positive")
        if self.time_budget_seconds <= 0.0:
            raise CalibrationError("time_budget_seconds MUST be positive")
        if not 0.0 <= self.similarity_tolerance < 1.0:
            raise CalibrationError("similarity_tolerance MUST be in [0, 1)")
        validate_search_space(self.search_space)
        if self.explore_configs > space_size(self.search_space):
            raise CalibrationError(
                f"explore_configs MUST NOT exceed the {space_size(self.search_space)} "
                "combinations of the search space"
            )

    def base_config(self) -> GAConfig:
        """Criterios de parada comunes a las etapas con presupuesto de evaluaciones."""
        return GAConfig(max_evaluations=self.evaluation_budget, max_generations=None)


def settings_record(settings: CalibrationSettings) -> dict[str, object]:
    """Ajustes para ``settings.json``."""
    return {
        "evaluation_budget": settings.evaluation_budget,
        "explore_configs": settings.explore_configs,
        "explore_seeds": settings.explore_seeds,
        "refine_top": settings.refine_top,
        "refine_neighbors": settings.refine_neighbors,
        "refine_seeds": settings.refine_seeds,
        "time_budget_seconds": settings.time_budget_seconds,
        "time_top": settings.time_top,
        "time_seeds": settings.time_seeds,
        "final_seeds": settings.final_seeds,
        "similarity_tolerance": settings.similarity_tolerance,
        "sampling_seed": settings.sampling_seed,
        "workers": settings.workers,
        "search_space": {name: list(values) for name, values in settings.search_space.items()},
    }


def stage_seeds(stage: Stage, count: int) -> list[int]:
    """Semillas de una etapa. Los rangos no se solapan entre etapas."""
    base = _SEED_BASE[stage]
    return list(range(base + 1, base + 1 + count))


@dataclass(frozen=True)
class Plan:
    stage: Stage
    configs: tuple[GAConfig, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "seeds": list(self.seeds),
            "configs": [
                {"config_id": config.config_id(), "config": config.to_dict()}
                for config in self.configs
            ],
        }

    def trials(self) -> list[tuple[GAConfig, int]]:
        return [(config, seed) for config in self.configs for seed in self.seeds]


@dataclass(frozen=True)
class RunSummary:
    """Campos de un registro de ejecución que usa la calibración."""

    config_id: str
    seed: int
    best_cost: float
    feasible: bool
    num_antennas: int
    elapsed_seconds: float
    evaluations: int
    generations: int
    evaluations_to_best: int
    seconds_to_best: float

    @staticmethod
    def from_record(record: dict[str, object]) -> "RunSummary":
        where = "run record"
        return RunSummary(
            config_id=str_field(record, "config_id", where),
            seed=int_field(record, "seed", where),
            best_cost=float_field(record, "best_cost", where),
            feasible=bool_field(record, "feasible", where),
            num_antennas=int_field(record, "num_antennas", where),
            elapsed_seconds=float_field(record, "elapsed_seconds", where),
            evaluations=int_field(record, "evaluations", where),
            generations=int_field(record, "generations", where),
            evaluations_to_best=int_field(record, "evaluations_to_best", where),
            seconds_to_best=float_field(record, "seconds_to_best", where),
        )


@dataclass(frozen=True)
class ConfigSummary:
    config: GAConfig
    runs: int
    feasible_rate: float
    median_cost: float
    mean_cost: float
    std_cost: float
    best_cost: float
    worst_cost: float
    mean_seconds: float
    mean_evaluations: float
    mean_generations: float
    median_evaluations_to_best: float
    median_seconds_to_best: float

    @property
    def config_id(self) -> str:
        return self.config.config_id()


def summarize(config: GAConfig, runs: Sequence[RunSummary]) -> ConfigSummary:
    if not runs:
        raise CalibrationError(f"no runs for configuration {config.config_id()}")
    costs = [run.best_cost for run in runs]
    return ConfigSummary(
        config=config,
        runs=len(runs),
        feasible_rate=sum(run.feasible for run in runs) / len(runs),
        median_cost=statistics.median(costs),
        mean_cost=statistics.fmean(costs),
        std_cost=statistics.stdev(costs) if len(costs) > 1 else 0.0,
        best_cost=min(costs),
        worst_cost=max(costs),
        mean_seconds=statistics.fmean(run.elapsed_seconds for run in runs),
        mean_evaluations=statistics.fmean(run.evaluations for run in runs),
        mean_generations=statistics.fmean(run.generations for run in runs),
        median_evaluations_to_best=statistics.median(run.evaluations_to_best for run in runs),
        median_seconds_to_best=statistics.median(run.seconds_to_best for run in runs),
    )


def rank(summaries: Iterable[ConfigSummary], tolerance: float) -> list[ConfigSummary]:
    """Ordena de mejor a peor según los criterios de selección.

    1. Primero las configuraciones con 100 % de ejecuciones factibles.
    2. Entre ellas, las que tienen mediana a lo sumo ``tolerance`` por encima de la mejor
       mediana se consideran similares y se ordenan por desviación estándar, tiempo medio
       y evaluaciones hasta la mejor solución (estabilidad, eficiencia y convergencia).
    3. Las demás se ordenan por mediana y luego por los mismos desempates.
    """
    candidates = list(summaries)
    feasible = [summary for summary in candidates if summary.feasible_rate == 1.0]
    infeasible = [summary for summary in candidates if summary.feasible_rate < 1.0]
    if not feasible:
        return sorted(infeasible, key=lambda summary: (-summary.feasible_rate, summary.median_cost))
    threshold = min(summary.median_cost for summary in feasible) * (1.0 + tolerance)

    def tiebreak(summary: ConfigSummary) -> tuple[float, float, float, str]:
        return (
            summary.std_cost,
            summary.mean_seconds,
            summary.median_evaluations_to_best,
            summary.config_id,
        )

    similar = sorted((s for s in feasible if s.median_cost <= threshold), key=tiebreak)
    rest = sorted(
        (s for s in feasible if s.median_cost > threshold),
        key=lambda summary: (summary.median_cost, *tiebreak(summary)),
    )
    return similar + rest + sorted(infeasible, key=lambda summary: summary.median_cost)


def validate_search_space(space: SearchSpace) -> None:
    """Exige los parámetros de ``SEARCH_SPACE``, con valores de ``REFINEMENT_GRID``.

    Así el refinamiento siempre encuentra cada valor en su rejilla.
    """
    if set(space) != set(SEARCH_SPACE):
        raise CalibrationError(f"the search space MUST define exactly {sorted(SEARCH_SPACE)}")
    for name, values in space.items():
        if not values:
            raise CalibrationError(f"{name} MUST have at least one value")
        if len(set(values)) != len(values):
            raise CalibrationError(f"{name} has repeated values")
        outside = [value for value in values if value not in REFINEMENT_GRID[name]]
        if outside:
            raise CalibrationError(f"{name}: {outside} are not in the refinement grid")


def read_search_space(path: Path) -> dict[str, tuple[ParameterValue, ...]]:
    """Espacio de búsqueda desde JSON: ``{"parametro": [valor, ...], ...}``."""
    data = to_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    space: dict[str, tuple[ParameterValue, ...]] = {}
    for name, raw in data.items():
        if not isinstance(raw, list):
            raise CalibrationError(f"{path}: {name} MUST be a list of values")
        values: list[ParameterValue] = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise CalibrationError(f"{path}: {name} has an invalid value {value!r}")
            values.append(value)
        space[name] = tuple(values)
    validate_search_space(space)
    return space


def space_size(space: SearchSpace) -> int:
    return int(numpy.prod([len(values) for values in space.values()]))


def refinement_grid(space: SearchSpace) -> dict[str, tuple[ParameterValue, ...]]:
    """Rejilla de refinamiento limitada a ``space``.

    Un parámetro categórico solo toma los valores elegidos. Uno numérico toma los valores
    de ``REFINEMENT_GRID`` entre su mínimo y su máximo elegidos.
    """
    validate_search_space(space)
    grid: dict[str, tuple[ParameterValue, ...]] = {}
    for name, values in space.items():
        if name in _CATEGORICAL:
            grid[name] = tuple(value for value in REFINEMENT_GRID[name] if value in values)
            continue
        numbers = [value for value in values if not isinstance(value, str)]
        low, high = min(numbers), max(numbers)
        grid[name] = tuple(
            value
            for value in REFINEMENT_GRID[name]
            if not isinstance(value, str) and low <= value <= high
        )
    return grid


def sample_configurations(
    base: GAConfig,
    count: int,
    rng: numpy.random.Generator,
    space: SearchSpace = SEARCH_SPACE,
) -> list[GAConfig]:
    """``count`` combinaciones distintas de ``space``, muestreadas sin reemplazo."""
    names = list(space)
    sizes = [len(space[name]) for name in names]
    total = int(numpy.prod(sizes))
    if count > total:
        raise CalibrationError(f"the search space has only {total} combinations")
    configs: list[GAConfig] = []
    for flat in rng.choice(total, size=count, replace=False).tolist():
        changes: dict[str, object] = {}
        remainder = int(flat)
        for name, size in zip(names, sizes, strict=True):
            remainder, position = divmod(remainder, size)
            changes[name] = space[name][position]
        configs.append(base.replace(**changes))
    return configs


def neighbors(
    config: GAConfig, grids: Mapping[str, Sequence[ParameterValue]] = REFINEMENT_GRID
) -> list[GAConfig]:
    """Configuraciones que cambian un único parámetro a un valor adyacente de la rejilla."""
    current = config.to_dict()
    result: list[GAConfig] = []
    for name, grid in grids.items():
        value = current[name]
        if value not in grid:
            continue
        position = grid.index(value)
        if name in _CATEGORICAL:
            options = [option for option in grid if option != value]
        else:
            options = [
                grid[index] for index in (position - 1, position + 1) if 0 <= index < len(grid)
            ]
        for option in options:
            candidate = config.replace(**{name: option})
            if candidate.tournament_size <= candidate.population_size:
                result.append(candidate)
    return result


class Calibration:
    """Ejecuta y persiste las etapas en ``output_dir``."""

    def __init__(
        self,
        problem: CoverProblem,
        settings: CalibrationSettings,
        output_dir: Path,
        log: Callable[[str], None],
        on_trial: TrialProgress | None = None,
    ) -> None:
        self.problem = problem
        self.settings = settings
        self.output_dir = output_dir
        self.log = log
        # Un espacio distinto del original limita también el refinamiento a sus valores.
        space = settings.search_space
        self.grids: Mapping[str, Sequence[ParameterValue]] = (
            REFINEMENT_GRID if space is SEARCH_SPACE else refinement_grid(space)
        )
        self.on_trial = on_trial

    def run(self, stages: Sequence[Stage]) -> None:
        for stage in stages:
            self.log(f"== stage {stage.value}")
            plan = self._plan(stage)
            self._execute(plan)
            self._write_summary(plan)

    def plan_path(self, stage: Stage) -> Path:
        return self.output_dir / f"plan_{stage.value}.json"

    def runs_path(self, stage: Stage) -> Path:
        return self.output_dir / f"runs_{stage.value}.jsonl"

    def summary_path(self, stage: Stage) -> Path:
        return self.output_dir / f"summary_{stage.value}.csv"

    @property
    def selected_config_path(self) -> Path:
        return self.output_dir / "selected_config.json"

    def ranked(self, stage: Stage) -> list[ConfigSummary]:
        """Resumen ordenado de una etapa completa."""
        plan = self._load_plan(stage)
        if plan is None:
            raise CalibrationError(f"stage {stage.value} has not been planned yet")
        runs = self._completed(plan)
        missing = len(plan.trials()) - len(runs)
        if missing:
            raise CalibrationError(
                f"stage {stage.value} is missing {missing} run(s); run it again to finish"
            )
        by_config: dict[str, list[RunSummary]] = {}
        for run in runs.values():
            by_config.setdefault(run.config_id, []).append(run)
        return rank(
            (summarize(config, by_config[config.config_id()]) for config in plan.configs),
            self.settings.similarity_tolerance,
        )

    def selected_config(self) -> GAConfig:
        return self.ranked(Stage.REFINE)[0].config

    def _plan(self, stage: Stage) -> Plan:
        settings = self.settings
        if stage is Stage.EXPLORE:
            rng = numpy.random.default_rng(settings.sampling_seed)
            configs = sample_configurations(
                settings.base_config(), settings.explore_configs, rng, settings.search_space
            )
            seeds = stage_seeds(stage, settings.explore_seeds)
        elif stage is Stage.REFINE:
            configs = self._refinement_configs()
            seeds = stage_seeds(stage, settings.refine_seeds)
        elif stage is Stage.TIME:
            configs = [
                summary.config.replace(
                    max_evaluations=None,
                    stall_generations=None,
                    max_generations=None,
                    time_limit=settings.time_budget_seconds,
                )
                for summary in self.ranked(Stage.REFINE)[: settings.time_top]
            ]
            seeds = stage_seeds(stage, settings.time_seeds)
        else:
            selected = self.selected_config()
            write_json(self.selected_config_path, selected.to_dict())
            configs = [selected]
            seeds = stage_seeds(stage, settings.final_seeds)
        plan = Plan(stage=stage, configs=tuple(_unique(configs)), seeds=tuple(seeds))
        stored = self._load_plan(stage)
        if stored is None:
            write_json(self.plan_path(stage), plan.to_dict())
        elif stored.to_dict() != plan.to_dict():
            raise CalibrationError(
                f"{self.plan_path(stage)} was created with different settings or results; "
                "use another output directory to start over"
            )
        return plan

    def _refinement_configs(self) -> list[GAConfig]:
        settings = self.settings
        rng = numpy.random.default_rng(settings.sampling_seed + 1)
        top = [summary.config for summary in self.ranked(Stage.EXPLORE)[: settings.refine_top]]
        configs = list(top)
        known = {config.config_id() for config in configs}
        for config in top:
            options = [
                option
                for option in neighbors(config, self.grids)
                if option.config_id() not in known
            ]
            if not options:
                continue
            chosen = rng.choice(
                len(options), size=min(settings.refine_neighbors, len(options)), replace=False
            )
            for index in sorted(chosen.tolist()):
                configs.append(options[index])
                known.add(options[index].config_id())
        return configs

    def _load_plan(self, stage: Stage) -> Plan | None:
        path = self.plan_path(stage)
        if not path.exists():
            return None
        data = to_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
        configs_raw = data.get("configs")
        seeds_raw = data.get("seeds")
        if not isinstance(configs_raw, list) or not isinstance(seeds_raw, list):
            raise RecordError(f"{path}: configs and seeds MUST be lists")
        configs = tuple(
            GAConfig.from_dict(to_mapping(to_mapping(item, str(path)).get("config"), str(path)))
            for item in configs_raw
        )
        seeds: list[int] = []
        for seed in seeds_raw:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise RecordError(f"{path}: seeds MUST be integers")
            seeds.append(seed)
        return Plan(stage=stage, configs=configs, seeds=tuple(seeds))

    def _completed(self, plan: Plan) -> dict[tuple[str, int], RunSummary]:
        planned = {(config.config_id(), seed) for config, seed in plan.trials()}
        completed: dict[tuple[str, int], RunSummary] = {}
        for record in read_json_lines(self.runs_path(plan.stage)):
            run = RunSummary.from_record(record)
            key = (run.config_id, run.seed)
            if key in planned:
                completed[key] = run
        return completed

    def _execute(self, plan: Plan) -> None:
        done = self._completed(plan)
        pending = [
            (config, seed)
            for config, seed in plan.trials()
            if (config.config_id(), seed) not in done
        ]
        total = len(plan.trials())
        self.log(f"{len(done)}/{total} runs already recorded, {len(pending)} pending")
        if self.on_trial is not None:
            self.on_trial(plan.stage, len(done), total)
        if not pending:
            return
        writer = JsonLinesWriter(self.runs_path(plan.stage))
        # Cada ejecución es secuencial; BLAS multihilo en varios procesos solo compite por CPU.
        for variable in _BLAS_THREAD_VARIABLES:
            os.environ.setdefault(variable, "1")
        executor = ProcessPoolExecutor(
            max_workers=self.settings.workers,
            mp_context=multiprocessing.get_context("forkserver"),
            initializer=_initialize_worker,
            initargs=(self.problem,),
        )
        futures: dict[Future[dict[str, object]], tuple[str, int]] = {
            executor.submit(_run_trial, config.to_dict(), seed): (config.config_id(), seed)
            for config, seed in pending
        }
        saved: set[Future[dict[str, object]]] = set()
        finished = len(done)

        def save(future: Future[dict[str, object]]) -> None:
            nonlocal finished
            record = future.result()
            record["stage"] = plan.stage.value
            writer.append(record)
            saved.add(future)
            finished += 1
            run = RunSummary.from_record(record)
            self.log(
                f"[{finished}/{total}] {run.config_id} seed {run.seed}: cost {run.best_cost:,.0f} "
                f"in {run.elapsed_seconds:.1f}s"
            )
            if self.on_trial is not None:
                self.on_trial(plan.stage, finished, total)

        try:
            for future in as_completed(futures):
                save(future)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            self.log("interrupted: saving the runs already in progress, then stopping")
            for future in futures:
                if future not in saved and not future.cancelled():
                    save(future)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            writer.close()

    def _write_summary(self, plan: Plan) -> None:
        ranked = self.ranked(plan.stage)
        write_text_atomic(self.summary_path(plan.stage), summary_csv(ranked, self.settings))
        best = ranked[0]
        self.log(
            f"best {best.config_id}: median {best.median_cost:,.0f}, std {best.std_cost:,.1f}, "
            f"{best.mean_seconds:.1f}s per run"
        )
        write_text_atomic(
            self.output_dir / f"parameter_effects_{plan.stage.value}.csv",
            parameter_effects_csv(plan.configs, list(self._completed(plan).values())),
        )
        if plan.stage is Stage.FINAL:
            records = [
                record
                for record in read_json_lines(self.runs_path(Stage.FINAL))
                if str_field(record, "config_id", "run record") == best.config_id
            ]
            write_json(self.output_dir / "final_statistics.json", final_statistics(best, records))
            write_text_atomic(self.output_dir / "final_convergence.csv", convergence_csv(records))


_SUMMARY_COLUMNS = (
    "rank",
    "config_id",
    "similar_to_best",
    "runs",
    "feasible_rate",
    "median_cost",
    "mean_cost",
    "std_cost",
    "best_cost",
    "worst_cost",
    "mean_seconds",
    "mean_evaluations",
    "mean_generations",
    "median_evaluations_to_best",
    "median_seconds_to_best",
    *SEARCH_SPACE,
)


def summary_csv(ranked: Sequence[ConfigSummary], settings: CalibrationSettings) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_SUMMARY_COLUMNS)
    feasible_medians = [s.median_cost for s in ranked if s.feasible_rate == 1.0]
    threshold = min(feasible_medians, default=numpy.inf) * (1.0 + settings.similarity_tolerance)
    for position, summary in enumerate(ranked, start=1):
        parameters = summary.config.to_dict()
        writer.writerow(
            [
                position,
                summary.config_id,
                summary.feasible_rate == 1.0 and summary.median_cost <= threshold,
                summary.runs,
                f"{summary.feasible_rate:.3f}",
                f"{summary.median_cost:.1f}",
                f"{summary.mean_cost:.1f}",
                f"{summary.std_cost:.1f}",
                f"{summary.best_cost:.1f}",
                f"{summary.worst_cost:.1f}",
                f"{summary.mean_seconds:.3f}",
                f"{summary.mean_evaluations:.1f}",
                f"{summary.mean_generations:.1f}",
                f"{summary.median_evaluations_to_best:.1f}",
                f"{summary.median_seconds_to_best:.3f}",
                *(parameters[name] for name in SEARCH_SPACE),
            ]
        )
    return buffer.getvalue()


def final_statistics(
    summary: ConfigSummary, records: Sequence[dict[str, object]]
) -> dict[str, object]:
    runs = [RunSummary.from_record(record) for record in records]
    best_record = min(records, key=lambda record: float_field(record, "best_cost", "run record"))
    seconds = [run.elapsed_seconds for run in runs]
    antennas = [run.num_antennas for run in runs]
    return {
        "config_id": summary.config_id,
        "config": summary.config.to_dict(),
        "runs": summary.runs,
        "seeds": [run.seed for run in runs],
        "feasible_rate": summary.feasible_rate,
        "cost": {
            "mean": summary.mean_cost,
            "median": summary.median_cost,
            "best": summary.best_cost,
            "worst": summary.worst_cost,
            "std": summary.std_cost,
        },
        "seconds": {
            "mean": statistics.fmean(seconds),
            "median": statistics.median(seconds),
            "min": min(seconds),
            "max": max(seconds),
            "std": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        },
        "num_antennas": {"min": min(antennas), "max": max(antennas)},
        "median_evaluations_to_best": summary.median_evaluations_to_best,
        "median_seconds_to_best": summary.median_seconds_to_best,
        "best_run": {
            "seed": int_field(best_record, "seed", "run record"),
            "cost": float_field(best_record, "best_cost", "run record"),
            "antenna_ids": best_record.get("antenna_ids"),
            "covered_customers": best_record.get("covered_customers"),
        },
    }


def parameter_effects_csv(configs: Sequence[GAConfig], runs: Sequence[RunSummary]) -> str:
    """Efecto marginal de cada valor de cada parámetro sobre todas las ejecuciones de una etapa.

    ``reached_best_rate`` es la fracción de ejecuciones que alcanzó el menor costo de la etapa.
    Las configuraciones no son un diseño factorial, así que los efectos son descriptivos.
    """
    parameters = {config.config_id(): config.to_dict() for config in configs}
    lowest = min((run.best_cost for run in runs), default=numpy.inf)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        (
            "parameter",
            "value",
            "configs",
            "runs",
            "mean_cost",
            "median_cost",
            "best_cost",
            "reached_best_rate",
            "mean_seconds",
        )
    )
    for name in SEARCH_SPACE:
        values = sorted({str(values[name]) for values in parameters.values()}, key=_value_order)
        for value in values:
            ids = {key for key, values in parameters.items() if str(values[name]) == value}
            selected = [run for run in runs if run.config_id in ids]
            if not selected:
                continue
            costs = [run.best_cost for run in selected]
            writer.writerow(
                (
                    name,
                    value,
                    len(ids),
                    len(selected),
                    f"{statistics.fmean(costs):.1f}",
                    f"{statistics.median(costs):.1f}",
                    f"{min(costs):.1f}",
                    f"{sum(cost == lowest for cost in costs) / len(costs):.3f}",
                    f"{statistics.fmean(run.elapsed_seconds for run in selected):.3f}",
                )
            )
    return buffer.getvalue()


def _value_order(value: str) -> tuple[int, float, str]:
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value)


_HISTORY_COLUMNS = (
    "generation",
    "evaluations",
    "elapsed_seconds",
    "best_so_far",
    "population_best",
    "population_mean",
    "population_worst",
    "unique_individuals",
    "improved",
)


def convergence_csv(records: Sequence[dict[str, object]]) -> str:
    """Historial de convergencia en formato largo: una fila por semilla y generación."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("config_id", "seed", *_HISTORY_COLUMNS))
    for record in records:
        history = to_mapping(record.get("history"), "run record.history")
        columns = [float_list_field(history, name, "history") for name in _HISTORY_COLUMNS[:-1]]
        columns.append([float(flag) for flag in _bool_list(history, "improved")])
        config_id = str_field(record, "config_id", "run record")
        seed = int_field(record, "seed", "run record")
        for row in zip(*columns, strict=True):
            writer.writerow((config_id, seed, *(_compact(value) for value in row)))
    return buffer.getvalue()


def _bool_list(data: dict[str, object], key: str) -> list[bool]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(item, bool) for item in value):
        raise RecordError(f"history.{key} MUST be a list of booleans")
    return [item is True for item in value]


def _compact(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6f}"


def _unique(configs: Iterable[GAConfig]) -> list[GAConfig]:
    seen: set[str] = set()
    result: list[GAConfig] = []
    for config in configs:
        if config.config_id() not in seen:
            seen.add(config.config_id())
            result.append(config)
    return result


_worker_problem: list[CoverProblem] = []


def _initialize_worker(problem: CoverProblem) -> None:
    # El proceso principal atiende Ctrl+C y espera a que terminen las ejecuciones en curso.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_problem.append(problem)


def _run_trial(config: dict[str, object], seed: int) -> dict[str, object]:
    return run_ga(_worker_problem[0], GAConfig.from_dict(config), seed).to_record()


def default_workers() -> int:
    """La mitad de los CPU disponibles, para limitar la interferencia en los tiempos medidos."""
    return max(1, (os.process_cpu_count() or 2) // 2)
