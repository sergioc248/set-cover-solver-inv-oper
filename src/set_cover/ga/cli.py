"""Interfaz de línea de comandos del algoritmo genético: ``set-cover-ga``."""

import argparse
import json
import sys
from pathlib import Path

from set_cover.checkpoint import CheckpointError
from set_cover.compare import Comparison, ComparisonError, ga_results, read_solver
from set_cover.ga.benchmark import run_benchmark
from set_cover.ga.calibration import (
    SEARCH_SPACE,
    Calibration,
    CalibrationError,
    CalibrationSettings,
    Stage,
    convergence_csv,
    default_workers,
    read_search_space,
    settings_record,
)
from set_cover.ga.config import ConfigError, CrossoverKind, GAConfig, MutationKind, PruneOrder
from set_cover.ga.engine import GAResult, GenerationReport, run_ga
from set_cover.ga.jsonio import RecordError, write_json, write_text_atomic
from set_cover.ga.problem import CoverProblem, UncoverableError
from set_cover.instance import Instance, InstanceError, load_instance

DEFAULT_COVERAGE = Path("data/set_cover_500x500.csv")
DEFAULT_COSTS = Path("data/antenna_costs_500.csv")
DEFAULT_CALIBRATION_DIR = Path("results/ga/calibration")
DEFAULT_COMPARISON_DIR = Path("results/ga/comparison")
DEFAULT_CHECKPOINT = Path("runs/default/checkpoint.json")

# Parámetros de GAConfig que ``run`` acepta como opción; sobrescriben al archivo --config.
_OVERRIDES: dict[str, type] = {
    "population_size": int,
    "crossover_rate": float,
    "mutation_rate": float,
    "tournament_size": int,
    "elite_count": int,
    "greedy_fraction": float,
    "rcl_alpha": float,
    "random_activation": float,
    "stall_generations": int,
    "max_generations": int,
    "max_evaluations": int,
    "time_limit": float,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "calibrate":
            return _calibrate(args)
        if args.command == "compare":
            return _compare(args)
        return _benchmark(args)
    except (
        InstanceError,
        UncoverableError,
        ConfigError,
        CalibrationError,
        ComparisonError,
        CheckpointError,
        RecordError,
        FileNotFoundError,
        json.JSONDecodeError,
    ):
        print(f"error: {sys.exception()}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted; completed runs are saved", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="set-cover-ga", description="Algoritmo genético para el set cover ponderado."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="ejecuta el algoritmo genético una vez")
    _add_instance_arguments(run_parser)
    run_parser.add_argument("--config", type=Path, help="configuración JSON (GAConfig)")
    run_parser.add_argument("--seed", type=int, default=0, help="semilla aleatoria")
    for name, kind in _OVERRIDES.items():
        run_parser.add_argument(f"--{name.replace('_', '-')}", type=kind, default=None)
    run_parser.add_argument(
        "--crossover", choices=[kind.value for kind in CrossoverKind], default=None
    )
    run_parser.add_argument(
        "--mutation", choices=[kind.value for kind in MutationKind], default=None
    )
    run_parser.add_argument(
        "--prune-order", choices=[order.value for order in PruneOrder], default=None
    )
    run_parser.add_argument(
        "--no-limit",
        action="append",
        default=[],
        choices=["stall_generations", "max_generations", "max_evaluations", "time_limit"],
        help="desactiva un criterio de parada de la configuración; repetible",
    )
    run_parser.add_argument("--output", type=Path, help="guarda el resultado completo en JSON")
    run_parser.add_argument(
        "--convergence", type=Path, help="guarda el historial de convergencia en CSV"
    )
    run_parser.add_argument(
        "--progress", type=int, default=0, metavar="N", help="informa cada N generaciones"
    )
    run_parser.add_argument("--json", action="store_true", help="imprime el resultado en JSON")

    calibrate_parser = commands.add_parser(
        "calibrate",
        help="calibra los hiperparámetros por etapas; reanudable",
        description="Ejecuta las etapas explore, refine, time y final. Ctrl+C guarda las "
        "ejecuciones terminadas; vuelva a ejecutar el mismo comando para continuar.",
    )
    _add_instance_arguments(calibrate_parser)
    defaults = CalibrationSettings()
    calibrate_parser.add_argument(
        "--stage",
        action="append",
        choices=[stage.value for stage in Stage],
        help="etapa a ejecutar; repetible. Por defecto, todas en orden",
    )
    calibrate_parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_DIR)
    calibrate_parser.add_argument("--workers", type=int, default=default_workers())
    calibrate_parser.add_argument("--budget", type=int, default=defaults.evaluation_budget)
    calibrate_parser.add_argument("--explore-configs", type=int, default=defaults.explore_configs)
    calibrate_parser.add_argument("--explore-seeds", type=int, default=defaults.explore_seeds)
    calibrate_parser.add_argument("--refine-top", type=int, default=defaults.refine_top)
    calibrate_parser.add_argument("--refine-neighbors", type=int, default=defaults.refine_neighbors)
    calibrate_parser.add_argument("--refine-seeds", type=int, default=defaults.refine_seeds)
    calibrate_parser.add_argument("--time-budget", type=float, default=defaults.time_budget_seconds)
    calibrate_parser.add_argument("--time-top", type=int, default=defaults.time_top)
    calibrate_parser.add_argument("--time-seeds", type=int, default=defaults.time_seeds)
    calibrate_parser.add_argument("--final-seeds", type=int, default=defaults.final_seeds)
    calibrate_parser.add_argument("--similarity", type=float, default=defaults.similarity_tolerance)
    calibrate_parser.add_argument(
        "--search-space",
        type=Path,
        help="JSON con la lista de valores a explorar de cada parámetro; por defecto, el "
        "espacio original",
    )
    calibrate_parser.add_argument("--sampling-seed", type=int, default=defaults.sampling_seed)

    compare_parser = commands.add_parser(
        "compare", help="compara la evaluación final con el solver exacto"
    )
    _add_instance_arguments(compare_parser)
    compare_parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "final_statistics.json",
        help="final_statistics.json de la calibración",
    )
    compare_parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        help="checkpoint.json del solver exacto; repetible. Use ETIQUETA=RUTA para nombrarlo",
    )
    compare_parser.add_argument("--output", type=Path, default=DEFAULT_COMPARISON_DIR)

    benchmark_parser = commands.add_parser("benchmark", help="mide las operaciones del algoritmo")
    _add_instance_arguments(benchmark_parser)
    benchmark_parser.add_argument("--repeat", type=int, default=2000)
    benchmark_parser.add_argument("--seed", type=int, default=0)
    return parser


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)


def _load(args: argparse.Namespace) -> tuple[Instance, CoverProblem]:
    instance = load_instance(args.coverage, args.costs)
    return instance, CoverProblem.from_instance(instance)


def _run_config(args: argparse.Namespace) -> GAConfig:
    base = GAConfig() if args.config is None else GAConfig.from_json(args.config.read_text("utf-8"))
    changes: dict[str, object] = {}
    for name in _OVERRIDES:
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.crossover is not None:
        changes["crossover"] = args.crossover
    if args.mutation is not None:
        changes["mutation"] = args.mutation
    if args.prune_order is not None:
        changes["prune_order"] = args.prune_order
    for name in args.no_limit:
        changes[name] = None
    return base.replace(**changes)


def _run(args: argparse.Namespace) -> int:
    instance, problem = _load(args)
    config = _run_config(args)

    def report(progress: GenerationReport) -> None:
        if args.progress > 0 and progress.generation % args.progress == 0:
            print(
                f"gen {progress.generation:>6}  evals {progress.evaluations:>9,}  "
                f"best {progress.best_so_far:>10,.0f}  mean {progress.population_mean:>10,.1f}  "
                f"{progress.elapsed_seconds:7.2f}s",
                file=sys.stderr,
                flush=True,
            )

    result = run_ga(problem, config, args.seed, report)
    # Verificación independiente con el cargador de la instancia, no con el estado del GA.
    if instance.uncovered_customers(list(result.evaluation.antenna_ids)):
        raise AssertionError("reported solution is not feasible")
    record = result.to_record()
    if args.output is not None:
        write_json(args.output, record)
    if args.convergence is not None:
        write_text_atomic(args.convergence, convergence_csv([record]))
    if args.json:
        print(json.dumps({key: value for key, value in record.items() if key != "history"}))
    else:
        print(_describe(result, instance))
    return 0


def _describe(result: GAResult, instance: Instance) -> str:
    evaluation = result.evaluation
    return "\n".join(
        [
            f"  best cost        : {evaluation.cost:,.0f}",
            f"  antennas         : {evaluation.num_antennas}",
            f"  covered          : {evaluation.covered}/{instance.num_customers} "
            f"({'feasible' if evaluation.feasible else 'INFEASIBLE'})",
            f"  time             : {result.elapsed_seconds:.2f}s",
            f"  generations      : {result.generations}",
            f"  evaluations      : {result.evaluations:,}",
            f"  stop reason      : {result.stop_reason.value}",
            f"  best found at    : generation {result.generation_of_best}, "
            f"evaluation {result.evaluations_to_best:,}, {result.seconds_to_best:.2f}s",
            f"  seed             : {result.seed}",
            f"  config           : {json.dumps(result.config.to_dict())}",
            f"  selected (1-based): {json.dumps(list(evaluation.antenna_ids))}",
        ]
    )


def _calibrate(args: argparse.Namespace) -> int:
    _, problem = _load(args)
    settings = CalibrationSettings(
        evaluation_budget=args.budget,
        explore_configs=args.explore_configs,
        explore_seeds=args.explore_seeds,
        refine_top=args.refine_top,
        refine_neighbors=args.refine_neighbors,
        refine_seeds=args.refine_seeds,
        time_budget_seconds=args.time_budget,
        time_top=args.time_top,
        time_seeds=args.time_seeds,
        final_seeds=args.final_seeds,
        similarity_tolerance=args.similarity,
        sampling_seed=args.sampling_seed,
        workers=args.workers,
        search_space=SEARCH_SPACE
        if args.search_space is None
        else read_search_space(args.search_space),
    )
    stages = [Stage(value) for value in args.stage] if args.stage else list(Stage)
    write_json(args.output / "settings.json", settings_record(settings))
    Calibration(problem, settings, args.output, _log).run(stages)
    return 0


def _compare(args: argparse.Namespace) -> int:
    instance = load_instance(args.coverage, args.costs)
    results = ga_results(instance, args.statistics)
    for entry in args.checkpoint or [str(DEFAULT_CHECKPOINT)]:
        label, separator, path = entry.partition("=")
        if not separator:
            label, path = "Gurobi", entry
        results.append(read_solver(instance, Path(path), label, f"checkpoint {path}"))
    comparison = Comparison(results)
    comparison.write(args.output)
    print(comparison.to_markdown(), end="")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    _, problem = _load(args)
    for measurement in run_benchmark(problem, args.repeat, args.seed):
        print(f"  {measurement.name:<45} {measurement.microseconds:9.1f} us")
    return 0


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run() -> None:
    sys.exit(main())
