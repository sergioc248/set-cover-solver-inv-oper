"""Interfaz de línea de comandos: ``set-cover solve``, ``status`` y ``lp``."""

import argparse
import json
import sys
from pathlib import Path

from gurobipy import GurobiError

from set_cover.checkpoint import Checkpoint, CheckpointError, read_checkpoint
from set_cover.instance import Instance, InstanceError, load_instance
from set_cover.relaxation import RelaxationError, RelaxationResult, solve_relaxation
from set_cover.run import RUNNING, RunDirectory, RunLockedError, checkpoint_path, is_locked
from set_cover.solver import ParameterError, SolveSettings, solve

DEFAULT_COVERAGE = Path("data/set_cover_500x500.csv")
DEFAULT_COSTS = Path("data/antenna_costs_500.csv")
DEFAULT_RUN_DIR = Path("runs/default")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "solve":
            return _solve(args)
        if args.command == "lp":
            return _lp(args)
        return _status(args)
    except (
        InstanceError,
        CheckpointError,
        RunLockedError,
        FileNotFoundError,
        GurobiError,
        ParameterError,
        RelaxationError,
    ):
        print(f"error: {sys.exception()}", file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="set-cover",
        description="Solver reanudable de Gurobi para la instancia de set cover ponderado.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    solve_parser = commands.add_parser(
        "solve",
        help="inicia o reanuda una sesión del solver",
        description="Inicia una sesión o reanuda el checkpoint del directorio de ejecución. "
        "Ctrl+C, SIGTERM o SIGHUP detienen la sesión limpiamente; vuelva a ejecutar "
        "para reanudar.",
    )
    _add_instance_arguments(solve_parser)
    solve_parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="directorio con el checkpoint y el log; reutilizarlo reanuda la ejecución",
    )
    solve_parser.add_argument(
        "--gap", type=_gap, default=0.01, help="brecha relativa objetivo; 0 prueba optimalidad"
    )
    solve_parser.add_argument(
        "--time-limit", type=_positive, default=None, help="límite de la sesión en segundos"
    )
    _add_threads_argument(solve_parser)
    solve_parser.add_argument("--seed", type=int, default=0, help="semilla aleatoria de Gurobi")
    solve_parser.add_argument(
        "--node-file-start",
        type=_positive,
        default=8.0,
        help="GB de memoria del árbol antes de que Gurobi escriba nodos al directorio de ejecución",
    )
    solve_parser.add_argument(
        "--checkpoint-interval",
        type=_positive,
        default=60.0,
        help="segundos entre checkpoints de la cota; las incumbentes se guardan de inmediato",
    )
    solve_parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=_gurobi_param,
        metavar="NOMBRE=VALOR",
        help="parámetro adicional de Gurobi, p. ej. MIPFocus=3; repetible",
    )

    status_parser = commands.add_parser(
        "status", help="muestra el checkpoint de un directorio de ejecución"
    )
    status_parser.add_argument(
        "--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="directorio de ejecución"
    )
    status_parser.add_argument("--json", action="store_true", help="imprime el checkpoint en JSON")

    lp_parser = commands.add_parser(
        "lp",
        help="resuelve la relajación LP (0 <= x <= 1)",
        description="Resuelve la relajación lineal, cuyo óptimo es una cota inferior del "
        "problema entero, y muestra qué antenas quedan fraccionales. No usa ni modifica "
        "ningún directorio de ejecución.",
    )
    _add_instance_arguments(lp_parser)
    _add_threads_argument(lp_parser)
    lp_parser.add_argument("--json", action="store_true", help="imprime el resultado en JSON")
    return parser


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--coverage", type=Path, default=DEFAULT_COVERAGE, help="matriz de cobertura (CSV)"
    )
    parser.add_argument(
        "--costs", type=Path, default=DEFAULT_COSTS, help="costos de las antenas (CSV)"
    )


def _add_threads_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threads", type=int, default=0, help="hilos de Gurobi; 0 deja que Gurobi decida"
    )


def _solve(args: argparse.Namespace) -> int:
    instance = load_instance(args.coverage, args.costs)
    print(_describe_instance(instance), flush=True)
    settings = SolveSettings(
        target_gap=args.gap,
        time_limit=args.time_limit,
        threads=args.threads,
        seed=args.seed,
        node_file_start_gb=args.node_file_start,
        checkpoint_interval=args.checkpoint_interval,
        gurobi_params=dict(args.param),
    )
    run = RunDirectory(args.run_dir, instance)
    try:
        if run.checkpoint.sessions:
            print(f"Resuming {args.run_dir}:", flush=True)
            print(_summary(run.checkpoint, running=False), flush=True)
        saved_gap = run.checkpoint.gap()
        if saved_gap is not None and saved_gap > args.gap:
            print(
                f"\nSaved gap: {saved_gap:.4%} across all sessions. Gurobi's log below "
                "reports this session only; its bound restarts from the LP relaxation.\n",
                flush=True,
            )
        session = solve(run, settings)
        if session is None:
            print(f"Checkpoint already meets the {args.gap:.4%} target gap; nothing to do.")
        else:
            print(f"\nSession ended: {session.status}")
        print(_summary(run.checkpoint, running=False))
    finally:
        run.close()
    return 0


def _lp(args: argparse.Namespace) -> int:
    instance = load_instance(args.coverage, args.costs)
    result = solve_relaxation(instance, args.threads)
    if args.json:
        print(
            json.dumps(
                {
                    "objective": result.objective,
                    "integer_bound": result.integer_bound,
                    "runtime_seconds": result.runtime_seconds,
                    "iterations": result.iterations,
                    "at_one": result.at_one,
                    "fractional": {str(k): v for k, v in result.fractional.items()},
                },
                indent=2,
            )
        )
    else:
        print(_describe_instance(instance))
        print(_lp_summary(result))
    return 0


def _describe_instance(instance: Instance) -> str:
    return (
        f"Instance: {instance.num_customers} customers, {instance.num_antennas} antennas, "
        f"{int(instance.coverage.sum()):,} coverage links"
    )


def _lp_summary(result: RelaxationResult) -> str:
    fractional = result.fractional
    lines = [
        f"  LP objective    : {result.objective:,.4f}",
        f"  integer bound   : {result.integer_bound:,.0f}",
        f"  antennas at 1   : {len(result.at_one)}",
        f"  fractional      : {len(fractional)}",
        f"  solve time      : {result.runtime_seconds:.3f}s, "
        f"{result.iterations:,.0f} simplex iterations",
        f"  at 1 (1-based)  : {json.dumps(result.at_one)}",
    ]
    if fractional:
        lines.append("  fractional (1-based antenna: value):")
        lines.extend(f"    {antenna:>4}: {value:.4f}" for antenna, value in fractional.items())
    return "\n".join(lines)


def _status(args: argparse.Namespace) -> int:
    path = checkpoint_path(args.run_dir)
    checkpoint = read_checkpoint(path)
    if args.json:
        print(checkpoint.to_json(), end="")
    else:
        print(_summary(checkpoint, running=is_locked(args.run_dir)))
    return 0


def _summary(checkpoint: Checkpoint, running: bool) -> str:
    incumbent = checkpoint.incumbent
    gap = checkpoint.gap()
    lines = [
        f"  incumbent cost : {'none' if incumbent is None else f'{incumbent.cost:,.0f}'}",
        f"  antennas       : {'none' if incumbent is None else len(incumbent.antenna_ids)}",
        f"  best bound     : "
        f"{'none' if checkpoint.best_bound is None else f'{checkpoint.best_bound:,.0f}'}",
        f"  gap            : {'unknown' if gap is None else f'{gap:.4%}'}",
        f"  solver time    : {_duration(checkpoint.total_runtime_seconds)} "
        f"over {len(checkpoint.sessions)} session(s)",
    ]
    if checkpoint.sessions:
        last = checkpoint.sessions[-1]
        status = last.status
        if status == RUNNING and not running:
            status = "aborted (process died)"
        lines.append(f"  last session   : {status}, started {last.started_at}")
    if incumbent is not None:
        lines.append(f"  selected (1-based): {json.dumps(incumbent.antenna_ids)}")
    return "\n".join(lines)


def _duration(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}h {whole % 3600 // 60:02d}m {whole % 60:02d}s"


def _gap(text: str) -> float:
    value = float(text)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError("gap MUST be in [0, 1)")
    return value


def _positive(text: str) -> float:
    value = float(text)
    if not value > 0.0:
        raise argparse.ArgumentTypeError("value MUST be positive")
    return value


def _gurobi_param(text: str) -> tuple[str, str]:
    name, separator, value = text.partition("=")
    if not separator or not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name.strip(), value.strip()


def run() -> None:
    sys.exit(main())
