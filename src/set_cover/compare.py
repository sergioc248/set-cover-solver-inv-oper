"""Comparación del algoritmo genético con el solver exacto sobre la misma instancia.

Nunca alimenta al algoritmo genético con soluciones del solver: solo lee resultados
ya producidos por cada método y los verifica contra la matriz original.
"""

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

from set_cover.checkpoint import Checkpoint, read_checkpoint
from set_cover.ga.jsonio import (
    float_field,
    int_field,
    int_list_field,
    to_mapping,
    write_json,
    write_text_atomic,
)
from set_cover.instance import Instance


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class MethodResult:
    method: str
    cost: float
    antenna_ids: list[int]
    covered_customers: int
    seconds: float
    best_bound: float | None
    solver_gap: float | None
    proven_optimal: bool
    note: str

    @property
    def num_antennas(self) -> int:
        return len(self.antenna_ids)


def ga_results(instance: Instance, statistics_path: Path) -> list[MethodResult]:
    """Mejor y mediana de la evaluación final del algoritmo genético."""
    data = to_mapping(json.loads(statistics_path.read_text(encoding="utf-8")), str(statistics_path))
    cost = to_mapping(data.get("cost"), "cost")
    seconds = to_mapping(data.get("seconds"), "seconds")
    best_run = to_mapping(data.get("best_run"), "best_run")
    runs = int_field(data, "runs", "final statistics")
    antenna_ids = int_list_field(best_run, "antenna_ids", "best_run")
    best_cost = float_field(best_run, "cost", "best_run")
    _verify(instance, antenna_ids, best_cost, "genetic algorithm")
    covered = instance.num_customers - len(instance.uncovered_customers(antenna_ids))
    mean_seconds = float_field(seconds, "mean", "seconds")
    return [
        MethodResult(
            method="GA (best of runs)",
            cost=best_cost,
            antenna_ids=antenna_ids,
            covered_customers=covered,
            seconds=mean_seconds,
            best_bound=None,
            solver_gap=None,
            proven_optimal=False,
            note=f"best of {runs} seeds; seconds = mean per run",
        ),
        MethodResult(
            method="GA (median of runs)",
            cost=float_field(cost, "median", "cost"),
            antenna_ids=[],
            covered_customers=instance.num_customers,
            seconds=float_field(seconds, "median", "seconds"),
            best_bound=None,
            solver_gap=None,
            proven_optimal=False,
            note=f"median of {runs} seeds; every run is feasible",
        ),
    ]


def solver_result(
    instance: Instance, checkpoint: Checkpoint, method: str, note: str
) -> MethodResult:
    incumbent = checkpoint.incumbent
    if incumbent is None:
        raise ComparisonError(f"{method}: the checkpoint has no incumbent solution")
    _verify(instance, incumbent.antenna_ids, incumbent.cost, method)
    gap = checkpoint.gap()
    return MethodResult(
        method=method,
        cost=incumbent.cost,
        antenna_ids=incumbent.antenna_ids,
        covered_customers=instance.num_customers,
        seconds=checkpoint.total_runtime_seconds,
        best_bound=checkpoint.best_bound,
        solver_gap=gap,
        proven_optimal=gap == 0.0,
        note=note,
    )


def read_solver(instance: Instance, checkpoint_path: Path, method: str, note: str) -> MethodResult:
    return solver_result(instance, read_checkpoint(checkpoint_path), method, note)


@dataclass(frozen=True)
class Comparison:
    results: list[MethodResult]

    @property
    def best_bound(self) -> float | None:
        bounds = [r.best_bound for r in self.results if r.best_bound is not None]
        return max(bounds) if bounds else None

    @property
    def optimum(self) -> float | None:
        """Costo óptimo, solo si algún solver lo demostró."""
        proven = [r.cost for r in self.results if r.proven_optimal]
        return min(proven) if proven else None

    @property
    def best_known(self) -> float:
        return min(r.cost for r in self.results)

    def rows(self) -> list[dict[str, object]]:
        optimum = self.optimum
        bound = self.best_bound
        rows: list[dict[str, object]] = []
        for result in self.results:
            row: dict[str, object] = {
                "method": result.method,
                "cost": result.cost,
                "num_antennas": result.num_antennas or None,
                "covered_customers": result.covered_customers,
                "seconds": round(result.seconds, 3),
                "solver_best_bound": result.best_bound,
                "solver_gap_pct": None if result.solver_gap is None else 100 * result.solver_gap,
                "proven_optimal": result.proven_optimal,
                "diff_vs_best_known_pct": _percent(result.cost, self.best_known),
                # Gap_GA(%) solo existe con óptimo demostrado.
                "gap_vs_optimum_pct": None if optimum is None else _percent(result.cost, optimum),
                # Sin óptimo demostrado, (Z - cota) / cota acota por arriba al Gap(%) real.
                "gap_upper_bound_pct": None if bound is None else _percent(result.cost, bound),
                "note": result.note,
            }
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "best_known_cost": self.best_known,
            "best_bound": self.best_bound,
            "optimum_proven": self.optimum is not None,
            "optimum": self.optimum,
            "rows": self.rows(),
            "antenna_ids": {r.method: r.antenna_ids for r in self.results if r.antenna_ids},
        }

    def to_csv(self) -> str:
        rows = self.rows()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    def to_markdown(self) -> str:
        header = (
            "| Método | Costo | Antenas | Clientes cubiertos | Tiempo (s) | Cota inferior "
            "| Brecha solver | Dif. vs mejor conocida | Cota sup. de la brecha |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        lines = [
            f"| {row['method']} | {_number(row['cost'])} | {_number(row['num_antennas'])} "
            f"| {_number(row['covered_customers'])} | {_number(row['seconds'], 1)} "
            f"| {_number(row['solver_best_bound'])} | {_pct(row['solver_gap_pct'])} "
            f"| {_pct(row['diff_vs_best_known_pct'])} | {_pct(row['gap_upper_bound_pct'])} |"
            for row in self.rows()
        ]
        return header + "\n".join(lines) + "\n"

    def write(self, output_dir: Path) -> None:
        write_json(output_dir / "comparison.json", self.to_dict())
        write_text_atomic(output_dir / "comparison.csv", self.to_csv())
        write_text_atomic(output_dir / "comparison.md", self.to_markdown())


def _verify(instance: Instance, antenna_ids: list[int], cost: float, method: str) -> None:
    uncovered = instance.uncovered_customers(antenna_ids)
    if uncovered:
        raise ComparisonError(f"{method}: solution leaves customers uncovered: {uncovered}")
    actual = instance.cost_of(antenna_ids)
    if abs(actual - cost) > 1e-6 * max(1.0, abs(cost)):
        raise ComparisonError(f"{method}: reported cost {cost} but the antennas cost {actual}")


def _percent(value: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return 100.0 * (value - reference) / reference


def _number(value: object, digits: int = 0) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:,.{digits}f}"


def _pct(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.2f} %"
