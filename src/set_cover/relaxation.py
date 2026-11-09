"""Relajación LP del set cover: ``x`` continua en [0, 1] en lugar de binaria.

Su óptimo es una cota inferior válida del MIP. Las antenas con valor fraccional
muestran dónde la relajación "hace trampa" respecto a una instalación real.
"""

import math
from dataclasses import dataclass

import gurobipy
from gurobipy import GRB

from set_cover.instance import Instance
from set_cover.model import add_set_cover

# Valores a esta distancia de 0 o 1 se consideran enteros.
_INTEGRALITY_TOLERANCE = 1e-6


class RelaxationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RelaxationResult:
    objective: float
    integer_bound: float
    values: dict[int, float]
    runtime_seconds: float
    iterations: float

    @property
    def at_one(self) -> list[int]:
        return [antenna for antenna, value in self.values.items() if _is_one(value)]

    @property
    def fractional(self) -> dict[int, float]:
        return {antenna: value for antenna, value in self.values.items() if not _is_one(value)}


def solve_relaxation(instance: Instance, threads: int) -> RelaxationResult:
    """Resuelve el LP; ``values`` contiene solo las antenas (base 1) con valor no nulo."""
    env = gurobipy.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    model = gurobipy.Model("weighted_set_cover_lp", env=env)
    try:
        antennas = add_set_cover(model, instance, GRB.CONTINUOUS)
        model.Params.Threads = threads
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RelaxationError(f"LP relaxation ended with Gurobi status {model.Status}")
        solution = {j: float(variable.X) for j, variable in antennas.items()}
        objective = float(model.ObjVal)
        return RelaxationResult(
            objective=objective,
            integer_bound=(
                float(math.ceil(objective - _INTEGRALITY_TOLERANCE))
                if instance.integral_costs
                else objective
            ),
            values={j: value for j, value in solution.items() if value > _INTEGRALITY_TOLERANCE},
            runtime_seconds=float(model.Runtime),
            iterations=float(model.IterCount),
        )
    finally:
        model.dispose()
        env.dispose()


def _is_one(value: float) -> bool:
    return abs(value - 1.0) <= _INTEGRALITY_TOLERANCE
