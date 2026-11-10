"""Ciclo evolutivo generacional con elitismo, reparación y registro de convergencia.

Todo individuo de la población es factible y no tiene antenas redundantes, así que
su fitness es exactamente la función objetivo ``sum_j c[j] x[j]``.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy
from numpy.typing import NDArray

from set_cover.ga.config import GAConfig
from set_cover.ga.operators import crossover, mutate, tournament
from set_cover.ga.problem import CoverProblem, Evaluation, Genes, evaluate
from set_cover.ga.repair import repair_and_prune

# Intentos seguidos de generar un individuo distinto antes de aceptar un duplicado. Evita un
# ciclo infinito en instancias con pocas soluciones irredundantes.
_MAX_DUPLICATE_RETRIES = 50
# Una mejora menor que esta no cuenta, para que el redondeo no reinicie el estancamiento.
_IMPROVEMENT_TOLERANCE = 1e-9


class StopReason(StrEnum):
    MAX_GENERATIONS = "max_generations"
    STALL = "stall"
    MAX_EVALUATIONS = "max_evaluations"
    TIME_LIMIT = "time_limit"


@dataclass
class History:
    """Estadísticas por generación, en columnas. La generación 0 es la población inicial."""

    generation: list[int] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    elapsed_seconds: list[float] = field(default_factory=list)
    best_so_far: list[float] = field(default_factory=list)
    population_best: list[float] = field(default_factory=list)
    population_mean: list[float] = field(default_factory=list)
    population_worst: list[float] = field(default_factory=list)
    unique_individuals: list[int] = field(default_factory=list)
    improved: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[int] | list[float] | list[bool]]:
        return {
            "generation": self.generation,
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "best_so_far": self.best_so_far,
            "population_best": self.population_best,
            "population_mean": self.population_mean,
            "population_worst": self.population_worst,
            "unique_individuals": self.unique_individuals,
            "improved": self.improved,
        }


@dataclass(frozen=True)
class GenerationReport:
    generation: int
    evaluations: int
    elapsed_seconds: float
    best_so_far: float
    population_mean: float


@dataclass(frozen=True)
class GAResult:
    seed: int
    config: GAConfig
    best_genes: Genes
    # Recalculada desde la matriz original, independiente del estado incremental.
    evaluation: Evaluation
    generations: int
    evaluations: int
    elapsed_seconds: float
    stop_reason: StopReason
    generation_of_best: int
    evaluations_to_best: int
    seconds_to_best: float
    history: History

    def to_record(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "config_id": self.config.config_id(),
            "config": self.config.to_dict(),
            "best_cost": self.evaluation.cost,
            "num_antennas": self.evaluation.num_antennas,
            "antenna_ids": list(self.evaluation.antenna_ids),
            "covered_customers": self.evaluation.covered,
            "uncovered_customers": list(self.evaluation.uncovered),
            "feasible": self.evaluation.feasible,
            "generations": self.generations,
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "stop_reason": self.stop_reason.value,
            "generation_of_best": self.generation_of_best,
            "evaluations_to_best": self.evaluations_to_best,
            "seconds_to_best": self.seconds_to_best,
            "history": self.history.to_dict(),
        }


type ProgressCallback = Callable[[GenerationReport], None]


def run_ga(
    problem: CoverProblem,
    config: GAConfig,
    seed: int,
    on_generation: ProgressCallback | None = None,
) -> GAResult:
    """Ejecuta el algoritmo y retorna la mejor solución factible encontrada.

    Con la misma instancia, configuración y semilla el resultado es idéntico, salvo
    cuando ``time_limit`` corta la ejecución: ese criterio depende del reloj.
    """
    return _Search(problem, config, seed, on_generation).run()


class _Search:
    def __init__(
        self,
        problem: CoverProblem,
        config: GAConfig,
        seed: int,
        on_generation: ProgressCallback | None,
    ) -> None:
        self.problem = problem
        self.config = config
        self.seed = seed
        self.rng = numpy.random.default_rng(seed)
        self.on_generation = on_generation
        self.started = time.perf_counter()
        self.evaluations = 0
        self.generation = 0
        self.best_genes = problem.empty_genes()
        self.best_cost = numpy.inf
        self.generation_of_best = 0
        self.evaluations_to_best = 0
        self.seconds_to_best = 0.0
        self.last_improvement = 0
        self.improved_this_generation = False
        self.history = History()

    def run(self) -> GAResult:
        population, fitness = self._initial_population()
        self._record(population, fitness)
        stop_reason = self._stop_reason()
        while stop_reason is None:
            self.generation += 1
            population, fitness = self._next_generation(population, fitness)
            self._record(population, fitness)
            stop_reason = self._stop_reason()
        evaluation = evaluate(self.problem, self.best_genes)
        if not evaluation.feasible:
            raise AssertionError(
                f"best solution leaves customers uncovered: {evaluation.uncovered}"
            )
        return GAResult(
            seed=self.seed,
            config=self.config,
            best_genes=self.best_genes.copy(),
            evaluation=evaluation,
            generations=self.generation,
            evaluations=self.evaluations,
            elapsed_seconds=self._elapsed(),
            stop_reason=stop_reason,
            generation_of_best=self.generation_of_best,
            evaluations_to_best=self.evaluations_to_best,
            seconds_to_best=self.seconds_to_best,
            history=self.history,
        )

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _budget_exhausted(self) -> bool:
        limit = self.config.max_evaluations
        if limit is not None and self.evaluations >= limit:
            return True
        return self.config.time_limit is not None and self._elapsed() >= self.config.time_limit

    def _stop_reason(self) -> StopReason | None:
        config = self.config
        if config.max_evaluations is not None and self.evaluations >= config.max_evaluations:
            return StopReason.MAX_EVALUATIONS
        if config.time_limit is not None and self._elapsed() >= config.time_limit:
            return StopReason.TIME_LIMIT
        if config.max_generations is not None and self.generation >= config.max_generations:
            return StopReason.MAX_GENERATIONS
        if (
            config.stall_generations is not None
            and self.generation - self.last_improvement >= config.stall_generations
        ):
            return StopReason.STALL
        return None

    def _finish(self, genes: Genes, rcl_alpha: float = 0.0) -> float:
        """Repara, elimina redundancias, cuenta la evaluación y actualiza la mejor global."""
        cost = repair_and_prune(self.problem, genes, self.config.prune_order, self.rng, rcl_alpha)
        self.evaluations += 1
        if cost < self.best_cost - _IMPROVEMENT_TOLERANCE:
            self.best_cost = cost
            self.best_genes = genes.copy()
            self.generation_of_best = self.generation
            self.evaluations_to_best = self.evaluations
            self.seconds_to_best = self._elapsed()
            self.last_improvement = self.generation
            self.improved_this_generation = True
        return cost

    def _initial_population(self) -> tuple[NDArray[numpy.bool_], NDArray[numpy.float64]]:
        """Greedy puro, greedy aleatorizado y aleatorios reparados, sin duplicados si se puede.

        El primer individuo greedy usa el greedy puro; los demás greedy usan la lista
        restringida de candidatos con ``rcl_alpha``.
        """
        config = self.config
        size = config.population_size
        greedy_count = round(config.greedy_fraction * size)
        population = numpy.zeros((size, self.problem.num_antennas), dtype=numpy.bool_)
        fitness = numpy.zeros(size, dtype=numpy.float64)
        seen: set[bytes] = set()
        for index in range(size):
            for attempt in range(_MAX_DUPLICATE_RETRIES):
                if index < greedy_count:
                    genes = self.problem.empty_genes()
                    alpha = 0.0 if index == 0 else config.rcl_alpha
                else:
                    genes = self.rng.random(self.problem.num_antennas) < config.random_activation
                    alpha = 0.0
                cost = self._finish(genes, alpha)
                key = genes.tobytes()
                if key not in seen or attempt == _MAX_DUPLICATE_RETRIES - 1:
                    break
            seen.add(key)
            population[index] = genes
            fitness[index] = cost
        return population, fitness

    def _next_generation(
        self, population: NDArray[numpy.bool_], fitness: NDArray[numpy.float64]
    ) -> tuple[NDArray[numpy.bool_], NDArray[numpy.float64]]:
        config = self.config
        size = config.population_size
        ranking = numpy.argsort(fitness, kind="stable")
        # Copias: los operadores trabajan sobre arreglos nuevos y la élite no se toca.
        next_population = population[ranking].copy()
        next_fitness = fitness[ranking].copy()
        filled = config.elite_count
        seen = {next_population[index].tobytes() for index in range(filled)}
        retries = 0
        while filled < size and not self._budget_exhausted():
            first = population[tournament(fitness, config.tournament_size, self.rng)]
            second = population[tournament(fitness, config.tournament_size, self.rng)]
            if self.rng.random() < config.crossover_rate:
                children = crossover(config.crossover, first, second, self.rng)
            else:
                children = (first.copy(), second.copy())
            for child in children:
                if filled >= size or self._budget_exhausted():
                    break
                mutate(config.mutation, child, config.mutation_rate, self.rng)
                cost = self._finish(child)
                key = child.tobytes()
                if config.deduplicate and key in seen and retries < _MAX_DUPLICATE_RETRIES:
                    retries += 1
                    continue
                retries = 0
                seen.add(key)
                next_population[filled] = child
                next_fitness[filled] = cost
                filled += 1
        if filled < size:
            # El presupuesto se agotó a mitad de generación: los cupos restantes conservan a los
            # mejores no élite de la generación anterior, que ya están evaluados.
            survivors = ranking[config.elite_count : config.elite_count + size - filled]
            next_population[filled:] = population[survivors]
            next_fitness[filled:] = fitness[survivors]
        return next_population, next_fitness

    def _record(self, population: NDArray[numpy.bool_], fitness: NDArray[numpy.float64]) -> None:
        history = self.history
        elapsed = self._elapsed()
        history.generation.append(self.generation)
        history.evaluations.append(self.evaluations)
        history.elapsed_seconds.append(elapsed)
        history.best_so_far.append(float(self.best_cost))
        history.population_best.append(float(fitness.min()))
        history.population_mean.append(float(fitness.mean()))
        history.population_worst.append(float(fitness.max()))
        history.unique_individuals.append(len({row.tobytes() for row in population}))
        history.improved.append(self.improved_this_generation)
        self.improved_this_generation = False
        if self.on_generation is not None:
            self.on_generation(
                GenerationReport(
                    generation=self.generation,
                    evaluations=self.evaluations,
                    elapsed_seconds=elapsed,
                    best_so_far=float(self.best_cost),
                    population_mean=float(fitness.mean()),
                )
            )
