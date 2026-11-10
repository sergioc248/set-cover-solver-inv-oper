"""Hiperparámetros del algoritmo genético y su forma serializable."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class ConfigError(ValueError):
    pass


class CrossoverKind(StrEnum):
    UNIFORM = "uniform"
    ONE_POINT = "one_point"


class MutationKind(StrEnum):
    """Operador de mutación. ``mixed`` elige uno de los otros cuatro al azar por descendiente."""

    BIT_FLIP = "bit_flip"
    ADD = "add"
    REMOVE = "remove"
    SWAP = "swap"
    MIXED = "mixed"


class PruneOrder(StrEnum):
    """Orden en que se intenta retirar cada antena seleccionada."""

    COST_DESC = "cost_desc"
    RATIO_DESC = "ratio_desc"
    RANDOM = "random"


@dataclass(frozen=True)
class GAConfig:
    """Configuración completa de una ejecución. Ver ``docs/genetic_algorithm.md``.

    Los criterios de parada son opcionales individualmente, pero DEBE haber al menos uno.
    """

    population_size: int = 100
    crossover_rate: float = 0.8
    # En bit_flip, probabilidad por gen. En add/remove/swap, el número de movimientos sigue
    # Binomial(num_antennas, mutation_rate): el mismo valor esperado que bit_flip.
    mutation_rate: float = 0.005
    mutation: MutationKind = MutationKind.BIT_FLIP
    tournament_size: int = 3
    elite_count: int = 2
    crossover: CrossoverKind = CrossoverKind.UNIFORM
    greedy_fraction: float = 0.5
    # Lista restringida de candidatos del greedy aleatorizado: 0 = greedy puro, 1 = aleatorio.
    rcl_alpha: float = 0.3
    # Probabilidad de activar cada gen en un individuo inicial aleatorio, antes de repararlo.
    random_activation: float = 0.02
    prune_order: PruneOrder = PruneOrder.COST_DESC
    # Descarta descendientes idénticos a un individuo ya presente en la nueva generación.
    deduplicate: bool = True
    stall_generations: int | None = 200
    max_generations: int | None = 2000
    max_evaluations: int | None = None
    time_limit: float | None = None

    def __post_init__(self) -> None:
        _require(self.population_size >= 2, "population_size MUST be at least 2")
        _require(0.0 <= self.crossover_rate <= 1.0, "crossover_rate MUST be in [0, 1]")
        _require(0.0 <= self.mutation_rate <= 1.0, "mutation_rate MUST be in [0, 1]")
        _require(
            1 <= self.tournament_size <= self.population_size,
            "tournament_size MUST be in [1, population_size]",
        )
        _require(
            0 <= self.elite_count < self.population_size,
            "elite_count MUST be in [0, population_size)",
        )
        _require(0.0 <= self.greedy_fraction <= 1.0, "greedy_fraction MUST be in [0, 1]")
        _require(0.0 <= self.rcl_alpha <= 1.0, "rcl_alpha MUST be in [0, 1]")
        _require(0.0 <= self.random_activation <= 1.0, "random_activation MUST be in [0, 1]")
        _require(
            self.stall_generations is None or self.stall_generations >= 1,
            "stall_generations MUST be positive",
        )
        _require(
            self.max_generations is None or self.max_generations >= 0,
            "max_generations MUST be non-negative",
        )
        _require(
            self.max_evaluations is None or self.max_evaluations >= self.population_size,
            "max_evaluations MUST be at least population_size",
        )
        _require(self.time_limit is None or self.time_limit > 0.0, "time_limit MUST be positive")
        _require(
            any(
                limit is not None
                for limit in (
                    self.stall_generations,
                    self.max_generations,
                    self.max_evaluations,
                    self.time_limit,
                )
            ),
            "at least one stopping criterion MUST be set",
        )

    def replace(self, **changes: object) -> "GAConfig":
        return GAConfig.from_dict({**self.to_dict(), **changes})

    def to_dict(self) -> dict[str, object]:
        return {
            "population_size": self.population_size,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "mutation": self.mutation.value,
            "tournament_size": self.tournament_size,
            "elite_count": self.elite_count,
            "crossover": self.crossover.value,
            "greedy_fraction": self.greedy_fraction,
            "rcl_alpha": self.rcl_alpha,
            "random_activation": self.random_activation,
            "prune_order": self.prune_order.value,
            "deduplicate": self.deduplicate,
            "stall_generations": self.stall_generations,
            "max_generations": self.max_generations,
            "max_evaluations": self.max_evaluations,
            "time_limit": self.time_limit,
        }

    def config_id(self) -> str:
        """Identificador estable derivado de todos los parámetros."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def from_dict(data: Mapping[str, object]) -> "GAConfig":
        """Construye una configuración; las claves ausentes toman el valor por defecto."""
        unknown = sorted(set(data) - _FIELD_NAMES)
        if unknown:
            raise ConfigError(f"unknown configuration keys: {unknown}")
        default = GAConfig()
        return GAConfig(
            population_size=_int(data, "population_size", default.population_size),
            crossover_rate=_float(data, "crossover_rate", default.crossover_rate),
            mutation_rate=_float(data, "mutation_rate", default.mutation_rate),
            mutation=_mutation(data.get("mutation", default.mutation.value)),
            tournament_size=_int(data, "tournament_size", default.tournament_size),
            elite_count=_int(data, "elite_count", default.elite_count),
            crossover=_crossover(data.get("crossover", default.crossover.value)),
            greedy_fraction=_float(data, "greedy_fraction", default.greedy_fraction),
            rcl_alpha=_float(data, "rcl_alpha", default.rcl_alpha),
            random_activation=_float(data, "random_activation", default.random_activation),
            prune_order=_prune_order(data.get("prune_order", default.prune_order.value)),
            deduplicate=_bool(data, "deduplicate", default.deduplicate),
            stall_generations=_optional_int(data, "stall_generations", default.stall_generations),
            max_generations=_optional_int(data, "max_generations", default.max_generations),
            max_evaluations=_optional_int(data, "max_evaluations", default.max_evaluations),
            time_limit=_optional_float(data, "time_limit", default.time_limit),
        )

    @staticmethod
    def from_json(text: str) -> "GAConfig":
        return GAConfig.from_dict(_mapping(json.loads(text)))


_FIELD_NAMES = frozenset(field.name for field in dataclasses.fields(GAConfig))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError("configuration MUST be a JSON object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError("configuration keys MUST be strings")
        result[key] = item
    return result


def _int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} MUST be an integer")
    return value


def _optional_int(data: Mapping[str, object], key: str, default: int | None) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} MUST be an integer or null")
    return value


def _float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} MUST be a number")
    return float(value)


def _optional_float(data: Mapping[str, object], key: str, default: float | None) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} MUST be a number or null")
    return float(value)


def _bool(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} MUST be a boolean")
    return value


def _crossover(value: object) -> CrossoverKind:
    for kind in CrossoverKind:
        if value == kind.value:
            return kind
    raise ConfigError(f"crossover MUST be one of {[kind.value for kind in CrossoverKind]}")


def _mutation(value: object) -> MutationKind:
    for kind in MutationKind:
        if value == kind.value:
            return kind
    raise ConfigError(f"mutation MUST be one of {[kind.value for kind in MutationKind]}")


def _prune_order(value: object) -> PruneOrder:
    for order in PruneOrder:
        if value == order.value:
            return order
    raise ConfigError(f"prune_order MUST be one of {[order.value for order in PruneOrder]}")
