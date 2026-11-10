import pytest

from set_cover.ga.config import ConfigError, CrossoverKind, GAConfig, MutationKind


def test_round_trips_through_json() -> None:
    config = GAConfig(
        population_size=50,
        crossover=CrossoverKind.ONE_POINT,
        mutation=MutationKind.SWAP,
        time_limit=3.0,
    )

    restored = GAConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.config_id() == config.config_id()
    assert GAConfig().config_id() != config.config_id()


def test_missing_keys_take_defaults_and_none_disables_a_limit() -> None:
    config = GAConfig.from_dict({"stall_generations": None})

    assert config.population_size == GAConfig().population_size
    assert config.stall_generations is None


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"population_size": 1}, "population_size"),
        ({"crossover_rate": 1.5}, "crossover_rate"),
        ({"elite_count": 100}, "elite_count"),
        ({"tournament_size": 101}, "tournament_size"),
        ({"crossover": "two_point"}, "crossover MUST be one of"),
        ({"mutation": "inversion"}, "mutation MUST be one of"),
        ({"population_size": "10"}, "integer"),
        ({"deduplicate": 1}, "boolean"),
        ({"max_evaluations": 10}, "max_evaluations"),
        ({"unknown": 1}, "unknown configuration keys"),
        (
            {"stall_generations": None, "max_generations": None},
            "at least one stopping criterion",
        ),
    ],
)
def test_rejects_invalid_configurations(data: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        GAConfig.from_dict(data)
