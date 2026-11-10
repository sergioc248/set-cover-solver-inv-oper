"""Pruebas de las páginas con el ejecutor de Streamlit, sin navegador."""

import pytest
from streamlit.testing.v1 import AppTest

_PAGES = ("page_instance", "page_exact", "page_ga", "page_compare", "page_calibration")


def _page(name: str) -> None:
    # AppTest ejecuta el cuerpo de esta función como script: los imports van adentro.
    import importlib  # noqa: PLC0415

    importlib.import_module(f"set_cover.ui.{name}").render()


def _app(name: str) -> AppTest:
    app = AppTest.from_function(_page, args=(name,), default_timeout=120)
    app.run()
    return app


@pytest.mark.parametrize("name", _PAGES)
def test_every_page_renders_without_errors(name: str) -> None:
    app = _app(name)

    assert not app.exception
    assert not app.error


def test_instance_page_shows_the_instance_size() -> None:
    app = _app("page_instance")

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Clientes"] == "500"
    assert metrics["Antenas"] == "500"
    assert metrics["Enlaces de cobertura"] == "24,971"


def test_ga_page_runs_the_algorithm_and_keeps_the_result() -> None:
    app = _app("page_ga")
    prefix = "ga-calibrated.json"
    app.number_input(key=f"{prefix}-pop").set_value(20)
    app.checkbox(key=f"{prefix}-use-eval").check()
    app.number_input(key=f"{prefix}-eval").set_value(200)
    app.button[0].click()
    app.run()

    assert not app.exception
    assert not app.error
    assert any("Costo" in message.value for message in app.success)
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Clientes cubiertos"] == "500/500"
    assert len(app.session_state["ga_runs"]) == 1


def test_comparison_page_explains_the_missing_optimum() -> None:
    app = _app("page_compare")

    assert any("Ningún método demostró optimalidad" in info.value for info in app.info)
    assert any("38,875" in info.value for info in app.info)


def test_calibration_page_switches_stages() -> None:
    app = _app("page_calibration")
    for stage in ("B · Refinamiento", "Tiempo fijo", "C · Evaluación final"):
        app.radio[0].set_value(stage)
        app.run()
        assert not app.exception
        assert not app.error
