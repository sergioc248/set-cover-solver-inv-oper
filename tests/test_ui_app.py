"""Pruebas de las páginas con el ejecutor de Streamlit, sin navegador."""

import dataclasses
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from set_cover.ui import common, page_calibration
from set_cover.ui.solver_jobs import read_output

_PAGES = ("page_problem", "page_exact", "page_ga", "page_compare", "page_calibration")
_GA_PREFIX = "ga-calibrated.json (proyecto)"


def _pages(*names: str) -> None:
    # AppTest ejecuta el cuerpo de esta función como script: los imports van adentro.
    import importlib  # noqa: PLC0415

    for name in names:
        importlib.import_module(f"set_cover.ui.{name}").render()


def _ga_and_sidebar() -> None:
    from set_cover.ui import page_ga, sidebar  # noqa: PLC0415

    page_ga.render()
    sidebar.render()


def _app(*names: str) -> AppTest:
    app = AppTest.from_function(_pages, args=names, default_timeout=180)
    app.run()
    return app


def _clean(app: AppTest) -> None:
    assert not app.exception
    assert not app.error


@pytest.mark.parametrize("name", _PAGES)
def test_every_page_renders_without_errors(name: str) -> None:
    _clean(_app(name))


def test_problem_page_explains_the_indices_and_explores_them() -> None:
    app = _app("page_problem")

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Clientes (i)"] == "500"
    assert metrics["Pares antena-cliente con señal"] == "24.971"
    notation = app.dataframe[0].value
    assert list(notation["Símbolo"]) == ["i", "j", "a_ij", "c_j", "x_j"]
    app.number_input[0].set_value(7).run()
    _clean(app)
    assert any("al cliente 7" in markdown.value for markdown in app.markdown)


def _run_ga(app: AppTest, runs: int) -> None:
    app.number_input(key=f"{_GA_PREFIX}-runs").set_value(runs)
    app.number_input(key=f"{_GA_PREFIX}-pop").set_value(20)
    app.checkbox(key=f"{_GA_PREFIX}-use-eval").check()
    app.number_input(key=f"{_GA_PREFIX}-eval").set_value(300)
    next(button for button in app.button if button.label == "Ejecutar el GA").click()
    app.run()
    _clean(app)


def test_ga_page_runs_a_batch_of_seeds() -> None:
    app = _app("page_ga")
    _run_ga(app, 3)

    assert any("Lote 1" in message.value for message in app.success)
    # Las pestañas se dibujan todas: «Resultados guardados» también tiene «Ejecuciones» (30).
    assert "3" in [metric.value for metric in app.metric if metric.label == "Ejecuciones"]
    batches = app.session_state["session-ga-batches"]
    assert [result.seed for result in batches[0].results] == [1, 2, 3]


def test_comparison_offers_saved_and_session_results_in_one_list() -> None:
    app = _app("page_ga", "page_compare")
    _run_ga(app, 2)

    options = next(item for item in app.multiselect if item.label == "Métodos a comparar").options
    assert "Gurobi · 92h (guardada)" in options
    assert "GA · evaluación final (30 semillas) (guardada)" in options
    assert any(option.endswith("(esta sesión)") for option in options)
    table = next(frame.value for frame in app.dataframe if "Cota inferior" in frame.value.columns)
    assert any(method.endswith("· mediana") for method in table["Método"])
    assert "None" not in table.to_string()
    assert any("Ningún método demostró optimalidad" in info.value for info in app.info)
    assert any("38.875" in info.value for info in app.info)


def test_calibration_page_switches_stages() -> None:
    app = _app("page_calibration")
    control = app.segmented_control(key=f"calibration-stage-{common.PATHS.calibration}")
    for stage in ("B · Refinamiento", "Tiempo fijo", "C · Evaluación final"):
        control.set_value(stage)
        app.run()
        _clean(app)


def test_custom_calibration_runs_and_feeds_the_ga(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        page_calibration, "PATHS", dataclasses.replace(common.PATHS, solver_runs=tmp_path)
    )
    app = _app("page_calibration", "page_ga")
    prefix = "calibration-Rápida"
    for name, value in (
        ("explore-configs", 2),
        ("explore-seeds", 1),
        ("refine-top", 1),
        ("refine-neighbors", 0),
        ("refine-seeds", 1),
        ("final-seeds", 2),
        ("budget", 1000),
        ("workers", 1),
    ):
        app.number_input(key=f"{prefix}-{name}").set_value(value)
    next(button for button in app.button if button.label == "Calibrar").click()
    app.run()
    _clean(app)

    # Corre en otro proceso: la página sigue disponible mientras tanto.
    (directory,) = tmp_path.iterdir()
    deadline = time.monotonic() + 120
    while not (directory / "final_statistics.json").is_file():
        assert time.monotonic() < deadline, read_output(directory)
        time.sleep(0.2)
    while common.solver_jobs().running(directory):
        time.sleep(0.1)
    app.run()
    _clean(app)

    label = f"Calibración de la sesión: {directory.name}"
    assert any("Calibración terminada" in message.value for message in app.success)
    assert app.selectbox(key="calibration-source").value == label
    assert label in app.selectbox(key="ga-source").options


def test_reset_forgets_the_session() -> None:
    app = AppTest.from_function(_ga_and_sidebar, default_timeout=180)
    app.run()
    _run_ga(app, 1)
    assert app.session_state["session-ga-batches"]

    app.button(key="session-reset").click()
    app.run()
    _clean(app)

    assert "session-ga-batches" not in app.session_state
    assert any("0 lote(s)" in markdown.value for markdown in app.markdown)


def test_selection_key_changes_when_new_options_appear() -> None:
    before = common.options_key("compare-methods", ["Gurobi · 4s (guardada)"])

    assert before == common.options_key("compare-methods", ["Gurobi · 4s (guardada)"])
    assert before != common.options_key(
        "compare-methods", ["Gurobi · 4s (guardada)", "GA · lote 1 (esta sesión)"]
    )
