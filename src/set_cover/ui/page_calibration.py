"""Página de calibración: etapas, ranking, variabilidad y efecto de cada parámetro."""

import sys
from collections.abc import Mapping

import streamlit

from set_cover.ga.calibration import Stage
from set_cover.ga.config import GAConfig
from set_cover.ui import charts
from set_cover.ui.common import PATHS, RESULT_ERRORS, csv_rows, json_object, stage_costs
from set_cover.ui.data import number

_STAGES: dict[Stage, tuple[str, str]] = {
    Stage.EXPLORE: (
        "A · Exploración",
        "Configuraciones muestreadas sin reemplazo del espacio inicial, 5 semillas cada una, "
        "con el mismo presupuesto de evaluaciones. Descarta las regiones malas del espacio.",
    ),
    Stage.REFINE: (
        "B · Refinamiento",
        "Las mejores de A y sus vecinas (un parámetro cambiado al valor adyacente de una "
        "rejilla más fina), con 15 semillas nuevas. La primera del ranking es la seleccionada.",
    ),
    Stage.TIME: (
        "Tiempo fijo",
        "Las mejores de B sin parada por estancamiento ni presupuesto, con el mismo tiempo "
        "por ejecución. Es informativa: no cambia la selección.",
    ),
    Stage.FINAL: (
        "C · Evaluación final",
        "La configuración seleccionada con 30 semillas nunca usadas en la calibración. Mide "
        "su rendimiento sin sesgo de selección; no cambia los parámetros.",
    ),
}

# Columnas del resumen por configuración que se muestran, con su encabezado.
_SUMMARY_COLUMNS = {
    "rank": "#",
    "config_id": "Configuración",
    "similar_to_best": "Similar a la mejor",
    "runs": "Ejecuciones",
    "median_cost": "Mediana",
    "mean_cost": "Media",
    "std_cost": "Desv.",
    "best_cost": "Mejor",
    "worst_cost": "Peor",
    "mean_seconds": "s/ejec.",
    "median_seconds_to_best": "s hasta la mejor (mediana)",
    "population_size": "Población",
    "crossover_rate": "P. cruce",
    "mutation": "Mutación",
    "mutation_rate": "Tasa mut.",
    "tournament_size": "Torneo",
    "elite_count": "Élites",
    "crossover": "Cruce",
    "greedy_fraction": "Greedy",
    "stall_generations": "Estanc.",
    "prune_order": "Poda",
}
_TEXT_COLUMNS = frozenset({"config_id", "similar_to_best", "mutation", "crossover", "prune_order"})


def render() -> None:
    streamlit.title("Calibración del GA")
    streamlit.markdown(
        "Calibración en tres etapas con semillas disjuntas, más una comparación con tiempo "
        "fijo. **Criterio de selección:** solo compiten configuraciones con 100 % de "
        "ejecuciones factibles; las que tienen mediana a lo sumo 0,2 % peor que la mejor se "
        "consideran similares, y entre ellas gana la de menor desviación estándar, luego menor "
        "tiempo y luego convergencia más rápida."
    )
    _overview()
    _selected_config()

    labels = {label: stage for stage, (label, _) in _STAGES.items()}
    chosen = streamlit.radio("Etapa", list(labels), horizontal=True)
    stage = labels[chosen]
    streamlit.caption(_STAGES[stage][1])
    _stage_detail(stage)


def _summary(stage: Stage) -> list[dict[str, str]]:
    path = PATHS.calibration / f"summary_{stage.value}.csv"
    return csv_rows(path) if path.is_file() else []


def _overview() -> None:
    rows: list[dict[str, object]] = []
    for stage, (label, _) in _STAGES.items():
        try:
            summary = _summary(stage)
        except RESULT_ERRORS:
            summary = []
        if not summary:
            rows.append({"Etapa": label, "Configuraciones": 0, "Ejecuciones": 0})
            continue
        best = summary[0]
        rows.append(
            {
                "Etapa": label,
                "Configuraciones": len(summary),
                "Ejecuciones": int(sum(number(row, "runs") or 0.0 for row in summary)),
                "Primera del ranking": best["config_id"],
                "Mediana": number(best, "median_cost"),
                "Desv.": number(best, "std_cost"),
                "Mejor costo": number(best, "best_cost"),
                "s/ejec.": number(best, "mean_seconds"),
            }
        )
    number_format = streamlit.column_config.NumberColumn(format="localized")
    streamlit.dataframe(
        rows,
        hide_index=True,
        column_config={"Mediana": number_format, "Mejor costo": number_format},
    )
    settings_path = PATHS.calibration / "settings.json"
    if settings_path.is_file():
        with streamlit.expander("Ajustes de la calibración (`settings.json`)"):
            try:
                streamlit.json(json_object(settings_path))
            except RESULT_ERRORS:
                streamlit.warning(f"No se pudo leer: {sys.exception()}")


def _selected_config() -> None:
    path = PATHS.calibration / "selected_config.json"
    if not path.is_file():
        return
    try:
        selected = GAConfig.from_dict(json_object(path))
    except RESULT_ERRORS:
        streamlit.warning(f"`{path}` no es una configuración válida: {sys.exception()}")
        return
    default = GAConfig().to_dict()
    with streamlit.expander(f"Configuración seleccionada: `{selected.config_id()}`", expanded=True):
        streamlit.dataframe(
            [
                {
                    "Parámetro": name,
                    "Por defecto": _text(default[name]),
                    "Calibrado": _text(value),
                    "Cambió": "sí" if default[name] != value else "",
                }
                for name, value in selected.to_dict().items()
            ],
            hide_index=True,
        )


def _text(value: object) -> str:
    return "ninguno" if value is None else str(value)


def _stage_detail(stage: Stage) -> None:
    try:
        summary = _summary(stage)
    except RESULT_ERRORS:
        streamlit.warning(f"No se pudo leer el resumen: {sys.exception()}")
        return
    if not summary:
        streamlit.info(f"La etapa `{stage.value}` no tiene resultados en `{PATHS.calibration}`.")
        return

    shown = len(summary)
    if len(summary) > 1:
        shown = streamlit.slider(
            "Configuraciones en la tabla", 1, len(summary), min(10, len(summary))
        )
    streamlit.dataframe(
        [_display_row(row) for row in summary[:shown]],
        hide_index=True,
        column_config={
            header: streamlit.column_config.NumberColumn(format="localized")
            for header in ("Mediana", "Media", "Mejor", "Peor")
        },
    )
    streamlit.caption(f"Ranking completo: `{PATHS.calibration / f'summary_{stage.value}.csv'}`.")

    left, right = streamlit.columns(2)
    left.plotly_chart(
        charts.calibration_scatter(summary, "Calidad vs tiempo por configuración"),
        key=f"calibration-scatter-{stage.value}",
    )
    try:
        costs = stage_costs(PATHS.calibration / f"runs_{stage.value}.jsonl")
    except RESULT_ERRORS:
        costs = {}
    top = summary[: min(8, len(summary))]
    boxes = {
        f"#{row['rank']} {row['config_id']}": costs[row["config_id"]]
        for row in top
        if row["config_id"] in costs
    }
    if boxes:
        right.plotly_chart(
            charts.calibration_boxes(boxes, f"Costo por ejecución, {len(boxes)} primeras"),
            key=f"calibration-boxes-{stage.value}",
        )

    _effects(stage)


def _display_row(row: Mapping[str, str]) -> dict[str, object]:
    display: dict[str, object] = {}
    for key, header in _SUMMARY_COLUMNS.items():
        text = row.get(key, "")
        if key in _TEXT_COLUMNS:
            display[header] = "sí" if text == "True" else "" if text == "False" else text
        else:
            display[header] = number(row, key)
    return display


def _effects(stage: Stage) -> None:
    path = PATHS.calibration / f"parameter_effects_{stage.value}.csv"
    if not path.is_file():
        return
    try:
        effects = csv_rows(path)
    except RESULT_ERRORS:
        streamlit.warning(f"No se pudo leer `{path}`: {sys.exception()}")
        return
    parameters = list(dict.fromkeys(row["parameter"] for row in effects))
    if not parameters:
        return
    streamlit.subheader("Efecto de cada parámetro")
    streamlit.caption(
        "Promedio sobre las ejecuciones de la etapa agrupadas por el valor del parámetro. "
        "Las configuraciones no forman un diseño factorial: el efecto es descriptivo y puede "
        "estar confundido con otros parámetros."
    )
    default = parameters.index("mutation") if "mutation" in parameters else 0
    parameter = streamlit.selectbox("Parámetro", parameters, index=default)
    streamlit.plotly_chart(
        charts.parameter_effect(effects, parameter), key=f"calibration-effect-{stage.value}"
    )
    with streamlit.expander("Tabla completa de efectos"):
        streamlit.dataframe(effects, hide_index=True)
