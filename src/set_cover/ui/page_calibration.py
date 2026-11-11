"""Página de calibración: cómo funciona, explorar resultados y calibrar con valores propios."""

import signal
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import streamlit

from set_cover.ga.calibration import (
    REFINEMENT_GRID,
    SEARCH_SPACE,
    CalibrationError,
    CalibrationSettings,
    ParameterValue,
    Stage,
    default_workers,
    settings_record,
    space_size,
)
from set_cover.ga.config import GAConfig
from set_cover.ga.jsonio import write_json
from set_cover.ui import charts, page_ga, session
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    chart,
    csv_rows,
    current_problem,
    json_object,
    show_table,
    solver_jobs,
    stage_costs,
)
from set_cover.ui.data import duration, number, spanish, stage_progress
from set_cover.ui.solver_jobs import (
    SEARCH_SPACE_NAME,
    SolverJobError,
    calibration_command,
    read_output,
)

# Segundos por evaluación medidos en esta instancia: 100.000 evaluaciones tardan unos 10 s.
_SECONDS_PER_EVALUATION = 1e-4
_SAVED_LABEL = "Calibración guardada del proyecto"
_CALIBRATION_STAGES = [Stage.EXPLORE, Stage.REFINE, Stage.FINAL]
_SOURCE_KEY = "calibration-source"
# Calibración recién terminada: se elige en «Explorar resultados» en la siguiente ejecución,
# antes de crear el selector, y se ofrece llevarla al GA.
_SELECT_ONCE_KEY = "calibration-select-once"
_MONITORED_KEY = "calibration-monitored"
_ANNOUNCED_KEY = "calibration-announced"


@dataclass(frozen=True)
class _StageInfo:
    title: str
    what: str
    why: str
    table: str


_STAGES: dict[Stage, _StageInfo] = {
    Stage.EXPLORE: _StageInfo(
        "A · Exploración",
        "Sortea configuraciones al azar del espacio de búsqueda y ejecuta cada una con pocas "
        "semillas y el mismo presupuesto de evaluaciones.",
        "El espacio tiene miles de combinaciones: probarlas todas con muchas semillas es "
        "imposible. Una muestra amplia y barata descarta rápido las zonas malas.",
        "Una fila por configuración, de mejor a peor. «Similar a la mejor» marca las que "
        "empatan en mediana (a menos de 0,2 %); entre ellas gana la más estable.",
    ),
    Stage.REFINE: _StageInfo(
        "B · Refinamiento",
        "Toma las mejores de A y, para cada una, algunas **vecinas**: la misma configuración "
        "con un solo parámetro movido a un valor cercano. Las ejecuta con más semillas, "
        "distintas de las de A.",
        "Con pocas semillas, A puede premiar la suerte. Más semillas nuevas confirman qué "
        "configuraciones son buenas de verdad, y las vecinas ajustan los valores finos. La "
        "primera del ranking es la **configuración seleccionada**.",
        "Igual que en A, con más ejecuciones por configuración: los números son más confiables.",
    ),
    Stage.TIME: _StageInfo(
        "Tiempo fijo",
        "Ejecuta las mejores de B sin parada por estancamiento ni presupuesto: todas corren "
        "el mismo tiempo.",
        "Comprueba si el orden se mantiene cuando se mide por tiempo en vez de por "
        "evaluaciones. Es informativa: no cambia la selección.",
        "Mismo formato; el tiempo por ejecución es igual para todas.",
    ),
    Stage.FINAL: _StageInfo(
        "C · Evaluación final",
        "Ejecuta solo la configuración seleccionada con muchas semillas que ninguna etapa "
        "anterior usó.",
        "La seleccionada ganó en B, en parte por azar: medirla con las mismas semillas "
        "exageraría su rendimiento. Semillas nuevas dan una estimación honesta. No cambia "
        "los parámetros.",
        "Una sola fila: el rendimiento esperado de la configuración elegida.",
    ),
}

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
_SUMMARY_DECIMALS: dict[str, int | None] = {
    "#": 0,
    "Ejecuciones": 0,
    "Mediana": 0,
    "Media": 1,
    "Desv.": 1,
    "Mejor": 0,
    "Peor": 0,
    "s/ejec.": 2,
    "s hasta la mejor (mediana)": 2,
    "Población": 0,
    "P. cruce": None,
    "Tasa mut.": None,
    "Torneo": 0,
    "Élites": 0,
    "Greedy": None,
    "Estanc.": 0,
}
_EFFECT_TEXT_COLUMNS = frozenset({"parameter", "value"})
_EFFECT_DECIMALS: dict[str, int | None] = {
    "configs": 0,
    "runs": 0,
    "mean_cost": 1,
    "median_cost": 0,
    "best_cost": 0,
    "reached_best_rate": 3,
    "mean_seconds": 2,
}
PARAMETER_NAMES = {
    "population_size": "Tamaño de población",
    "crossover_rate": "Probabilidad de cruce",
    "mutation_rate": "Tasa de mutación",
    "mutation": "Operador de mutación",
    "tournament_size": "Tamaño del torneo",
    "elite_count": "Élites",
    "crossover": "Cruce",
    "greedy_fraction": "Proporción greedy",
    "stall_generations": "Generaciones sin mejora",
    "prune_order": "Orden de poda",
}


@dataclass(frozen=True)
class _Preset:
    explore_configs: int
    explore_seeds: int
    refine_top: int
    refine_neighbors: int
    refine_seeds: int
    final_seeds: int
    evaluation_budget: int


_PRESETS = {
    "Rápida": _Preset(12, 3, 3, 2, 5, 10, 20_000),
    "Media": _Preset(24, 4, 4, 3, 8, 20, 50_000),
    "Completa (como la del proyecto)": _Preset(40, 5, 6, 4, 15, 30, 100_000),
}


def render() -> None:
    current_problem()
    streamlit.title("Calibración del GA")
    streamlit.markdown(
        "El algoritmo genético tiene **10 parámetros** (tamaño de población, mutación, cruce…) "
        "y su rendimiento depende mucho de ellos. **Calibrar** es buscar la combinación que da "
        "soluciones más baratas de forma confiable. Como el GA es aleatorio, cada combinación "
        "se prueba con varias **semillas** y se compara por su **mediana**, no por una sola "
        "ejecución afortunada."
    )
    how_tab, results_tab, run_tab = streamlit.tabs(
        ["Cómo funciona", "Explorar resultados", "Calibrar con mis valores"]
    )
    with how_tab:
        _how_it_works()
    with results_tab:
        _explore_results()
    with run_tab:
        _calibrate()


def _saved_settings() -> dict[str, object]:
    try:
        return json_object(PATHS.calibration / "settings.json")
    except RESULT_ERRORS:
        return settings_record(CalibrationSettings())


def _setting(settings: Mapping[str, object], key: str) -> int:
    value = settings.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _node(name: str, lines: Sequence[str], style: str = "") -> str:
    label = "\\n".join(lines)
    return f'{name} [label="{label}"{style}];'


def _how_it_works() -> None:
    settings = _saved_settings()
    top = _setting(settings, "refine_top")
    muted = ', fillcolor="#f8f9fa", color="#adb5bd"'
    nodes = [
        _node(
            "space",
            ["Espacio de búsqueda", f"{spanish(space_size(SEARCH_SPACE))} combinaciones"],
            muted,
        ),
        _node(
            "explore",
            [
                "A · Exploración",
                f"{_setting(settings, 'explore_configs')} configuraciones",
                f"x {_setting(settings, 'explore_seeds')} semillas",
            ],
        ),
        _node(
            "refine",
            [
                "B · Refinamiento",
                f"{top} mejores + hasta {_setting(settings, 'refine_neighbors')} vecinas c/u",
                f"x {_setting(settings, 'refine_seeds')} semillas nuevas",
            ],
        ),
        _node(
            "final",
            [
                "C · Evaluación final",
                "1 configuración",
                f"x {_setting(settings, 'final_seeds')} semillas nuevas",
            ],
            ', fillcolor="#2a9d8f", fontcolor="white"',
        ),
        _node(
            "time",
            [
                "Tiempo fijo (informativa)",
                f"{_setting(settings, 'time_top')} mejores",
                f"x {_setting(settings, 'time_seeds')} semillas",
            ],
            muted,
        ),
    ]
    streamlit.markdown(
        "La calibración avanza como un **embudo**: muchas configuraciones con pocas pruebas al "
        "principio, pocas configuraciones con muchas pruebas al final. Los números son los de "
        "la calibración guardada del proyecto."
    )
    streamlit.graphviz_chart(
        "digraph {\n"
        'rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef6f5", '
        'color="#2a9d8f", fontname="sans-serif", fontsize=11]; '
        'edge [color="#6c757d", fontname="sans-serif", fontsize=10];\n'
        + "\n".join(nodes)
        + '\nspace -> explore [label="muestra al azar"];\n'
        f'explore -> refine [label="las {top} mejores"];\n'
        'refine -> final [label="la primera del ranking"];\n'
        "refine -> time [style=dashed];\n}"
    )
    columns = streamlit.columns(4)
    for column, stage in zip(
        columns, (Stage.EXPLORE, Stage.REFINE, Stage.FINAL, Stage.TIME), strict=True
    ):
        info = _STAGES[stage]
        with column.container(border=True):
            streamlit.markdown(f"**{info.title}**")
            streamlit.markdown(f"**Qué hace.** {info.what}")
            streamlit.markdown(f"**Por qué.** {info.why}")

    streamlit.subheader("Cómo se elige la ganadora")
    streamlit.markdown(
        "1. Solo compiten las configuraciones en que **todas** las ejecuciones cubren a todos "
        "los clientes.\n"
        "2. Se busca la **mediana** de costo más baja. Las que quedan a menos de 0,2 % de ella "
        "se consideran **empatadas**: esa diferencia es ruido de las semillas.\n"
        "3. Entre las empatadas gana la de **menor desviación estándar** (más estable); si "
        "siguen empatadas, la más **rápida**, y luego la que converge antes."
    )
    streamlit.subheader("Por qué cada etapa usa semillas distintas")
    streamlit.markdown(
        "Si se elige una configuración por sus resultados con ciertas semillas y luego se mide "
        "con esas mismas semillas, el resultado sale mejor de lo que es (sesgo de selección). "
        "Por eso A, B y C usan rangos de semillas que no se solapan: 1001…, 2001… y 9001…"
    )
    streamlit.info(
        "En la pestaña **Calibrar con mis valores** puede repetir el procedimiento con su "
        "propio espacio de búsqueda y luego ejecutar la configuración ganadora en la página "
        "del algoritmo genético."
    )


@dataclass(frozen=True)
class _Source:
    label: str
    directory: Path
    # Etiqueta de su configuración en «Partir de la configuración» del GA.
    ga_source: str


def _sources() -> list[_Source]:
    sources = [_Source(_SAVED_LABEL, PATHS.calibration, page_ga.PROJECT_CALIBRATED_SOURCE)]
    sources.extend(
        _Source(item.label, item.directory, item.label) for item in session.calibrations()
    )
    return sources


def _explore_results() -> None:
    sources = _sources()
    labels = [source.label for source in sources]
    requested = streamlit.session_state.pop(_SELECT_ONCE_KEY, None)
    if requested in labels:
        streamlit.session_state[_SOURCE_KEY] = requested
    elif streamlit.session_state.get(_SOURCE_KEY) not in labels:
        streamlit.session_state.pop(_SOURCE_KEY, None)
    chosen = streamlit.selectbox(
        "Calibración",
        labels,
        key=_SOURCE_KEY,
        help="La guardada del proyecto y las que ejecute en esta sesión.",
    )
    source = sources[labels.index(chosen)]
    _overview(source.directory)
    _selected_config(source)

    present = [stage for stage in _STAGES if _summary_path(source.directory, stage).is_file()]
    if not present:
        streamlit.info("Esta calibración todavía no terminó ninguna etapa.")
        return
    streamlit.subheader("Detalle de una etapa")
    stage_labels = {_STAGES[stage].title: stage for stage in present}
    first = next(iter(stage_labels))
    picked = streamlit.segmented_control(
        "Etapa", list(stage_labels), default=first, key=f"calibration-stage-{source.directory}"
    )
    stage = stage_labels[picked or first]
    info = _STAGES[stage]
    streamlit.markdown(f"**Qué hace.** {info.what}  \n**Por qué.** {info.why}")
    _stage_detail(source.directory, stage)


def _summary_path(directory: Path, stage: Stage) -> Path:
    return directory / f"summary_{stage.value}.csv"


def _summary(directory: Path, stage: Stage) -> list[dict[str, str]]:
    path = _summary_path(directory, stage)
    return csv_rows(path) if path.is_file() else []


def _overview(directory: Path) -> None:
    rows: list[dict[str, object]] = []
    for stage, info in _STAGES.items():
        try:
            summary = _summary(directory, stage)
        except RESULT_ERRORS:
            summary = []
        if not summary:
            continue
        best = summary[0]
        rows.append(
            {
                "Etapa": info.title,
                "Configuraciones": len(summary),
                "Ejecuciones": int(sum(number(row, "runs") or 0.0 for row in summary)),
                "Primera del ranking": best.get("config_id"),
                "Mediana": number(best, "median_cost"),
                "Desv.": number(best, "std_cost"),
                "Mejor costo": number(best, "best_cost"),
                "s/ejec.": number(best, "mean_seconds"),
            }
        )
    if rows:
        streamlit.markdown("**Resumen por etapa** (la primera del ranking de cada una)")
        show_table(rows, {"Mediana": 0, "Desv.": 1, "Mejor costo": 0, "s/ejec.": 2})


def _selected_config(source: _Source) -> None:
    path = source.directory / "selected_config.json"
    if not path.is_file():
        return
    try:
        selected = GAConfig.from_dict(json_object(path))
    except RESULT_ERRORS:
        streamlit.warning(f"`{path}` no es una configuración válida: {sys.exception()}")
        return
    default = GAConfig().to_dict()
    with streamlit.container(border=True):
        streamlit.markdown(f"**Configuración seleccionada:** `{selected.config_id()}`")
        streamlit.caption(
            "Ganó el refinamiento (B). «Cambió» marca los parámetros distintos de los valores "
            "por defecto del algoritmo."
        )
        show_table(
            [
                {
                    "Parámetro": PARAMETER_NAMES.get(name, name),
                    "Por defecto": _text(default[name]),
                    "Calibrado": _text(value),
                    "Cambió": "sí" if default[name] != value else "",
                }
                for name, value in selected.to_dict().items()
            ]
        )
        if streamlit.button(
            "Ejecutar el GA con esta configuración",
            icon=":material/play_arrow:",
            type="primary",
            key=f"use-{source.directory}",
        ):
            open_in_ga(source.ga_source)


def open_in_ga(ga_source: str) -> None:
    """Abre la página del GA con ``ga_source`` cargada en el formulario."""
    streamlit.session_state[page_ga.SOURCE_KEY] = ga_source
    streamlit.switch_page(streamlit.Page(page_ga.render, url_path=page_ga.URL_PATH))


def _text(value: object) -> str:
    if value is None:
        return "ninguno"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    return spanish(value, None)


def _stage_detail(directory: Path, stage: Stage) -> None:
    try:
        summary = _summary(directory, stage)
    except RESULT_ERRORS:
        streamlit.warning(f"No se pudo leer el resumen: {sys.exception()}")
        return
    key = f"{directory}-{stage.value}"
    shown = len(summary)
    if len(summary) > 1:
        shown = streamlit.slider(
            "Configuraciones en la tabla",
            1,
            len(summary),
            min(10, len(summary)),
            key=f"calibration-rows-{key}",
        )
    show_table([_display_row(row) for row in summary[:shown]], _SUMMARY_DECIMALS)
    streamlit.caption(
        f":material/visibility: **Cómo leerla.** {_STAGES[stage].table} Mediana, media, mejor "
        "y peor son costos: **más bajo es mejor**. Desv. es la variación entre semillas: más "
        f"baja es más estable. Archivo: `{_summary_path(directory, stage)}`."
    )

    left, right = streamlit.columns(2)
    with left:
        chart(
            charts.calibration_scatter(summary, "Calidad vs tiempo por configuración"),
            f"calibration-scatter-{key}",
            "Cada punto es una configuración, coloreada por su operador de mutación. Eje "
            "vertical: mediana del costo; horizontal: tiempo medio por ejecución. **Abajo y a "
            "la izquierda es mejor**: barata y rápida. Pase el cursor para ver cuál es.",
        )
    try:
        costs = stage_costs(directory / f"runs_{stage.value}.jsonl")
    except RESULT_ERRORS:
        costs = {}
    top = summary[: min(8, len(summary))]
    boxes = {
        f"#{row['rank']} {row['config_id']}": costs[row["config_id"]]
        for row in top
        if row["config_id"] in costs
    }
    if boxes:
        with right:
            chart(
                charts.calibration_boxes(boxes, f"Costo por ejecución, {len(boxes)} primeras"),
                f"calibration-boxes-{key}",
                "Una caja por configuración, en orden del ranking; cada punto es una semilla. "
                "**Más abajo es mejor**, y una caja más corta es más estable. Explica el "
                "ranking: una configuración puede tener la misma mediana que otra y ganar por "
                "variar menos.",
            )
    _effects(directory, stage)


def _display_row(row: Mapping[str, str]) -> dict[str, object]:
    display: dict[str, object] = {}
    for key, header in _SUMMARY_COLUMNS.items():
        text = row.get(key, "")
        if key in _TEXT_COLUMNS:
            display[header] = "sí" if text == "True" else "" if text == "False" else text
        else:
            display[header] = number(row, key)
    return display


def _effects(directory: Path, stage: Stage) -> None:
    path = directory / f"parameter_effects_{stage.value}.csv"
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
    streamlit.markdown(
        "Agrupa todas las ejecuciones de la etapa según el valor de un parámetro y promedia. "
        "Sirve para ver tendencias, por ejemplo qué operador de mutación funciona mejor."
    )
    default = parameters.index("mutation") if "mutation" in parameters else 0
    parameter = streamlit.selectbox(
        "Parámetro",
        parameters,
        index=default,
        format_func=lambda name: PARAMETER_NAMES.get(name, name),
        key=f"calibration-effect-{directory}-{stage.value}",
    )
    chart(
        charts.parameter_effect(effects, parameter),
        f"calibration-effect-{directory}-{stage.value}-chart",
        "Barras: costo medio y mediano de las ejecuciones con cada valor (eje izquierdo; **más "
        "bajas es mejor**). Línea: porcentaje de ejecuciones que alcanzaron el mejor costo de "
        "la etapa (eje derecho; **más alta es mejor**). Ojo: las configuraciones no cubren "
        "todas las combinaciones, así que un efecto puede deberse a otro parámetro que varió "
        "a la vez.",
    )
    with streamlit.expander("Tabla completa de efectos"):
        show_table([_effect_row(row) for row in effects], _EFFECT_DECIMALS)


def _effect_row(row: Mapping[str, str]) -> dict[str, object]:
    display: dict[str, object] = {}
    for key, text in row.items():
        value = number(row, key)
        if key == "value" and value is not None:
            display[key] = spanish(value, None)
        elif key in _EFFECT_TEXT_COLUMNS or value is None:
            display[key] = text
        else:
            display[key] = value
    return display


def _calibrate() -> None:
    streamlit.markdown(
        "Repite el procedimiento A → B → C en esta máquina con **sus** valores. Las ejecuciones "
        "corren en paralelo. Al terminar, la configuración ganadora aparece en la página del "
        "algoritmo genético y sus resultados en «Explorar resultados». La etapa de tiempo fijo "
        "se omite porque no cambia la selección."
    )
    _monitor()

    preset_name = streamlit.segmented_control(
        "Tamaño",
        list(_PRESETS),
        default=next(iter(_PRESETS)),
        key="calibration-preset",
        help="Valores iniciales del formulario; puede cambiarlos.",
    )
    preset_name = preset_name or next(iter(_PRESETS))
    preset = _PRESETS[preset_name]
    prefix = f"calibration-{preset_name}"

    with streamlit.expander("Valores a explorar (espacio de búsqueda)"):
        streamlit.caption(
            "Elija qué valores puede tomar cada parámetro. Menos valores: búsqueda más "
            "enfocada. El refinamiento solo prueba valores dentro de lo que elija."
        )
        space: dict[str, list[ParameterValue]] = {}
        halves = streamlit.columns(2)
        for index, (name, grid) in enumerate(REFINEMENT_GRID.items()):
            space[name] = halves[index % 2].multiselect(
                PARAMETER_NAMES[name],
                list(grid),
                default=list(SEARCH_SPACE[name]),
                format_func=_value_text,
                key=f"calibration-space-{name}",
            )
    missing = [PARAMETER_NAMES[name] for name, values in space.items() if not values]
    if missing:
        streamlit.error(f"Elija al menos un valor para: {', '.join(missing)}.")
        return
    ordered = {name: _in_grid_order(name, values) for name, values in space.items()}
    combinations = space_size(ordered)
    streamlit.caption(f"Espacio de búsqueda: **{spanish(combinations)} combinaciones**.")

    columns = streamlit.columns(4)
    explore_configs = columns[0].number_input(
        "A · configuraciones",
        min_value=1,
        max_value=combinations,
        value=min(preset.explore_configs, combinations),
        key=f"{prefix}-explore-configs",
        help="Cuántas combinaciones sortear del espacio.",
    )
    explore_seeds = columns[1].number_input(
        "A · semillas por configuración",
        min_value=1,
        max_value=50,
        value=preset.explore_seeds,
        key=f"{prefix}-explore-seeds",
    )
    refine_top = columns[2].number_input(
        "B · mejores que pasan",
        min_value=1,
        max_value=int(explore_configs),
        value=min(preset.refine_top, int(explore_configs)),
        key=f"{prefix}-refine-top",
    )
    refine_neighbors = columns[3].number_input(
        "B · vecinas por cada una",
        min_value=0,
        max_value=10,
        value=preset.refine_neighbors,
        key=f"{prefix}-refine-neighbors",
        help="Variantes con un parámetro movido a un valor cercano.",
    )
    columns = streamlit.columns(4)
    refine_seeds = columns[0].number_input(
        "B · semillas por configuración",
        min_value=1,
        max_value=100,
        value=preset.refine_seeds,
        key=f"{prefix}-refine-seeds",
    )
    final_seeds = columns[1].number_input(
        "C · semillas finales",
        min_value=1,
        max_value=200,
        value=preset.final_seeds,
        key=f"{prefix}-final-seeds",
    )
    budget = columns[2].number_input(
        "Evaluaciones por ejecución",
        min_value=1_000,
        max_value=1_000_000,
        value=preset.evaluation_budget,
        step=5_000,
        key=f"{prefix}-budget",
        help="Cada ejecución para aquí o tras sus generaciones sin mejora. Más evaluaciones: "
        "mejores resultados, más lentos.",
    )
    workers = columns[3].number_input(
        "Procesos en paralelo",
        min_value=1,
        max_value=64,
        value=default_workers(),
        key=f"{prefix}-workers",
        help="Por defecto, la mitad de los núcleos de la máquina.",
    )

    try:
        settings = CalibrationSettings(
            evaluation_budget=int(budget),
            explore_configs=int(explore_configs),
            explore_seeds=int(explore_seeds),
            refine_top=int(refine_top),
            refine_neighbors=int(refine_neighbors),
            refine_seeds=int(refine_seeds),
            final_seeds=int(final_seeds),
            workers=int(workers),
            search_space=ordered,
        )
    except CalibrationError:
        streamlit.error(f"Ajustes inválidos: {sys.exception()}")
        return
    runs = (
        settings.explore_configs * settings.explore_seeds
        + settings.refine_top * (1 + settings.refine_neighbors) * settings.refine_seeds
        + settings.final_seeds
    )
    seconds = runs * settings.evaluation_budget * _SECONDS_PER_EVALUATION / settings.workers
    streamlit.markdown(
        f"Hasta **{spanish(runs)} ejecuciones**; unos **{duration(seconds)}** con "
        f"{settings.workers} procesos. Corre en segundo plano: puede seguir usando la interfaz. "
        "Si la detiene, puede continuar después sin repetir lo ya hecho."
    )
    busy = _any_running()
    if streamlit.button(
        "Calibrar",
        type="primary",
        icon=":material/tune:",
        disabled=busy,
        help="Espere a que termine la calibración en curso." if busy else None,
    ):
        directory = PATHS.solver_runs / f"calibracion-{datetime.now():%Y%m%d-%H%M%S}"
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "settings.json", settings_record(settings))
        calibration = session.SessionCalibration(directory, settings)
        session.add_calibration(calibration)
        _start(calibration)
        streamlit.rerun()


def _value_text(value: ParameterValue) -> str:
    return value if isinstance(value, str) else spanish(value, None)


def _in_grid_order(name: str, values: Sequence[ParameterValue]) -> tuple[ParameterValue, ...]:
    return tuple(value for value in REFINEMENT_GRID[name] if value in values)


def _complete(directory: Path) -> bool:
    return (directory / "final_statistics.json").is_file()


def _start(calibration: session.SessionCalibration) -> None:
    space = {name: list(values) for name, values in calibration.settings.search_space.items()}
    write_json(calibration.directory / SEARCH_SPACE_NAME, space)
    command = calibration_command(
        calibration.directory,
        calibration.settings,
        PATHS.coverage,
        PATHS.costs,
        _CALIBRATION_STAGES,
    )
    try:
        solver_jobs().start(calibration.directory, command, signal.SIGINT)
    except (SolverJobError, OSError):
        streamlit.error(f"No se pudo iniciar la calibración: {sys.exception()}")
        return
    streamlit.session_state[_MONITORED_KEY] = calibration.label


def _monitored() -> session.SessionCalibration | None:
    label = streamlit.session_state.get(_MONITORED_KEY)
    return next((item for item in session.calibrations() if item.label == label), None)


def _any_running() -> bool:
    jobs = solver_jobs()
    return any(jobs.running(item.directory) for item in session.calibrations())


@streamlit.fragment(run_every=2)
def _monitor() -> None:
    calibration = _monitored()
    if calibration is None:
        return
    jobs = solver_jobs()
    with streamlit.container(border=True):
        streamlit.markdown(f"**{calibration.label}**")
        _progress_bars(calibration.directory)
        if jobs.running(calibration.directory):
            streamlit.caption(
                "Corre en otro proceso: puede seguir usando la interfaz. Se actualiza cada 2 s."
            )
            if streamlit.button("Detener", key="calibration-stop"):
                jobs.stop(calibration.directory)
                streamlit.toast("Terminando las ejecuciones en curso…")
        elif _complete(calibration.directory):
            _finished(calibration)
        else:
            _stopped(calibration, jobs.exit_code(calibration.directory))
        with streamlit.expander("Salida del proceso"):
            streamlit.code(read_output(calibration.directory) or "(sin salida)", language=None)


def _progress_bars(directory: Path) -> None:
    for stage in _CALIBRATION_STAGES:
        try:
            progress = stage_progress(directory, stage.value)
        except RESULT_ERRORS:
            continue
        title = _STAGES[stage].title
        if progress.total is None:
            streamlit.progress(0.0, text=f"{title}: en espera")
        else:
            streamlit.progress(
                progress.finished / progress.total if progress.total else 1.0,
                text=f"{title}: {progress.finished} de {progress.total} ejecuciones",
            )


def _finished(calibration: session.SessionCalibration) -> None:
    _announce(calibration)
    try:
        selected = GAConfig.from_dict(json_object(calibration.directory / "selected_config.json"))
    except RESULT_ERRORS:
        selected = None
    streamlit.success(
        "Calibración terminada."
        + ("" if selected is None else f" Seleccionada: `{selected.config_id()}`.")
        + " Vea el detalle en «Explorar resultados»."
    )
    left, right = streamlit.columns(2)
    if left.button(
        "Ejecutar el GA con esta configuración",
        type="primary",
        icon=":material/play_arrow:",
        key="use-finished-calibration",
    ):
        open_in_ga(calibration.label)
    if right.button("Cerrar", key="calibration-dismiss"):
        streamlit.session_state.pop(_MONITORED_KEY, None)
        streamlit.rerun(scope="app")


def _stopped(calibration: session.SessionCalibration, code: int | None) -> None:
    streamlit.warning(
        "La calibración se detuvo antes de terminar"
        + ("" if code in (None, 0, 130) else f" (código {code}; revise la salida)")
        + ". «Continuar» retoma sin repetir las ejecuciones registradas."
    )
    if streamlit.button("Continuar", key="calibration-resume"):
        _start(calibration)
        streamlit.rerun(scope="fragment")


def _announce(calibration: session.SessionCalibration) -> None:
    """La primera vez que se ve terminada, recarga la página para que el GA y la exploración la
    muestren."""
    announced = streamlit.session_state.get(_ANNOUNCED_KEY)
    done = announced if isinstance(announced, list) else []
    if calibration.label in done:
        return
    streamlit.session_state[_ANNOUNCED_KEY] = [*done, calibration.label]
    streamlit.session_state[_SELECT_ONCE_KEY] = calibration.label
    streamlit.rerun(scope="app")
