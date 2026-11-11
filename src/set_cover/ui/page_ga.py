"""Página del algoritmo genético: ejecutar lotes de semillas y revisar sus resultados."""

import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import streamlit
from streamlit.delta_generator import DeltaGenerator

from set_cover.ga.config import ConfigError, CrossoverKind, GAConfig, MutationKind, PruneOrder
from set_cover.ga.engine import GAResult, GenerationReport, run_ga
from set_cover.instance import Instance
from set_cover.ui import charts, session
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    chart,
    convergence,
    current_problem,
    ga_records,
    json_object,
    show_table,
)
from set_cover.ui.data import (
    GAGroup,
    GARunRecord,
    duration,
    extend_to_run_end,
    percent,
    spanish,
)

SOURCE_KEY = "ga-source"
URL_PATH = "genetico"
PROJECT_CALIBRATED_SOURCE = "calibrated.json (proyecto)"
_LAST_BATCH_KEY = "ga-last-batch"
DEFAULT_SOURCE = "Valores por defecto"
_PREFERRED_CONFIG = "calibrated.json"
_REDRAW_SECONDS = 0.5
_MAX_RUNS = 100

HISTORY_GUIDE = (
    "Eje horizontal: generaciones. La línea gruesa es el mejor costo encontrado hasta esa "
    "generación: solo baja. Las demás son el mejor, la media y el peor de la población en "
    "cada generación. **Más abajo es mejor.** Cuando la línea gruesa queda plana, el "
    "algoritmo dejó de mejorar; tras las generaciones sin mejora configuradas, se detiene."
)
DIVERSITY_GUIDE = (
    "Cuántos individuos distintos tiene la población en cada generación. **Más alto** "
    "significa más variedad para seguir explorando; si cae, la población se llena de copias "
    "de la misma solución. Con «descartar descendientes duplicados» se mantiene igual al "
    "tamaño de la población."
)
DISTRIBUTION_GUIDE = (
    "Cada punto es una ejecución (una semilla). La caja va del primer al tercer cuartil y la "
    "raya interior es la mediana. **Más abajo es mejor.** Una caja corta indica que las "
    "semillas dan resultados parecidos: el algoritmo es confiable, no depende de la suerte."
)
QUALITY_GUIDE = (
    "Cada línea tenue es una semilla y la gruesa es la mediana entre semillas: el mejor costo "
    "encontrado (eje vertical) a medida que pasa el tiempo (eje horizontal, logarítmico). "
    "**Abajo y a la izquierda es mejor**: menor costo en menos tiempo. Una curva que se "
    "aplana dejó de mejorar."
)
_LIVE_GUIDE = (
    "Se actualiza mientras corre. Línea continua: mejor costo hasta ahora (**más abajo es "
    "mejor**). Punteada: costo medio de la población, que fluctúa porque la mutación sigue "
    "probando cambios."
)


def render() -> None:
    instance, _ = current_problem()
    streamlit.title("Algoritmo genético")
    streamlit.markdown(
        "El algoritmo mantiene una **población** de soluciones. Cada solución es una cadena de "
        "500 bits: el bit j vale 1 si se instala la antena j. En cada **generación** elige "
        "padres por torneo, los **cruza**, aplica una **mutación** y repara a cada hijo para "
        "que cubra a todos los clientes y no tenga antenas sobrantes. Las mejores soluciones "
        "(**élites**) pasan intactas a la siguiente generación."
    )
    streamlit.markdown(
        "Es **aleatorio**: cada semilla da otro resultado. Para saber si una configuración es "
        "buena, ejecútela con varias semillas y mire la mediana y la dispersión."
    )
    run_tab, session_tab, saved_tab = streamlit.tabs(
        ["Ejecutar", "Resultados de esta sesión", "Resultados guardados del proyecto"]
    )
    with run_tab:
        _run_section(instance)
    with session_tab:
        _session_section(instance)
    with saved_tab:
        _saved_section()


def config_sources() -> dict[str, GAConfig | Path]:
    """Configuraciones para cargar: archivos del proyecto, calibraciones de la sesión y default."""
    files = sorted(
        PATHS.configs.glob("*.json"), key=lambda path: (path.name != _PREFERRED_CONFIG, path.name)
    )
    sources: dict[str, GAConfig | Path] = {f"{path.name} (proyecto)": path for path in files}
    for calibration in session.calibrations():
        selected = calibration.directory / "selected_config.json"
        if selected.is_file():
            sources[calibration.label] = selected
    sources[DEFAULT_SOURCE] = GAConfig()
    return sources


def _load(source: GAConfig | Path) -> GAConfig:
    if isinstance(source, GAConfig):
        return source
    return GAConfig.from_json(source.read_text(encoding="utf-8"))


def _run_section(instance: Instance) -> None:
    sources = config_sources()
    if streamlit.session_state.get(SOURCE_KEY) not in sources:
        streamlit.session_state.pop(SOURCE_KEY, None)
    label = streamlit.selectbox(
        "Partir de la configuración",
        list(sources),
        key=SOURCE_KEY,
        help="Carga los parámetros en el formulario; puede cambiarlos antes de ejecutar. Las "
        "calibraciones hechas en esta sesión aparecen aquí.",
    )
    try:
        base = _load(sources[label])
    except (ConfigError, OSError, json.JSONDecodeError):
        streamlit.error(f"«{label}» no es una configuración válida: {sys.exception()}")
        base = GAConfig()

    request = _config_form(base, label)
    if request is not None:
        batch = _execute(instance, request)
        if batch is not None:
            streamlit.session_state[_LAST_BATCH_KEY] = batch.number

    batches = session.ga_batches()
    last = streamlit.session_state.get(_LAST_BATCH_KEY)
    latest = next((batch for batch in batches if batch.number == last), None)
    if latest is not None:
        streamlit.divider()
        streamlit.subheader(f"Resultado: {latest.label}")
        streamlit.caption("Todos los lotes de la sesión están en «Resultados de esta sesión».")
        _batch_view(instance, latest, "latest")


@dataclass(frozen=True)
class _Request:
    config: GAConfig
    seeds: list[int]
    source: str


def _config_form(base: GAConfig, source: str) -> _Request | None:
    """Formulario con todos los parámetros; las claves dependen de la fuente para recargarlos."""
    prefix = f"ga-{source}"
    with streamlit.form(f"{prefix}-form"):
        streamlit.markdown("**Ejecuciones**")
        columns = streamlit.columns(4)
        runs = columns[0].number_input(
            "Número de ejecuciones",
            min_value=1,
            max_value=_MAX_RUNS,
            value=1,
            step=1,
            key=f"{prefix}-runs",
            help="Cada ejecución usa otra semilla: 1, 2, 3… a partir de la primera. Con 5 o "
            "más verá la mediana y la dispersión.",
        )
        first_seed = columns[1].number_input(
            "Primera semilla",
            min_value=0,
            value=1,
            step=1,
            key=f"{prefix}-seed",
            help="La misma semilla y configuración repiten exactamente el resultado.",
        )

        streamlit.markdown("**Población y selección**")
        columns = streamlit.columns(4)
        population_size = columns[0].number_input(
            "Tamaño de población",
            min_value=2,
            value=base.population_size,
            step=10,
            key=f"{prefix}-pop",
            help="Soluciones que conviven en cada generación. Más: explora más, pero cada "
            "generación tarda más.",
        )
        tournament_size = columns[1].number_input(
            "Tamaño del torneo",
            min_value=1,
            value=base.tournament_size,
            step=1,
            key=f"{prefix}-tour",
            help="Se sortean tantos individuos y gana el más barato. Más grande: presión "
            "selectiva más fuerte.",
        )
        elite_count = columns[2].number_input(
            "Élites",
            min_value=0,
            value=base.elite_count,
            step=1,
            key=f"{prefix}-elite",
            help="Mejores soluciones copiadas sin cambios a la siguiente generación.",
        )

        streamlit.markdown("**Cruce y mutación**")
        columns = streamlit.columns(4)
        crossover = columns[0].selectbox(
            "Cruce",
            [kind.value for kind in CrossoverKind],
            index=list(CrossoverKind).index(base.crossover),
            key=f"{prefix}-cx",
            help="uniform: cada gen viene de uno u otro padre al azar. one_point: un corte; "
            "la primera parte de un padre y el resto del otro.",
        )
        crossover_rate = columns[1].slider(
            "Probabilidad de cruce",
            0.0,
            1.0,
            base.crossover_rate,
            0.05,
            key=f"{prefix}-pc",
            help="Si no hay cruce, el hijo es copia de un padre (y luego muta).",
        )
        mutation = columns[2].selectbox(
            "Operador de mutación",
            [kind.value for kind in MutationKind],
            index=list(MutationKind).index(base.mutation),
            key=f"{prefix}-mut",
            help="bit_flip invierte genes; add activa antenas; remove desactiva antenas (la "
            "reparación busca otra cobertura); swap intercambia una activa por una inactiva; "
            "mixed elige uno de los anteriores al azar por hijo.",
        )
        mutation_rate = columns[3].number_input(
            "Tasa de mutación",
            min_value=0.0,
            max_value=1.0,
            value=base.mutation_rate,
            step=0.001,
            format="%.4f",
            key=f"{prefix}-pm",
            help="Movimientos esperados por hijo = 500 * tasa. Con 0,03 son unos 15.",
        )

        streamlit.markdown("**Población inicial y poda**")
        columns = streamlit.columns(4)
        greedy_fraction = columns[0].slider(
            "Proporción greedy",
            0.0,
            1.0,
            base.greedy_fraction,
            0.05,
            key=f"{prefix}-greedy",
            help="Fracción de la población inicial construida con una heurística voraz "
            "aleatorizada; el resto parte de antenas al azar.",
        )
        rcl_alpha = columns[1].slider(
            "Alfa de la lista restringida (RCL)",
            0.0,
            1.0,
            base.rcl_alpha,
            0.05,
            key=f"{prefix}-rcl",
            help="0: la heurística voraz siempre elige la mejor antena; 1: elige al azar.",
        )
        random_activation = columns[2].number_input(
            "Activación aleatoria inicial",
            min_value=0.0,
            max_value=1.0,
            value=base.random_activation,
            step=0.005,
            format="%.3f",
            key=f"{prefix}-act",
            help="Probabilidad de que cada antena empiece activa en los individuos aleatorios.",
        )
        prune_order = columns[3].selectbox(
            "Orden de eliminación de redundancias",
            [order.value for order in PruneOrder],
            index=list(PruneOrder).index(base.prune_order),
            key=f"{prefix}-prune",
            help="En qué orden se prueba quitar antenas que sobran: las más caras primero, "
            "las de peor costo por cliente primero, o al azar.",
        )
        deduplicate = streamlit.checkbox(
            "Descartar descendientes duplicados",
            value=base.deduplicate,
            key=f"{prefix}-dedup",
            help="Mantiene la variedad de la población.",
        )

        streamlit.markdown("**Criterios de parada** (se detiene con el primero que se cumpla)")
        columns = streamlit.columns(4)
        use_stall = columns[0].checkbox(
            "Generaciones sin mejora",
            value=base.stall_generations is not None,
            key=f"{prefix}-use-stall",
        )
        stall = columns[0].number_input(
            "Generaciones",
            min_value=1,
            value=base.stall_generations or 200,
            step=10,
            key=f"{prefix}-stall",
            label_visibility="collapsed",
        )
        use_generations = columns[1].checkbox(
            "Máximo de generaciones",
            value=base.max_generations is not None,
            key=f"{prefix}-use-gen",
        )
        generations = columns[1].number_input(
            "Generaciones máximas",
            min_value=0,
            value=base.max_generations or 2000,
            step=100,
            key=f"{prefix}-gen",
            label_visibility="collapsed",
        )
        use_evaluations = columns[2].checkbox(
            "Máximo de evaluaciones",
            value=base.max_evaluations is not None,
            key=f"{prefix}-use-eval",
        )
        evaluations = columns[2].number_input(
            "Evaluaciones máximas",
            min_value=1,
            value=base.max_evaluations or 100_000,
            step=10_000,
            key=f"{prefix}-eval",
            label_visibility="collapsed",
        )
        use_time = columns[3].checkbox(
            "Límite de tiempo (s)", value=base.time_limit is not None, key=f"{prefix}-use-time"
        )
        time_limit = columns[3].number_input(
            "Segundos",
            min_value=1.0,
            value=base.time_limit or 30.0,
            step=5.0,
            key=f"{prefix}-time",
            label_visibility="collapsed",
        )
        submitted = streamlit.form_submit_button("Ejecutar el GA", type="primary")

    if not submitted:
        return None
    values: dict[str, object] = {
        "population_size": int(population_size),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "mutation": mutation,
        "tournament_size": int(tournament_size),
        "elite_count": int(elite_count),
        "crossover": crossover,
        "greedy_fraction": float(greedy_fraction),
        "rcl_alpha": float(rcl_alpha),
        "random_activation": float(random_activation),
        "prune_order": prune_order,
        "deduplicate": deduplicate,
        "stall_generations": int(stall) if use_stall else None,
        "max_generations": int(generations) if use_generations else None,
        "max_evaluations": int(evaluations) if use_evaluations else None,
        "time_limit": float(time_limit) if use_time else None,
    }
    try:
        config = GAConfig.from_dict(values)
    except ConfigError:
        streamlit.error(f"Configuración inválida: {sys.exception()}")
        return None
    start = int(first_seed)
    return _Request(config, list(range(start, start + int(runs))), short_source(source))


def short_source(label: str) -> str:
    """Nombre corto de una fuente de configuración, para etiquetas de lotes y gráficos."""
    for prefix in ("Calibración de la sesión: ",):
        label = label.removeprefix(prefix)
    return label.removesuffix(" (proyecto)")


class _LiveProgress:
    """Callback del GA que redibuja el progreso como mucho cada ``_REDRAW_SECONDS``."""

    def __init__(self, chart_slot: DeltaGenerator, text: DeltaGenerator, prefix: str) -> None:
        self._chart = chart_slot
        self._text = text
        self._prefix = prefix
        self._generations: list[int] = []
        self._best: list[float] = []
        self._mean: list[float] = []
        self._last_draw = 0.0

    def __call__(self, report: GenerationReport) -> None:
        self._generations.append(report.generation)
        self._best.append(report.best_so_far)
        self._mean.append(report.population_mean)
        now = time.monotonic()
        if now - self._last_draw >= _REDRAW_SECONDS:
            self._last_draw = now
            self._text.markdown(
                f"{self._prefix} · generación **{spanish(report.generation)}** · "
                f"evaluaciones {spanish(report.evaluations)} · "
                f"mejor **{spanish(report.best_so_far)}** · {spanish(report.elapsed_seconds, 1)} s"
            )
            self._chart.plotly_chart(
                charts.ga_progress(self._generations, self._best, self._mean),
                key=f"ga-live-{self._prefix}-{len(self._generations)}",
            )


def _execute(instance: Instance, request: _Request) -> session.GABatch | None:
    _, problem = current_problem()
    total = len(request.seeds)
    progress = streamlit.progress(0.0, text=f"0 de {total} ejecuciones")
    text = streamlit.empty()
    live = streamlit.empty()
    streamlit.caption(f":material/visibility: **Cómo leerlo.** {_LIVE_GUIDE}")
    finished = streamlit.empty()
    results: list[GAResult] = []
    for index, seed in enumerate(request.seeds):
        label = f"Ejecución {index + 1} de {total} (semilla {seed})"
        result = run_ga(problem, request.config, seed, _LiveProgress(live, text, label))
        if instance.uncovered_customers(list(result.evaluation.antenna_ids)):
            streamlit.error(f"La semilla {seed} reportó una solución que no cubre a todos.")
            return None
        results.append(result)
        progress.progress((index + 1) / total, text=f"{index + 1} de {total} ejecuciones")
        with finished.container():
            show_table(
                [
                    {
                        "Semilla": str(run.seed),
                        "Costo": run.evaluation.cost,
                        "Tiempo (s)": run.elapsed_seconds,
                    }
                    for run in results
                ],
                {"Costo": 0, "Tiempo (s)": 1},
            )
    progress.empty()
    text.empty()
    live.empty()
    finished.empty()
    batch = session.add_ga_batch(request.source, results)
    group = batch.group()
    streamlit.success(
        f"{batch.label}: mejor costo {spanish(group.best.cost)}"
        + (f", mediana {spanish(group.median_cost)}." if total > 1 else ".")
    )
    return batch


def _batch_view(instance: Instance, batch: session.GABatch, key: str) -> None:
    group = batch.group()
    if len(batch.results) > 1:
        group_summary(group, f"{key}-{batch.number}")
        streamlit.markdown("**Detalle de una ejecución**")
        by_seed = {f"Semilla {result.seed}": result for result in batch.results}
        best = f"Semilla {group.best.seed}"
        chosen = streamlit.selectbox(
            "Ejecución",
            list(by_seed),
            index=list(by_seed).index(best),
            key=f"ga-detail-{key}-{batch.number}",
            help="Por defecto, la de menor costo.",
        )
        _run_detail(instance, by_seed[chosen], f"{key}-{batch.number}")
    else:
        _run_detail(instance, batch.results[0], f"{key}-{batch.number}")
    _downloads(batch, key)


def group_summary(group: GAGroup, key: str) -> None:
    """Métricas, distribución, calidad vs tiempo y tabla de un grupo de ejecuciones."""
    costs = group.costs
    columns = streamlit.columns(4)
    columns[0].metric("Ejecuciones", len(costs))
    columns[1].metric("Mejor costo", spanish(min(costs)), help="Más bajo es mejor.")
    columns[2].metric(
        "Mediana", spanish(group.median_cost), help="La mitad de las semillas logra esto o menos."
    )
    columns[3].metric("Peor costo", spanish(max(costs)))
    columns = streamlit.columns(4)
    columns[0].metric(
        "Desviación estándar",
        spanish(group.std_cost, 1),
        help="Cuánto varía el costo entre semillas. Más bajo: más confiable.",
    )
    columns[1].metric(
        "Rango", percent((max(costs) - min(costs)) / min(costs)), help="(peor - mejor) / mejor."
    )
    columns[2].metric("Tiempo medio por ejecución", duration(group.mean_seconds))
    columns[3].metric("Antenas de la mejor", group.best.num_antennas)

    left, right = streamlit.columns(2)
    with left:
        chart(
            charts.cost_distribution({group.label: costs}, {}),
            f"ga-distribution-{key}",
            DISTRIBUTION_GUIDE,
        )
    if group.curves:
        with right:
            chart(
                charts.quality_vs_time({group.label: group.curves}, []),
                f"ga-quality-{key}",
                QUALITY_GUIDE,
            )
    show_table(records_table(group.records), {"Costo": 0})


def _run_detail(instance: Instance, result: GAResult, key: str) -> None:
    evaluation = result.evaluation
    covered = instance.num_customers - len(
        instance.uncovered_customers(list(evaluation.antenna_ids))
    )
    columns = streamlit.columns(4)
    columns[0].metric("Costo", spanish(evaluation.cost), help="Más bajo es mejor.")
    columns[1].metric("Antenas instaladas", evaluation.num_antennas)
    columns[2].metric("Clientes cubiertos", f"{covered}/{instance.num_customers}")
    columns[3].metric("Tiempo", duration(result.elapsed_seconds))
    columns = streamlit.columns(4)
    columns[0].metric("Generaciones", spanish(result.generations))
    columns[1].metric("Soluciones evaluadas", spanish(result.evaluations))
    columns[2].metric(
        "Mejor encontrada en",
        f"gen. {spanish(result.generation_of_best)}",
        help="Después de esta generación no hubo mejoras.",
    )
    columns[3].metric(
        "Se detuvo por",
        _STOP_REASONS.get(result.stop_reason.value, "—"),
        help="sin mejora: alcanzó las generaciones sin mejora; evaluaciones, generaciones o "
        "tiempo: agotó ese límite.",
    )
    left, right = streamlit.columns(2)
    with left:
        chart(charts.ga_history(result.history), f"ga-history-{key}-{result.seed}", HISTORY_GUIDE)
    with right:
        chart(
            charts.ga_diversity(result.history),
            f"ga-diversity-{key}-{result.seed}",
            DIVERSITY_GUIDE,
        )
    streamlit.markdown(
        "**Antenas instaladas (j):** "
        + ", ".join(str(antenna) for antenna in evaluation.antenna_ids)
    )


_STOP_REASONS = {
    "stall": "sin mejora",
    "max_generations": "generaciones",
    "max_evaluations": "evaluaciones",
    "time_limit": "tiempo",
}


def _downloads(batch: session.GABatch, key: str) -> None:
    left, right = streamlit.columns(2)
    left.download_button(
        "Descargar ejecuciones (JSON Lines)",
        "".join(json.dumps(result.to_record()) + "\n" for result in batch.results),
        file_name=f"ga_lote{batch.number}.jsonl",
        mime="application/jsonl",
        key=f"ga-download-{key}-{batch.number}",
        help="Un registro por semilla, con historial y antenas elegidas.",
    )
    right.download_button(
        "Descargar configuración (JSON)",
        json.dumps(batch.results[0].config.to_dict(), indent=2),
        file_name=f"ga_config_{batch.results[0].config.config_id()}.json",
        mime="application/json",
        key=f"ga-config-{key}-{batch.number}",
        help=f"Guárdela en `{PATHS.configs}/` para que aparezca en «Partir de la configuración».",
    )


def _session_section(instance: Instance) -> None:
    batches = session.ga_batches()
    if not batches:
        streamlit.info(
            "Todavía no ejecutó el GA en esta sesión. Use la pestaña «Ejecutar». Cada envío "
            "del formulario crea un **lote**: una o más semillas con la misma configuración."
        )
        return
    streamlit.markdown("**Lotes de esta sesión**")
    groups = [batch.group() for batch in batches]
    show_table(
        [
            {
                "Lote": batch.label,
                "Ejecuciones": len(batch.results),
                "Mejor": group.best.cost,
                "Mediana": group.median_cost,
                "Desv.": group.std_cost,
                "Tiempo medio": duration(group.mean_seconds),
                "Configuración": batch.results[0].config.config_id(),
            }
            for batch, group in zip(batches, groups, strict=True)
        ],
        {"Mejor": 0, "Mediana": 0, "Desv.": 1},
    )
    streamlit.caption("Para comparar lotes entre sí y con Gurobi, vaya a la página «Comparación».")
    labels = [batch.label for batch in batches]
    chosen = streamlit.selectbox("Ver lote", labels, index=len(labels) - 1, key="ga-session-batch")
    _batch_view(instance, batches[labels.index(chosen)], "session")


@dataclass(frozen=True)
class SavedGroup:
    title: str
    description: str
    records: Path
    curves: Path
    statistics: Path | None


SAVED_GROUPS = (
    SavedGroup(
        "Evaluación final (30 semillas)",
        "La configuración calibrada ejecutada con 30 semillas que la calibración nunca usó. "
        "Mide cómo rinde de verdad, sin el sesgo de haberla elegido por sus resultados. Cada "
        "ejecución se detiene tras 200 generaciones sin mejora o 100.000 evaluaciones.",
        PATHS.calibration / "runs_final.jsonl",
        PATHS.calibration / "final_convergence.csv",
        PATHS.calibration / "final_statistics.json",
    ),
    SavedGroup(
        "Ejecuciones de 30 min (6 semillas)",
        "La configuración calibrada sin parada por estancamiento ni por evaluaciones: corre "
        "30 minutos por semilla. Responde si darle mucho más tiempo mejora la solución.",
        PATHS.long_runs / "summary.jsonl",
        PATHS.long_runs / "improvements.csv",
        None,
    ),
)


def saved_group(saved: SavedGroup) -> GAGroup | None:
    """El grupo guardado, o ``None`` si sus archivos no existen o no se pueden leer."""
    try:
        records = ga_records(saved.records)
        curves = convergence(saved.curves) if saved.curves.is_file() else []
    except RESULT_ERRORS:
        return None
    if not records:
        return None
    return GAGroup(
        f"GA · {saved.title.lower()}", tuple(records), tuple(extend_to_run_end(curves, records))
    )


def _saved_section() -> None:
    streamlit.markdown(
        "Resultados obtenidos con la terminal (`set-cover-ga`) y guardados en el proyecto. "
        "Sirven de referencia para sus propias ejecuciones."
    )
    titles = [saved.title for saved in SAVED_GROUPS]
    chosen = streamlit.radio("Resultado", titles, horizontal=True, key="ga-saved-choice")
    saved = SAVED_GROUPS[titles.index(chosen)]
    streamlit.caption(saved.description)
    group = saved_group(saved)
    if group is None:
        streamlit.info(f"No hay resultados legibles en `{saved.records}`.")
        return
    group_summary(group, f"saved-{titles.index(chosen)}")
    if saved.statistics is not None:
        try:
            config = json_object(saved.statistics).get("config")
        except RESULT_ERRORS:
            config = None
        if isinstance(config, dict):
            with streamlit.expander("Configuración usada"):
                streamlit.json(config)


def records_table(records: Sequence[GARunRecord]) -> list[dict[str, object]]:
    return [
        {
            "Semilla": str(record.seed),
            "Costo": record.cost,
            "Antenas": record.num_antennas,
            "Tiempo": duration(record.seconds),
            "Hasta la mejor": duration(record.seconds_to_best),
            "Evaluaciones": record.evaluations,
            "Generaciones": record.generations,
            "Parada": _STOP_REASONS.get(record.stop_reason, record.stop_reason),
        }
        for record in sorted(records, key=lambda record: (record.cost, record.seed))
    ]
