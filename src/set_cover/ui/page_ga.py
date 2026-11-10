"""Página del algoritmo genético: ejecutar con progreso en vivo y revisar resultados guardados."""

import json
import statistics
import sys
import time
from pathlib import Path

import streamlit
from streamlit.delta_generator import DeltaGenerator

from set_cover.ga.config import ConfigError, CrossoverKind, GAConfig, MutationKind, PruneOrder
from set_cover.ga.engine import GAResult, GenerationReport, run_ga
from set_cover.instance import Instance
from set_cover.ui import charts
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    add_session_ga_run,
    convergence,
    current_problem,
    ga_records,
    json_object,
    session_ga_runs,
    session_run_label,
)
from set_cover.ui.data import GARunRecord, duration, median_series

_PREFERRED_CONFIG = "calibrated.json"
_REDRAW_SECONDS = 0.5


def render() -> None:
    instance, _ = current_problem()
    streamlit.title("Algoritmo genético")
    streamlit.markdown(
        "Cromosoma binario de 500 genes (uno por antena). Cada hijo pasa por **reparación** "
        "voraz y **eliminación de redundancias**, así que toda la población es factible y el "
        "fitness es el costo. Selección por torneo, cruce uniforme o de un punto, mutación "
        "configurable y elitismo. Detalle en `docs/genetic_algorithm.md`."
    )
    run_tab, final_tab, long_tab = streamlit.tabs(
        ["Ejecutar", "Evaluación final (30 semillas)", "Ejecuciones de 30 min"]
    )
    with run_tab:
        _run_section(instance)
    with final_tab:
        _final_section()
    with long_tab:
        _long_runs_section()


def _config_files() -> list[Path]:
    files = sorted(PATHS.configs.glob("*.json"))
    return sorted(files, key=lambda path: path.name != _PREFERRED_CONFIG)


def _run_section(instance: Instance) -> None:
    files = _config_files()
    if not files:
        streamlit.warning(
            f"No hay configuraciones en `{PATHS.configs}`; se usan los valores por defecto."
        )
    source = streamlit.selectbox(
        "Configuración base",
        [path.name for path in files] or ["valores por defecto"],
        help="Carga los parámetros; puede cambiarlos antes de ejecutar.",
    )
    path = PATHS.configs / source if files else None
    try:
        base = GAConfig() if path is None else GAConfig.from_json(path.read_text(encoding="utf-8"))
    except (ConfigError, OSError, json.JSONDecodeError):
        streamlit.error(f"`{path}` no es una configuración válida: {sys.exception()}")
        base = GAConfig()

    config, seed, submitted = _config_form(base, source)
    if submitted and config is not None:
        _execute(instance, config, seed)

    runs = session_ga_runs()
    if not runs:
        return
    streamlit.divider()
    labels = [session_run_label(index, run) for index, run in enumerate(runs)]
    chosen = streamlit.selectbox("Resultado de esta sesión", labels, index=len(labels) - 1)
    result = runs[labels.index(chosen)]
    _show_result(instance, result)
    with streamlit.expander(f"Todas las ejecuciones de esta sesión ({len(runs)})"):
        streamlit.dataframe(
            [
                {
                    "Ejecución": label,
                    "Costo": run.evaluation.cost,
                    "Antenas": run.evaluation.num_antennas,
                    "Tiempo (s)": round(run.elapsed_seconds, 2),
                    "Mutación": run.config.mutation.value,
                    "Configuración": run.config.config_id(),
                }
                for label, run in zip(labels, runs, strict=True)
            ],
            hide_index=True,
        )


def _config_form(base: GAConfig, source: str) -> tuple[GAConfig | None, int, bool]:
    """Formulario con todos los parámetros; las claves dependen de la fuente para reiniciarlos."""
    prefix = f"ga-{source}"
    with streamlit.form(f"{prefix}-form"):
        streamlit.markdown("**Población y selección**")
        columns = streamlit.columns(4)
        population_size = columns[0].number_input(
            "Tamaño de población",
            min_value=2,
            value=base.population_size,
            step=10,
            key=f"{prefix}-pop",
        )
        tournament_size = columns[1].number_input(
            "Tamaño del torneo",
            min_value=1,
            value=base.tournament_size,
            step=1,
            key=f"{prefix}-tour",
        )
        elite_count = columns[2].number_input(
            "Élites", min_value=0, value=base.elite_count, step=1, key=f"{prefix}-elite"
        )
        seed = columns[3].number_input(
            "Semilla", min_value=0, value=1, step=1, key=f"{prefix}-seed"
        )

        streamlit.markdown("**Cruce y mutación**")
        columns = streamlit.columns(4)
        crossover = columns[0].selectbox(
            "Cruce",
            [kind.value for kind in CrossoverKind],
            index=list(CrossoverKind).index(base.crossover),
            key=f"{prefix}-cx",
        )
        crossover_rate = columns[1].slider(
            "Probabilidad de cruce", 0.0, 1.0, base.crossover_rate, 0.05, key=f"{prefix}-pc"
        )
        mutation = columns[2].selectbox(
            "Operador de mutación",
            [kind.value for kind in MutationKind],
            index=list(MutationKind).index(base.mutation),
            key=f"{prefix}-mut",
            help="bit_flip invierte genes; add activa; remove desactiva (la reparación busca "
            "otra cobertura); swap intercambia; mixed elige uno al azar por hijo.",
        )
        mutation_rate = columns[3].number_input(
            "Tasa de mutación",
            min_value=0.0,
            max_value=1.0,
            value=base.mutation_rate,
            step=0.001,
            format="%.4f",
            key=f"{prefix}-pm",
            help="Movimientos esperados por hijo = 500 * tasa.",
        )

        streamlit.markdown("**Población inicial y poda**")
        columns = streamlit.columns(4)
        greedy_fraction = columns[0].slider(
            "Proporción greedy", 0.0, 1.0, base.greedy_fraction, 0.05, key=f"{prefix}-greedy"
        )
        rcl_alpha = columns[1].slider(
            "Alfa de la lista restringida (RCL)",
            0.0,
            1.0,
            base.rcl_alpha,
            0.05,
            key=f"{prefix}-rcl",
        )
        random_activation = columns[2].number_input(
            "Activación aleatoria inicial",
            min_value=0.0,
            max_value=1.0,
            value=base.random_activation,
            step=0.005,
            format="%.3f",
            key=f"{prefix}-act",
        )
        prune_order = columns[3].selectbox(
            "Orden de eliminación de redundancias",
            [order.value for order in PruneOrder],
            index=list(PruneOrder).index(base.prune_order),
            key=f"{prefix}-prune",
        )
        deduplicate = streamlit.checkbox(
            "Descartar descendientes duplicados", value=base.deduplicate, key=f"{prefix}-dedup"
        )

        streamlit.markdown("**Criterios de parada** (al menos uno)")
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
        return None, int(seed), False
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
        return GAConfig.from_dict(values), int(seed), True
    except ConfigError:
        streamlit.error(f"Configuración inválida: {sys.exception()}")
        return None, int(seed), True


class _LiveProgress:
    """Callback del GA que redibuja el progreso como mucho cada ``_REDRAW_SECONDS``."""

    def __init__(self, chart: DeltaGenerator, text: DeltaGenerator) -> None:
        self._chart = chart
        self._text = text
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
                f"Generación **{report.generation:,}** · evaluaciones {report.evaluations:,} · "
                f"mejor **{report.best_so_far:,.0f}** · {report.elapsed_seconds:.1f} s"
            )
            self._chart.plotly_chart(
                charts.ga_progress(self._generations, self._best, self._mean),
                key=f"ga-live-{len(self._generations)}",
            )


def _execute(instance: Instance, config: GAConfig, seed: int) -> None:
    _, problem = current_problem()
    text = streamlit.empty()
    chart = streamlit.empty()
    text.markdown("Iniciando…")
    result = run_ga(problem, config, seed, _LiveProgress(chart, text))
    text.empty()
    chart.empty()
    ids = list(result.evaluation.antenna_ids)
    if instance.uncovered_customers(ids):
        streamlit.error("La solución reportada no cubre a todos los clientes.")
        return
    add_session_ga_run(result)
    streamlit.success(
        f"Costo {result.evaluation.cost:,.0f} con {len(ids)} antenas en "
        f"{result.elapsed_seconds:.1f} s ({result.stop_reason.value})."
    )


def _show_result(instance: Instance, result: GAResult) -> None:
    evaluation = result.evaluation
    covered = instance.num_customers - len(
        instance.uncovered_customers(list(evaluation.antenna_ids))
    )
    columns = streamlit.columns(4)
    columns[0].metric("Costo", f"{evaluation.cost:,.0f}")
    columns[1].metric("Antenas", evaluation.num_antennas)
    columns[2].metric("Clientes cubiertos", f"{covered}/{instance.num_customers}")
    columns[3].metric("Tiempo", f"{result.elapsed_seconds:.2f} s")
    columns = streamlit.columns(4)
    columns[0].metric("Generaciones", f"{result.generations:,}")
    columns[1].metric("Evaluaciones", f"{result.evaluations:,}")
    columns[2].metric("Mejor encontrada en", f"gen. {result.generation_of_best:,}")
    columns[3].metric("Motivo de parada", result.stop_reason.value)
    left, right = streamlit.columns(2)
    left.plotly_chart(charts.ga_history(result.history), key=f"ga-history-{id(result)}")
    right.plotly_chart(charts.ga_diversity(result.history), key=f"ga-diversity-{id(result)}")
    streamlit.markdown(
        "**Antenas (base 1):** " + ", ".join(str(antenna) for antenna in evaluation.antenna_ids)
    )
    record = result.to_record()
    left, right = streamlit.columns(2)
    left.download_button(
        "Descargar resultado (JSON)",
        json.dumps(record, indent=2),
        file_name=f"ga_seed{result.seed}_{result.config.config_id()}.json",
        mime="application/json",
        key=f"ga-download-{id(result)}",
    )
    right.download_button(
        "Descargar configuración (JSON)",
        json.dumps(result.config.to_dict(), indent=2),
        file_name=f"ga_config_{result.config.config_id()}.json",
        mime="application/json",
        key=f"ga-config-{id(result)}",
    )


def _final_section() -> None:
    statistics_path = PATHS.calibration / "final_statistics.json"
    runs_path = PATHS.calibration / "runs_final.jsonl"
    try:
        stats = json_object(statistics_path)
        records = ga_records(runs_path)
        curves = convergence(PATHS.calibration / "final_convergence.csv")
    except RESULT_ERRORS:
        streamlit.info(
            f"No hay evaluación final legible en `{PATHS.calibration}`: {sys.exception()}"
        )
        return
    if not records:
        streamlit.info("La evaluación final no tiene ejecuciones.")
        return
    costs = [record.cost for record in records]
    seconds = [record.seconds for record in records]
    feasible = stats.get("feasible_rate")
    streamlit.markdown(
        "La configuración calibrada con 30 semillas que no se usaron en la calibración. "
        "Así se mide la robustez: un GA es aleatorio y una sola ejecución dice poco."
    )
    columns = streamlit.columns(4)
    columns[0].metric("Ejecuciones", len(records))
    columns[1].metric("Factibles", f"{feasible:.0%}" if isinstance(feasible, (int, float)) else "—")
    columns[2].metric("Mediana", f"{statistics.median(costs):,.0f}")
    columns[3].metric("Mejor", f"{min(costs):,.0f}")
    columns = streamlit.columns(4)
    columns[0].metric("Media", f"{sum(costs) / len(costs):,.1f}")
    columns[1].metric("Peor", f"{max(costs):,.0f}")
    columns[2].metric(
        "Desviación estándar", f"{statistics.stdev(costs) if len(costs) > 1 else 0.0:,.1f}"
    )
    columns[3].metric("Tiempo medio", f"{sum(seconds) / len(seconds):.2f} s")

    left, right = streamlit.columns(2)
    left.plotly_chart(
        charts.cost_distribution({"Evaluación final": costs}, {}), key="ga-final-distribution"
    )
    if curves:
        right.plotly_chart(
            charts.quality_vs_time(curves, median_series(curves), []), key="ga-final-convergence"
        )
    streamlit.dataframe(_records_table(records), hide_index=True)
    config = stats.get("config")
    if isinstance(config, dict):
        with streamlit.expander("Configuración evaluada"):
            streamlit.json(config)


def _long_runs_section() -> None:
    try:
        records = ga_records(PATHS.long_runs / "summary.jsonl")
        curves = convergence(PATHS.long_runs / "improvements.csv")
    except RESULT_ERRORS:
        streamlit.info(
            f"No hay ejecuciones largas legibles en `{PATHS.long_runs}`: {sys.exception()}"
        )
        return
    if not records:
        streamlit.info("No hay ejecuciones largas.")
        return
    streamlit.markdown(
        "La configuración calibrada sin parada por estancamiento ni presupuesto de "
        "evaluaciones, limitada a 30 minutos por semilla. Muestra si más tiempo mejora la solución."
    )
    if curves:
        streamlit.plotly_chart(charts.quality_vs_time(curves, None, []), key="ga-long-convergence")
    streamlit.dataframe(_records_table(records), hide_index=True)


def _records_table(records: list[GARunRecord]) -> list[dict[str, object]]:
    return [
        {
            "Semilla": record.seed,
            "Costo": record.cost,
            "Antenas": record.num_antennas,
            "Tiempo": duration(record.seconds),
            "Hasta la mejor": duration(record.seconds_to_best),
            "Evaluaciones": record.evaluations,
            "Generaciones": record.generations,
            "Parada": record.stop_reason,
        }
        for record in sorted(records, key=lambda record: (record.cost, record.seed))
    ]
