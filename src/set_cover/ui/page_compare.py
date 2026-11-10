"""Página de comparación: el GA frente al solver exacto, con las mismas métricas que el CLI."""

import json
import sys

import streamlit

from set_cover.compare import Comparison, ComparisonError, MethodResult, ga_results
from set_cover.instance import Instance
from set_cover.ui import charts
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    convergence,
    current_problem,
    exact_run,
    exact_run_directories,
    ga_records,
    session_ga_runs,
    session_run_label,
)
from set_cover.ui.data import (
    ExactRun,
    exact_method_result,
    ga_record_result,
    ga_session_result,
    jaccard,
    median_series,
)

# Columnas de ``Comparison.rows`` y su encabezado en la tabla.
_COLUMNS = {
    "method": "Método",
    "cost": "Costo",
    "num_antennas": "Antenas",
    "covered_customers": "Clientes cubiertos",
    "seconds": "Tiempo (s)",
    "solver_best_bound": "Cota inferior",
    "solver_gap_pct": "Brecha del solver (%)",
    "diff_vs_best_known_pct": "Dif. vs mejor conocida (%)",
    "gap_vs_optimum_pct": "Gap vs óptimo (%)",
    "gap_upper_bound_pct": "Cota sup. del gap (%)",
    "note": "Nota",
}


def render() -> None:
    instance, _ = current_problem()
    streamlit.title("Comparación: GA vs método exacto")
    streamlit.markdown(
        "Cada solución se verifica contra la matriz original (500 restricciones y costo) antes "
        "de compararla. El GA nunca recibe soluciones del solver."
    )

    exact_runs, results = _choose_methods(instance)
    if not results:
        streamlit.info("Elija al menos un método.")
        return
    comparison = Comparison(results)
    _interpretation(comparison)
    _table(comparison)

    cost_tab, time_tab, distribution_tab, antennas_tab = streamlit.tabs(
        ["Costo", "Calidad vs tiempo", "Distribución del GA", "Antenas elegidas"]
    )
    with cost_tab:
        streamlit.plotly_chart(
            charts.cost_bars(results, comparison.best_bound), key="compare-costs"
        )
    with time_tab:
        _quality_vs_time(exact_runs)
    with distribution_tab:
        _distribution(results)
    with antennas_tab:
        _antennas(results)

    left, middle, right = streamlit.columns(3)
    left.download_button(
        "Descargar CSV", comparison.to_csv(), "comparison.csv", "text/csv", key="compare-csv"
    )
    middle.download_button(
        "Descargar JSON",
        json.dumps(comparison.to_dict(), indent=2),
        "comparison.json",
        "application/json",
        key="compare-json",
    )
    right.download_button(
        "Descargar Markdown",
        comparison.to_markdown(),
        "comparison.md",
        "text/markdown",
        key="compare-md",
    )


def _choose_methods(instance: Instance) -> tuple[list[ExactRun], list[MethodResult]]:
    directories = exact_run_directories()
    by_label = {str(directory): directory for directory in directories}
    saved = [str(d) for d in directories if d.parent == PATHS.saved_solver_runs]
    session_runs = session_ga_runs()
    session_labels = [session_run_label(index, run) for index, run in enumerate(session_runs)]

    left, right = streamlit.columns(2)
    chosen_exact = left.multiselect(
        "Ejecuciones del solver exacto", list(by_label), default=saved or list(by_label)[:1]
    )
    include_final = right.checkbox("GA: evaluación final (mejor y mediana de 30)", value=True)
    include_long = right.checkbox("GA: mejor de las ejecuciones de 30 min", value=True)
    chosen_session = right.multiselect(
        "GA: ejecuciones de esta sesión",
        session_labels,
        default=session_labels[-1:],
        disabled=not session_labels,
        placeholder="Ninguna todavía" if not session_labels else "Elija",
    )

    results: list[MethodResult] = []
    if include_final:
        try:
            results.extend(ga_results(instance, PATHS.calibration / "final_statistics.json"))
        except (*RESULT_ERRORS, ComparisonError):
            streamlit.warning(f"Evaluación final no disponible: {sys.exception()}")
    if include_long:
        try:
            records = ga_records(PATHS.long_runs / "summary.jsonl")
            if records:
                results.append(
                    ga_record_result(instance, records, f"GA 30 min (mejor de {len(records)})")
                )
        except RESULT_ERRORS:
            streamlit.warning(f"Ejecuciones de 30 min no disponibles: {sys.exception()}")
    for label in chosen_session:
        index = session_labels.index(label)
        results.append(ga_session_result(instance, session_runs[index], label))

    exact_runs: list[ExactRun] = []
    for label in chosen_exact:
        try:
            run = exact_run(by_label[label])
        except RESULT_ERRORS:
            streamlit.warning(f"No se pudo leer `{label}`: {sys.exception()}")
            continue
        exact_runs.append(run)
        result = exact_method_result(instance, run)
        if result is None:
            streamlit.warning(f"`{label}` todavía no tiene incumbente.")
        else:
            results.append(result)
    return exact_runs, results


def _interpretation(comparison: Comparison) -> None:
    best_known = comparison.best_known
    bound = comparison.best_bound
    best_method = min(comparison.results, key=lambda result: result.cost).method
    if comparison.optimum is not None:
        streamlit.success(
            f"**Óptimo demostrado: {comparison.optimum:,.0f}.** La columna «Gap vs óptimo» es "
            "`(Z - Z*) / Z*`."
        )
    elif bound is not None:
        streamlit.info(
            f"**Ningún método demostró optimalidad.** El óptimo está entre la mejor cota "
            f"inferior probada, **{bound:,.0f}**, y la mejor solución conocida, "
            f"**{best_known:,.0f}** ({best_method}). Por eso no hay «Gap vs óptimo»: se reporta "
            "la diferencia con la mejor conocida y `(Z - cota) / cota`, que acota el gap real "
            "por arriba."
        )
    else:
        streamlit.info(
            f"Sin cota inferior: incluya una ejecución del solver exacto. Mejor solución "
            f"conocida: **{best_known:,.0f}** ({best_method})."
        )


def _table(comparison: Comparison) -> None:
    rows = [{header: row.get(key) for key, header in _COLUMNS.items()} for row in comparison.rows()]
    number = streamlit.column_config.NumberColumn(format="localized")
    percent = streamlit.column_config.NumberColumn(format="%.2f")
    streamlit.dataframe(
        rows,
        hide_index=True,
        column_config={
            "Costo": number,
            "Cota inferior": number,
            "Tiempo (s)": streamlit.column_config.NumberColumn(format="%.1f"),
            "Brecha del solver (%)": percent,
            "Dif. vs mejor conocida (%)": percent,
            "Gap vs óptimo (%)": percent,
            "Cota sup. del gap (%)": percent,
        },
    )
    streamlit.caption(
        "Tiempo: el GA reporta segundos por ejecución (un hilo); el solver, tiempo acumulado de "
        "todas sus sesiones (todos los hilos de la máquina)."
    )


def _quality_vs_time(exact_runs: list[ExactRun]) -> None:
    try:
        curves = convergence(PATHS.calibration / "final_convergence.csv")
    except RESULT_ERRORS:
        curves = []
    streamlit.plotly_chart(
        charts.quality_vs_time(curves, median_series(curves) if curves else None, exact_runs),
        key="compare-quality-time",
    )
    streamlit.caption(
        "Curvas tenues: cada semilla de la evaluación final del GA. Línea gruesa: mediana entre "
        "semillas. Solver: incumbente (continua) y cota inferior (discontinua). El óptimo está "
        "siempre entre la cota más alta y la mejor solución."
    )


def _distribution(results: list[MethodResult]) -> None:
    groups: dict[str, list[float]] = {}
    for title, path in (
        ("Evaluación final (≈10 s c/u)", PATHS.calibration / "runs_final.jsonl"),
        ("30 min c/u", PATHS.long_runs / "summary.jsonl"),
    ):
        try:
            records = ga_records(path)
        except RESULT_ERRORS:
            continue
        if records:
            groups[f"{title}, n={len(records)}"] = [record.cost for record in records]
    if not groups:
        streamlit.info("No hay ejecuciones del GA guardadas.")
        return
    references = {result.method: result.cost for result in results if result.best_bound is not None}
    streamlit.plotly_chart(charts.cost_distribution(groups, references), key="compare-distribution")


def _antennas(results: list[MethodResult]) -> None:
    selections = {result.method: result.antenna_ids for result in results if result.antenna_ids}
    if len(selections) < 2:
        streamlit.info("Se necesitan al menos dos soluciones con antenas para compararlas.")
        return
    streamlit.plotly_chart(charts.antenna_overlap(selections), key="compare-overlap")
    names = list(selections)
    streamlit.markdown("**Similitud de Jaccard** (antenas en común / antenas en la unión)")
    streamlit.dataframe(
        [
            {
                "Método": first,
                **{
                    second: round(jaccard(selections[first], selections[second]), 2)
                    for second in names
                },
            }
            for first in names
        ],
        hide_index=True,
    )
