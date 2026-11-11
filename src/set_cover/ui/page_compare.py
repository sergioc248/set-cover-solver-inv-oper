"""Página de comparación: el GA frente al solver exacto, con las mismas métricas que el CLI."""

import json
import sys
from dataclasses import dataclass

import streamlit

from set_cover.compare import Comparison, MethodResult
from set_cover.instance import Instance
from set_cover.ui import charts, session
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    chart,
    current_problem,
    default_exact_choices,
    exact_run,
    exact_run_choices,
    options_key,
    show_table,
)
from set_cover.ui.data import (
    ExactRun,
    GAGroup,
    exact_method_result,
    group_results,
    jaccard,
    spanish,
)
from set_cover.ui.page_ga import (
    DISTRIBUTION_GUIDE,
    SAVED_GROUPS,
    saved_group,
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
_COLUMN_HELP = {
    "Costo": "Costo total de las antenas de la solución. Más bajo es mejor.",
    "Tiempo (s)": "GA: segundos de esa ejecución (un hilo). Gurobi: tiempo acumulado de todas "
    "sus sesiones, con todos los hilos de la máquina.",
    "Cota inferior": "Solo Gurobi: ningún costo puede ser menor que esto (demostrado).",
    "Brecha del solver (%)": "Solo Gurobi: (incumbente - su cota) / incumbente.",
    "Dif. vs mejor conocida (%)": "Cuánto más cara es que la mejor solución de la tabla. "
    "0 = es la mejor.",
    "Gap vs óptimo (%)": "Solo si algún método demostró el óptimo: distancia a él.",
    "Cota sup. del gap (%)": "(costo - mejor cota) / mejor cota: como máximo, cuánto más cara "
    "que el óptimo puede ser. El gap real es menor o igual.",
}
_SESSION_SUFFIX = "(esta sesión)"


@dataclass(frozen=True)
class _Catalog:
    """Todo lo comparable, por etiqueta: ejecuciones del solver y grupos del GA."""

    exact: dict[str, ExactRun]
    ga: dict[str, GAGroup]

    def labels(self) -> list[str]:
        return [*self.exact, *self.ga]


def render() -> None:
    instance, _ = current_problem()
    streamlit.title("Comparación: GA vs método exacto")
    streamlit.markdown(
        "Elija qué resultados comparar. Puede mezclar los **guardados del proyecto** con los "
        "que generó **en esta sesión** en las páginas del método exacto y del algoritmo "
        "genético. Antes de comparar, cada solución se verifica contra la matriz: todas deben "
        "cubrir a los 500 clientes."
    )
    catalog = _catalog()
    chosen = _choose(catalog)
    exact_runs = [catalog.exact[label] for label in chosen if label in catalog.exact]
    groups = [catalog.ga[label] for label in chosen if label in catalog.ga]
    results = _results(instance, exact_runs, groups)
    if not results:
        streamlit.info("Elija al menos un método.")
        return
    comparison = Comparison(results)
    _interpretation(comparison)
    _table(comparison)

    cost_tab, time_tab, distribution_tab, antennas_tab = streamlit.tabs(
        ["Costo", "Calidad vs tiempo", "Variabilidad del GA", "Antenas elegidas"]
    )
    with cost_tab:
        chart(
            charts.cost_bars(results, comparison.best_bound),
            "compare-costs",
            "Una barra por método, ordenadas de la más barata (arriba) a la más cara; **más "
            "corta es mejor**. Verde: GA; naranja: Gurobi. La línea punteada es la mejor "
            "solución conocida y la discontinua la mejor cota inferior: el óptimo está en algún "
            "punto entre ambas. El eje no empieza en cero para que se noten las diferencias.",
        )
    with time_tab:
        _quality_vs_time(exact_runs, groups)
    with distribution_tab:
        _distribution(groups, results)
    with antennas_tab:
        _antennas(results)
    _downloads(comparison)


def _catalog() -> _Catalog:
    include_other = streamlit.toggle(
        f"Incluir ejecuciones anteriores del solver en `{PATHS.solver_runs}/`",
        key="compare-include-other",
    )
    exact: dict[str, ExactRun] = {}
    for label, directory in exact_run_choices(include_other).items():
        try:
            exact[label] = exact_run(directory)
        except RESULT_ERRORS:
            streamlit.warning(f"No se pudo leer `{directory}`: {sys.exception()}")
    ga: dict[str, GAGroup] = {}
    for saved in SAVED_GROUPS:
        group = saved_group(saved)
        if group is not None:
            ga[f"{group.label} (guardada)"] = group
    for batch in session.ga_batches():
        group = batch.group()
        ga[f"{group.label} {_SESSION_SUFFIX}"] = group
    return _Catalog(exact, ga)


def _choose(catalog: _Catalog) -> list[str]:
    labels = catalog.labels()
    defaults = [*default_exact_choices(catalog.exact), *catalog.ga]
    chosen = streamlit.multiselect(
        "Métodos a comparar",
        labels,
        default=defaults,
        key=options_key("compare-methods", labels),
        help="«guardada»: resultado del proyecto. «esta sesión»: lo que ejecutó aquí. Un grupo "
        "del GA con varias semillas aporta dos filas: su mejor ejecución y la mediana.",
    )
    streamlit.caption(
        f"{len(catalog.exact)} ejecuciones de Gurobi y {len(catalog.ga)} grupos del GA disponibles."
    )
    return chosen


def _results(
    instance: Instance, exact_runs: list[ExactRun], groups: list[GAGroup]
) -> list[MethodResult]:
    results: list[MethodResult] = []
    for group in groups:
        try:
            results.extend(group_results(instance, group))
        except RESULT_ERRORS:
            streamlit.warning(f"{group.label}: {sys.exception()}")
    for run in exact_runs:
        result = exact_method_result(instance, run)
        if result is None:
            streamlit.warning(f"`{run.directory}` todavía no tiene una solución.")
        else:
            results.append(result)
    return results


def _interpretation(comparison: Comparison) -> None:
    best_known = comparison.best_known
    bound = comparison.best_bound
    best_method = min(comparison.results, key=lambda result: result.cost).method
    if comparison.optimum is not None:
        streamlit.success(
            f"**Óptimo demostrado: {spanish(comparison.optimum)}.** La columna «Gap vs óptimo» "
            "es (Z - Z*) / Z*."
        )
    elif bound is not None:
        streamlit.info(
            f"**Ningún método demostró optimalidad.** El óptimo está entre la mejor cota "
            f"inferior probada, **{spanish(bound)}**, y la mejor solución conocida, "
            f"**{spanish(best_known)}** ({best_method}). Por eso no hay «Gap vs óptimo»: se "
            "reporta la diferencia con la mejor conocida y (Z - cota) / cota, que acota el gap "
            "real por arriba."
        )
    else:
        streamlit.info(
            f"Sin cota inferior: incluya una ejecución de Gurobi. Mejor solución conocida: "
            f"**{spanish(best_known)}** ({best_method})."
        )


def _table(comparison: Comparison) -> None:
    rows = [{header: row.get(key) for key, header in _COLUMNS.items()} for row in comparison.rows()]
    # Una columna sin ningún valor (p. ej. «Gap vs óptimo» sin óptimo demostrado) no informa.
    filled = [
        header for header in _COLUMNS.values() if any(row[header] not in (None, "") for row in rows)
    ]
    show_table(
        [{header: row[header] for header in filled} for row in rows],
        {
            "Costo": 0,
            "Cota inferior": 0,
            "Tiempo (s)": 1,
            "Brecha del solver (%)": 2,
            "Dif. vs mejor conocida (%)": 2,
            "Gap vs óptimo (%)": 2,
            "Cota sup. del gap (%)": 2,
        },
    )
    with streamlit.expander("¿Qué significa cada columna?"):
        streamlit.markdown(
            "\n".join(
                f"- **{header}:** {text}"
                for header, text in _COLUMN_HELP.items()
                if header in filled
            )
        )


def _quality_vs_time(exact_runs: list[ExactRun], groups: list[GAGroup]) -> None:
    curves = {group.label: group.curves for group in groups if group.curves}
    chart(
        charts.quality_vs_time(curves, exact_runs),
        "compare-quality-time",
        "El mejor costo de cada método (eje vertical) a medida que pasa el tiempo (eje "
        "horizontal, logarítmico: cada marca es 10 veces más). **Abajo y a la izquierda es "
        "mejor.** GA: líneas tenues por semilla y gruesa para la mediana. Gurobi: incumbente "
        "(continua, baja) y cota inferior (discontinua, sube). El óptimo está siempre entre la "
        "cota más alta y la curva más baja.",
    )


def _distribution(groups: list[GAGroup], results: list[MethodResult]) -> None:
    several = {group.label: group.costs for group in groups if len(group.records) > 1}
    if not several:
        streamlit.info("Elija un grupo del GA con varias semillas para ver su variabilidad.")
        return
    references = {result.method: result.cost for result in results if result.best_bound is not None}
    chart(
        charts.cost_distribution(several, references),
        "compare-distribution",
        DISTRIBUTION_GUIDE + " Las líneas horizontales son las soluciones de Gurobi: los "
        "puntos por debajo de una línea son semillas del GA que la superan.",
    )


def _antennas(results: list[MethodResult]) -> None:
    selections = {result.method: result.antenna_ids for result in results if result.antenna_ids}
    if len(selections) < 2:
        streamlit.info("Se necesitan al menos dos soluciones con antenas para compararlas.")
        return
    chart(
        charts.antenna_overlap(selections),
        "compare-overlap",
        "Una fila por solución y una columna por antena que alguna eligió. Un cuadro de color "
        "indica que esa solución instala esa antena. Columnas llenas (marcadas con *) son "
        "antenas que todas eligen; filas con patrones distintos indican soluciones distintas "
        "aunque cuesten parecido.",
    )
    names = list(selections)
    streamlit.markdown("**Similitud de Jaccard** (antenas en común / antenas en la unión)")
    show_table(
        [
            {
                "Método": first,
                **{second: jaccard(selections[first], selections[second]) for second in names},
            }
            for first in names
        ],
    )
    streamlit.caption(
        "1 = mismas antenas; 0 = ninguna en común. Valores bajos entre soluciones de costo "
        "parecido muestran que hay muchas formas distintas de cubrir a los clientes."
    )


def _downloads(comparison: Comparison) -> None:
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
