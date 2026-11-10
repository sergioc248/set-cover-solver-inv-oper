"""Figuras de Plotly. Cada función recibe datos ya cargados y retorna una figura nueva."""

from collections.abc import Mapping, Sequence

import numpy
from numpy.typing import NDArray
from plotly import graph_objects

from set_cover.compare import MethodResult
from set_cover.ga.engine import History
from set_cover.ui.data import Curve, ExactRun, number

GA_COLOR = "#2a9d8f"
GA_LIGHT = "rgba(42, 157, 143, 0.18)"
SOLVER_COLORS = ("#e76f51", "#9c6644", "#f4a261", "#bc4749", "#6a4c93")
BOUND_COLOR = "#6c757d"
REFERENCE_COLOR = "#264653"
_COST_FORMAT = ",.0f"


def _base(title: str, x_title: str, y_title: str) -> graph_objects.Figure:
    figure = graph_objects.Figure()
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.35, "x": 0},
        separators=",.",
    )
    return figure


def histogram(
    values: Sequence[float], title: str, x_title: str, bins: int = 30
) -> graph_objects.Figure:
    figure = _base(title, x_title, "Frecuencia")
    figure.add_trace(graph_objects.Histogram(x=list(values), nbinsx=bins, marker_color=GA_COLOR))
    figure.update_layout(bargap=0.05)
    return figure


def coverage_heatmap(coverage: NDArray[numpy.int8]) -> graph_objects.Figure:
    figure = _base("Matriz de cobertura a[i, j]", "Antena j", "Cliente i")
    rows, columns = coverage.shape
    figure.add_trace(
        graph_objects.Heatmap(
            z=coverage,
            x=list(range(1, columns + 1)),
            y=list(range(1, rows + 1)),
            colorscale=[[0.0, "#ffffff"], [1.0, REFERENCE_COLOR]],
            showscale=False,
            hovertemplate="cliente %{y}, antena %{x}: %{z}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(height=600)
    return figure


def exact_trajectory(runs: Sequence[ExactRun]) -> graph_objects.Figure:
    """Incumbente (línea continua) y cota inferior probada (discontinua) de cada ejecución."""
    figure = _base(
        "Solver exacto: incumbente y cota inferior", "Tiempo acumulado del solver (s)", "Costo"
    )
    for index, run in enumerate(runs):
        _add_exact_traces(figure, run, SOLVER_COLORS[index % len(SOLVER_COLORS)])
    figure.update_xaxes(type="log")
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def _add_exact_traces(figure: graph_objects.Figure, run: ExactRun, color: str) -> None:
    points = [point for point in run.trajectory if point.seconds > 0]
    seconds = [point.seconds for point in points]
    figure.add_trace(
        graph_objects.Scatter(
            x=seconds,
            y=[point.incumbent for point in points],
            mode="lines",
            line={"color": color, "shape": "hv"},
            name=f"{run.label}: incumbente",
            hovertemplate="%{x:,.1f} s<br>incumbente %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        graph_objects.Scatter(
            x=seconds,
            y=[point.bound for point in points],
            mode="lines",
            line={"color": color, "dash": "dash", "shape": "hv"},
            name=f"{run.label}: cota inferior",
            hovertemplate="%{x:,.1f} s<br>cota %{y:,.0f}<extra></extra>",
        )
    )


def ga_progress(
    generations: Sequence[int], best: Sequence[float], mean: Sequence[float]
) -> graph_objects.Figure:
    figure = _base("Progreso del GA", "Generación", "Costo")
    figure.add_trace(
        graph_objects.Scatter(
            x=list(generations), y=list(best), name="mejor global", line={"color": GA_COLOR}
        )
    )
    figure.add_trace(
        graph_objects.Scatter(
            x=list(generations),
            y=list(mean),
            name="media de la población",
            line={"color": BOUND_COLOR, "dash": "dot"},
        )
    )
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def ga_history(history: History) -> graph_objects.Figure:
    figure = _base("Convergencia por generación", "Generación", "Costo")
    series = (
        ("mejor global", history.best_so_far, {"color": GA_COLOR, "width": 3}),
        ("mejor de la población", history.population_best, {"color": REFERENCE_COLOR}),
        ("media de la población", history.population_mean, {"color": BOUND_COLOR, "dash": "dot"}),
        ("peor de la población", history.population_worst, {"color": "#adb5bd", "dash": "dash"}),
    )
    for name, values, line in series:
        figure.add_trace(
            graph_objects.Scatter(x=history.generation, y=values, name=name, line=line)
        )
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def ga_diversity(history: History) -> graph_objects.Figure:
    figure = _base("Diversidad de la población", "Generación", "Individuos distintos")
    figure.add_trace(
        graph_objects.Scatter(
            x=history.generation,
            y=history.unique_individuals,
            line={"color": GA_COLOR},
            name="individuos distintos",
        )
    )
    return figure


def quality_vs_time(
    curves: Sequence[Curve],
    median: tuple[Sequence[float], Sequence[float | None]] | None,
    exact_runs: Sequence[ExactRun],
) -> graph_objects.Figure:
    """Curvas del GA (una por semilla, más la mediana) y trayectorias del solver exacto."""
    figure = _base("Calidad vs tiempo", "Tiempo (s, escala logarítmica)", "Mejor costo")
    for index, curve in enumerate(curves):
        figure.add_trace(
            graph_objects.Scatter(
                x=curve.seconds,
                y=curve.best,
                mode="lines",
                line={"color": GA_LIGHT, "shape": "hv", "width": 1},
                name="GA: ejecuciones individuales",
                legendgroup="ga-runs",
                showlegend=index == 0,
                hovertemplate=f"{curve.label}<br>%{{x:,.2f}} s<br>%{{y:,.0f}}<extra></extra>",
            )
        )
    if median is not None:
        grid, values = median
        figure.add_trace(
            graph_objects.Scatter(
                x=list(grid),
                y=list(values),
                mode="lines",
                line={"color": GA_COLOR, "width": 3, "shape": "hv"},
                name="GA: mediana entre semillas",
                hovertemplate="%{x:,.2f} s<br>mediana %{y:,.0f}<extra></extra>",
            )
        )
    for index, run in enumerate(exact_runs):
        _add_exact_traces(figure, run, SOLVER_COLORS[index % len(SOLVER_COLORS)])
    figure.update_xaxes(type="log")
    figure.update_yaxes(tickformat=_COST_FORMAT)
    figure.update_layout(height=520)
    return figure


def cost_bars(results: Sequence[MethodResult], best_bound: float | None) -> graph_objects.Figure:
    figure = _base("Costo por método", "", "Costo")
    figure.add_trace(
        graph_objects.Bar(
            x=[result.method for result in results],
            y=[result.cost for result in results],
            marker_color=[
                SOLVER_COLORS[0] if result.best_bound is not None else GA_COLOR
                for result in results
            ],
            text=[f"{result.cost:,.0f}" for result in results],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    if best_bound is not None:
        figure.add_hline(
            y=best_bound,
            line={"color": BOUND_COLOR, "dash": "dash"},
            annotation_text=f"mejor cota inferior probada: {best_bound:,.0f}",
            annotation_position="bottom right",
        )
    best_known = min(result.cost for result in results)
    figure.add_hline(
        y=best_known,
        line={"color": REFERENCE_COLOR, "dash": "dot"},
        annotation_text=f"mejor solución conocida: {best_known:,.0f}",
        annotation_position="top left",
    )
    lowest = min(best_bound if best_bound is not None else best_known, best_known)
    highest = max(result.cost for result in results)
    figure.update_yaxes(range=[lowest * 0.95, highest * 1.04], tickformat=_COST_FORMAT)
    return figure


def cost_distribution(
    groups: Mapping[str, Sequence[float]], references: Mapping[str, float]
) -> graph_objects.Figure:
    """Distribución de costos del GA por grupo, con los costos del solver como referencias."""
    figure = _base("Distribución de costos del GA", "", "Costo")
    for name, costs in groups.items():
        figure.add_trace(
            graph_objects.Box(
                y=list(costs),
                name=name,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"color": GA_COLOR, "size": 6},
                line={"color": GA_COLOR},
            )
        )
    for index, (name, cost) in enumerate(references.items()):
        figure.add_hline(
            y=cost,
            line={"color": SOLVER_COLORS[index % len(SOLVER_COLORS)], "dash": "dash"},
            annotation_text=f"{name}: {cost:,.0f}",
            annotation_position="top right",
        )
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def antenna_overlap(selections: Mapping[str, Sequence[int]]) -> graph_objects.Figure:
    """Antenas elegidas por cada método; columnas = antenas elegidas por al menos uno."""
    union = sorted({antenna for ids in selections.values() for antenna in ids})
    names = list(selections)
    matrix = [[1 if antenna in set(selections[name]) else 0 for antenna in union] for name in names]
    shared = [sum(column) == len(names) for column in zip(*matrix, strict=True)] if names else []
    figure = _base(
        "Antenas seleccionadas por método", "Antena (id base 1; * = elegida por todos)", ""
    )
    figure.add_trace(
        graph_objects.Heatmap(
            z=matrix,
            x=[
                f"{antenna}*" if common else str(antenna)
                for antenna, common in zip(union, shared, strict=True)
            ],
            y=names,
            colorscale=[[0.0, "#f1f3f5"], [1.0, GA_COLOR]],
            showscale=False,
            xgap=2,
            ygap=2,
            hovertemplate="%{y}<br>antena %{x}: %{z}<extra></extra>",
        )
    )
    figure.update_xaxes(type="category", tickangle=-90)
    figure.update_layout(height=140 + 45 * len(names))
    return figure


def calibration_scatter(rows: Sequence[Mapping[str, str]], title: str) -> graph_objects.Figure:
    """Mediana del costo vs tiempo medio por configuración, coloreada por operador de mutación."""
    figure = _base(title, "Tiempo medio por ejecución (s)", "Mediana del costo")
    groups: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("mutation", "bit_flip") or "bit_flip", []).append(row)
    palette = (GA_COLOR, *SOLVER_COLORS)
    for index, (mutation, members) in enumerate(sorted(groups.items())):
        figure.add_trace(
            graph_objects.Scatter(
                x=[_cell(row, "mean_seconds") for row in members],
                y=[_cell(row, "median_cost") for row in members],
                mode="markers",
                name=mutation,
                marker={"size": 10, "color": palette[index % len(palette)], "opacity": 0.85},
                text=[f"#{row['rank']} {row['config_id']}" for row in members],
                hovertemplate="%{text}<br>%{x:.1f} s<br>mediana %{y:,.0f}<extra></extra>",
            )
        )
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def calibration_boxes(costs: Mapping[str, Sequence[float]], title: str) -> graph_objects.Figure:
    figure = _base(title, "Configuración (orden del ranking)", "Costo")
    for name, values in costs.items():
        figure.add_trace(
            graph_objects.Box(
                y=list(values),
                name=name,
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                marker={"color": GA_COLOR, "size": 5},
                line={"color": GA_COLOR},
                showlegend=False,
            )
        )
    figure.update_yaxes(tickformat=_COST_FORMAT)
    return figure


def parameter_effect(rows: Sequence[Mapping[str, str]], parameter: str) -> graph_objects.Figure:
    """Costo medio y mediano por valor de un parámetro, y fracción que alcanzó el mejor costo."""
    members = [row for row in rows if row["parameter"] == parameter]
    values = [row["value"] for row in members]
    figure = _base(f"Efecto de {parameter}", "Valor", "Costo")
    figure.add_trace(
        graph_objects.Bar(
            x=values,
            y=[_cell(row, "mean_cost") for row in members],
            name="costo medio",
            marker_color=GA_COLOR,
        )
    )
    figure.add_trace(
        graph_objects.Bar(
            x=values,
            y=[_cell(row, "median_cost") for row in members],
            name="mediana",
            marker_color=REFERENCE_COLOR,
        )
    )
    figure.add_trace(
        graph_objects.Scatter(
            x=values,
            y=[100 * (_cell(row, "reached_best_rate") or 0.0) for row in members],
            name="% que alcanza el mejor costo",
            yaxis="y2",
            mode="lines+markers",
            line={"color": SOLVER_COLORS[0]},
        )
    )
    costs = [
        cost
        for row in members
        for cost in (_cell(row, "mean_cost"), _cell(row, "median_cost"))
        if cost is not None
    ]
    if costs:
        figure.update_yaxes(range=[min(costs) * 0.98, max(costs) * 1.01], tickformat=_COST_FORMAT)
    figure.update_layout(
        barmode="group",
        xaxis={"type": "category"},
        yaxis2={
            "title": "% alcanza el mejor",
            "overlaying": "y",
            "side": "right",
            "range": [0, 100],
        },
    )
    return figure


def _cell(row: Mapping[str, str], key: str) -> float | None:
    return number(row, key)
