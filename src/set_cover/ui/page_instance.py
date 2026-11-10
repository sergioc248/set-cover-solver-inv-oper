"""Página de la instancia: modelo, tamaño, costos y estructura de cobertura."""

import streamlit

from set_cover.ui import charts
from set_cover.ui.common import PATHS, current_problem
from set_cover.ui.data import instance_stats


def render() -> None:
    instance, _ = current_problem()
    stats = instance_stats(instance)

    streamlit.title("Instancia")
    streamlit.markdown(
        "Ubicación de antenas como **set cover ponderado**: elegir el conjunto de antenas de "
        "menor costo total tal que cada cliente quede cubierto por al menos una antena. "
        "Ambos métodos resuelven exactamente este modelo sobre los mismos archivos."
    )
    streamlit.latex(
        r"\min \sum_{j} c_j x_j \quad \text{s.a.} \quad \sum_{j} a_{ij} x_j \ge 1 \;\; "
        r"\forall i, \qquad x_j \in \{0, 1\}"
    )
    streamlit.caption(f"Datos: `{PATHS.coverage}` y `{PATHS.costs}`. Ids de antena en base 1.")

    columns = streamlit.columns(4)
    columns[0].metric("Clientes", f"{stats.customers:,}")
    columns[1].metric("Antenas", f"{stats.antennas:,}")
    columns[2].metric("Enlaces de cobertura", f"{stats.links:,}")
    columns[3].metric("Densidad de la matriz", f"{stats.density:.1%}")
    columns = streamlit.columns(4)
    columns[0].metric("Costo mínimo de una antena", f"{stats.cost.minimum:,.0f}")
    columns[1].metric("Costo máximo de una antena", f"{stats.cost.maximum:,.0f}")
    columns[2].metric("Costo medio", f"{stats.cost.mean:,.1f}")
    columns[3].metric("Costo de instalar todas", f"{stats.total_cost:,.0f}")

    costs_tab, coverage_tab, matrix_tab = streamlit.tabs(
        ["Costos", "Cobertura", "Matriz de cobertura"]
    )
    with costs_tab:
        streamlit.plotly_chart(
            charts.histogram(
                instance.costs.tolist(), "Distribución de costos de las antenas", "Costo"
            )
        )
    with coverage_tab:
        left, right = streamlit.columns(2)
        left.plotly_chart(
            charts.histogram(
                instance.coverage.sum(axis=1).tolist(),
                "Antenas que cubren a cada cliente",
                "Número de antenas",
                bins=20,
            )
        )
        right.plotly_chart(
            charts.histogram(
                instance.coverage.sum(axis=0).tolist(),
                "Clientes que cubre cada antena",
                "Número de clientes",
                bins=20,
            )
        )
        streamlit.caption(
            f"Cada cliente tiene entre {stats.antennas_per_customer.minimum:.0f} y "
            f"{stats.antennas_per_customer.maximum:.0f} antenas posibles; cada antena cubre "
            f"entre {stats.customers_per_antenna.minimum:.0f} y "
            f"{stats.customers_per_antenna.maximum:.0f} clientes."
        )
    with matrix_tab:
        streamlit.plotly_chart(charts.coverage_heatmap(instance.coverage))
