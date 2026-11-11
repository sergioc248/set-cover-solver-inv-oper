"""Página del problema: qué se decide, la notación, los datos y un explorador."""

import streamlit

from set_cover.instance import Instance
from set_cover.ui import charts
from set_cover.ui.common import PATHS, chart, current_problem, show_table
from set_cover.ui.data import instance_stats, percent, spanish

# Tamaño del recorte de la matriz que ilustra la notación.
_SAMPLE = 8


def render() -> None:
    instance, _ = current_problem()
    stats = instance_stats(instance)

    streamlit.title("El problema: dónde instalar antenas")
    streamlit.markdown(
        f"Una empresa debe dar señal a **{stats.customers} clientes**. Tiene **{stats.antennas} "
        "lugares candidatos** donde puede instalar una antena; cada antena tiene un costo de "
        "instalación y llega solo a algunos clientes. **¿Qué antenas instalar para que todos "
        "los clientes tengan señal al menor costo total?**"
    )
    streamlit.markdown(
        "Es un problema de *cobertura de conjuntos ponderado* (weighted set cover). El método "
        "exacto (Gurobi) y el algoritmo genético resuelven exactamente este modelo con los "
        "mismos archivos."
    )

    _notation()
    _model()
    _sample(instance)
    _metrics(instance)
    _explorer(instance)
    _charts(instance)


def _notation() -> None:
    streamlit.subheader("Notación")
    show_table(
        [
            {
                "Símbolo": "i",
                "Significa": "un cliente (una persona o zona que necesita señal)",
                "En los datos": f"fila i de {PATHS.coverage}; i = 1 … 500",
            },
            {
                "Símbolo": "j",
                "Significa": "una antena candidata (un lugar donde se puede instalar)",
                "En los datos": f"columna j de {PATHS.coverage} y fila j de {PATHS.costs}; "
                "j = 1 … 500",
            },
            {
                "Símbolo": "a_ij",
                "Significa": "1 si la antena j llega al cliente i; 0 si no",
                "En los datos": "celda (i, j) de la matriz de cobertura",
            },
            {
                "Símbolo": "c_j",
                "Significa": "costo de instalar la antena j",
                "En los datos": str(PATHS.costs),
            },
            {
                "Símbolo": "x_j",
                "Significa": "la decisión: 1 si se instala la antena j; 0 si no",
                "En los datos": "lo que calculan los métodos",
            },
        ]
    )
    streamlit.caption(
        "Los índices empiezan en 1, igual que en los resultados: «antena 44» es la columna 44."
    )


def _model() -> None:
    streamlit.subheader("Modelo")
    streamlit.latex(
        r"\min \sum_{j=1}^{500} c_j x_j \quad \text{s.a.} \quad \sum_{j=1}^{500} a_{ij} x_j "
        r"\ge 1 \;\; \forall i = 1, \dots, 500, \qquad x_j \in \{0, 1\}"
    )
    streamlit.markdown(
        "- **Objetivo:** el costo total de las antenas instaladas, lo más bajo posible.\n"
        "- **Restricción:** cada cliente i queda cubierto por al menos una antena instalada.\n"
        "- **Decisión:** instalar (1) o no (0) cada antena."
    )


def _sample(instance: Instance) -> None:
    with streamlit.expander(f"Ejemplo: las primeras {_SAMPLE} filas y columnas de la matriz"):
        streamlit.markdown(
            "Cada fila es un cliente y cada columna una antena. Un **1** en (i, j) significa "
            "que la antena j le da señal al cliente i. La matriz completa tiene "
            f"{instance.num_customers} filas y {instance.num_antennas} columnas."
        )
        streamlit.dataframe(
            [
                {
                    "cliente": f"i = {row + 1}",
                    **{
                        f"j = {column + 1}": int(instance.coverage[row, column])
                        for column in range(_SAMPLE)
                    },
                }
                for row in range(_SAMPLE)
            ],
            hide_index=True,
        )
        streamlit.markdown(
            "Costos de esas antenas: "
            + ", ".join(
                f"c_{column + 1} = {spanish(float(instance.costs[column]))}"
                for column in range(_SAMPLE)
            )
        )


def _metrics(instance: Instance) -> None:
    stats = instance_stats(instance)
    streamlit.subheader("Los datos en números")
    columns = streamlit.columns(4)
    columns[0].metric("Clientes (i)", spanish(stats.customers))
    columns[1].metric("Antenas candidatas (j)", spanish(stats.antennas))
    columns[2].metric(
        "Pares antena-cliente con señal",
        spanish(stats.links),
        help="Cantidad de unos en la matriz de cobertura.",
    )
    columns[3].metric(
        "Densidad de la matriz",
        percent(stats.density, 1),
        help="Fracción de pares (i, j) en que la antena j llega al cliente i.",
    )
    columns = streamlit.columns(4)
    columns[0].metric("Antena más barata", spanish(stats.cost.minimum))
    columns[1].metric("Antena más cara", spanish(stats.cost.maximum))
    columns[2].metric("Costo medio de una antena", spanish(stats.cost.mean, 1))
    columns[3].metric(
        "Instalar todas costaría",
        spanish(stats.total_cost),
        help="Solución trivial y factible: el peor caso razonable.",
    )


def _explorer(instance: Instance) -> None:
    streamlit.subheader("Explorar clientes y antenas")
    left, right = streamlit.columns(2)
    with left:
        customer = int(
            streamlit.number_input(
                "Cliente i", min_value=1, max_value=instance.num_customers, value=1, step=1
            )
        )
        antennas = [
            int(column) + 1 for column in instance.coverage[customer - 1].nonzero()[0].tolist()
        ]
        streamlit.markdown(
            f"**{len(antennas)} antenas** le dan señal al cliente {customer}. Basta instalar "
            "una de ellas para cubrirlo."
        )
        by_cost = sorted((float(instance.costs[antenna - 1]), antenna) for antenna in antennas)
        show_table(
            [{"Antena j": str(antenna), "Costo c_j": cost} for cost, antenna in by_cost],
            {"Costo c_j": 0},
        )
    with right:
        antenna = int(
            streamlit.number_input(
                "Antena j", min_value=1, max_value=instance.num_antennas, value=1, step=1
            )
        )
        customers = [
            int(row) + 1 for row in instance.coverage[:, antenna - 1].nonzero()[0].tolist()
        ]
        cost = float(instance.costs[antenna - 1])
        streamlit.markdown(
            f"La antena {antenna} cuesta **{spanish(cost)}** y le da señal a "
            f"**{len(customers)} clientes**: {spanish(cost / len(customers), 1)} por cliente "
            "cubierto."
            if customers
            else f"La antena {antenna} cuesta **{spanish(cost)}** y no cubre a ningún cliente."
        )
        streamlit.markdown(
            "Clientes cubiertos: " + ", ".join(str(customer) for customer in customers)
        )


def _charts(instance: Instance) -> None:
    streamlit.subheader("Cómo son los datos")
    costs_tab, coverage_tab, matrix_tab = streamlit.tabs(
        ["Costos de las antenas", "Cuántas antenas y clientes", "Matriz de cobertura"]
    )
    with costs_tab:
        chart(
            charts.histogram(
                instance.costs.tolist(), "Distribución de costos de las antenas", "Costo c_j"
            ),
            "problem-costs",
            "Cada barra agrupa antenas con costos parecidos; su altura es cuántas antenas "
            "hay en ese rango. Una forma plana indica que los costos se reparten de manera "
            "pareja entre el mínimo y el máximo: no hay un grupo de antenas mucho más baratas.",
        )
    with coverage_tab:
        left, right = streamlit.columns(2)
        with left:
            chart(
                charts.histogram(
                    instance.coverage.sum(axis=1).tolist(),
                    "Antenas que llegan a cada cliente",
                    "Número de antenas",
                    bins=20,
                ),
                "problem-per-customer",
                "Eje horizontal: cuántas antenas llegan a un cliente. Altura: cuántos clientes "
                "tienen esa cantidad. Más a la derecha, más opciones para cubrir a ese cliente; "
                "los clientes de la izquierda son los más difíciles de cubrir.",
            )
        with right:
            chart(
                charts.histogram(
                    instance.coverage.sum(axis=0).tolist(),
                    "Clientes a los que llega cada antena",
                    "Número de clientes",
                    bins=20,
                ),
                "problem-per-antenna",
                "Eje horizontal: a cuántos clientes llega una antena. Altura: cuántas antenas "
                "tienen esa cantidad. Las antenas de la derecha cubren más clientes y, si no "
                "son caras, suelen convenir.",
            )
    with matrix_tab:
        chart(
            charts.coverage_heatmap(instance.coverage),
            "problem-matrix",
            "Fila = cliente i, columna = antena j. Un punto oscuro en (i, j) significa que la "
            "antena j llega al cliente i. Los puntos se reparten sin patrón: no hay zonas que "
            "una sola antena resuelva. Pase el cursor para ver el par exacto.",
        )
