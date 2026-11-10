"""Aplicación Streamlit: ``uv run set-cover-ui`` desde la raíz del proyecto."""

import streamlit

from set_cover.ui import page_calibration, page_compare, page_exact, page_ga, page_instance


def main() -> None:
    streamlit.set_page_config(
        page_title="Set cover: GA vs Gurobi", page_icon=":material/cell_tower:", layout="wide"
    )
    pages = [
        streamlit.Page(
            page_instance.render,
            title="Instancia",
            icon=":material/grid_on:",
            url_path="instancia",
            default=True,
        ),
        streamlit.Page(
            page_exact.render, title="Método exacto", icon=":material/functions:", url_path="exacto"
        ),
        streamlit.Page(
            page_ga.render,
            title="Algoritmo genético",
            icon=":material/genetics:",
            url_path="genetico",
        ),
        streamlit.Page(
            page_compare.render,
            title="Comparación",
            icon=":material/compare_arrows:",
            url_path="comparacion",
        ),
        streamlit.Page(
            page_calibration.render,
            title="Calibración",
            icon=":material/tune:",
            url_path="calibracion",
        ),
    ]
    page = streamlit.navigation(pages)
    with streamlit.sidebar:
        streamlit.caption(
            "Ubicación de antenas como set cover ponderado: 500 clientes, 500 antenas. "
            "Compare el solver exacto (Gurobi) con el algoritmo genético."
        )
    page.run()


main()
