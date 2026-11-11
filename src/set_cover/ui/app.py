"""Aplicación Streamlit: ``uv run set-cover-ui`` desde la raíz del proyecto."""

import streamlit

from set_cover.ui import page_calibration, page_compare, page_exact, page_ga, page_problem, sidebar


def main() -> None:
    streamlit.set_page_config(
        page_title="Antenas: GA vs Gurobi", page_icon=":material/cell_tower:", layout="wide"
    )
    pages = [
        streamlit.Page(
            page_problem.render,
            title="El problema",
            icon=":material/cell_tower:",
            url_path="problema",
            default=True,
        ),
        streamlit.Page(
            page_exact.render, title="Método exacto", icon=":material/functions:", url_path="exacto"
        ),
        streamlit.Page(
            page_calibration.render,
            title="Calibración del GA",
            icon=":material/tune:",
            url_path="calibracion",
        ),
        streamlit.Page(
            page_ga.render,
            title="Algoritmo genético",
            icon=":material/genetics:",
            url_path=page_ga.URL_PATH,
        ),
        streamlit.Page(
            page_compare.render,
            title="Comparación",
            icon=":material/compare_arrows:",
            url_path="comparacion",
        ),
    ]
    page = streamlit.navigation(pages)
    page.run()
    # Después de la página: así los contadores incluyen lo que acaba de ejecutar.
    with streamlit.sidebar:
        sidebar.render(pages[0])


main()
