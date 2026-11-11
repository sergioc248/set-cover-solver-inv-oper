"""Barra lateral: resumen de la sesión y botón para restablecerla."""

import streamlit
from streamlit.navigation.page import StreamlitPage

from set_cover.ui import session
from set_cover.ui.common import solver_jobs

_RESETS_KEY = "session-resets"


def render(home: StreamlitPage | None = None) -> None:
    """``home``: página a la que vuelve «Restablecer»; sin ella, solo recarga la actual."""
    streamlit.caption(
        "Dónde instalar antenas para dar señal a 500 clientes al menor costo: el método "
        "exacto (Gurobi) frente a un algoritmo genético."
    )
    batches, calibrations, solver_runs = session.counts()
    streamlit.markdown(
        "**En esta sesión**  \n"
        f"{batches} lote(s) del GA · {calibrations} calibración(es) · "
        f"{solver_runs} ejecución(es) de Gurobi"
    )
    # Una clave nueva tras cada restablecimiento cierra el popover.
    resets = streamlit.session_state.get(_RESETS_KEY, 0)
    with streamlit.popover(
        "Restablecer sesión", icon=":material/restart_alt:", key=f"session-reset-{resets}"
    ):
        streamlit.markdown(
            "Vuelve al estado inicial: solo se muestran los **resultados guardados del "
            "proyecto**. Olvida los lotes del GA, las calibraciones y las ejecuciones de "
            "Gurobi de esta sesión, y los valores de los formularios. Las sesiones de Gurobi "
            "y calibraciones en curso se detienen (guardan su avance). Los archivos en "
            "`runs/` no se borran."
        )
        if streamlit.button("Sí, restablecer", type="primary", key="session-reset"):
            jobs = solver_jobs()
            directories = [
                *session.solver_runs(),
                *(item.directory for item in session.calibrations()),
            ]
            for directory in directories:
                if jobs.running(directory):
                    jobs.stop(directory)
            session.reset()
            streamlit.session_state[_RESETS_KEY] = resets + 1
            # Cambiar de página descarta también los valores que el navegador conserva en los
            # widgets de la página actual.
            if home is None:
                streamlit.rerun()
            streamlit.switch_page(home)
