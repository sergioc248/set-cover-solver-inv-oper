"""Página del método exacto: lanzar o reanudar Gurobi, seguirlo en vivo y revisar ejecuciones."""

import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit

from set_cover.run import RUNNING
from set_cover.ui import charts
from set_cover.ui.common import (
    PATHS,
    RESULT_ERRORS,
    current_problem,
    exact_run,
    exact_run_directories,
    solver_jobs,
)
from set_cover.ui.data import ExactRun, duration
from set_cover.ui.solver_jobs import SolverJobError, SolverRequest, read_output, solver_command

_MONITORED_KEY = "solver_monitored_dir"
_NAME_KEY = "solver_new_name"
_ROTATE_NAME_KEY = "solver_rotate_name"
_VALID_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_NEW = "Nueva ejecución"
_RESUME = "Reanudar una existente"


def render() -> None:
    current_problem()
    streamlit.title("Método exacto (Gurobi)")
    streamlit.markdown(
        "Branch-and-bound de Gurobi sobre el modelo entero. En cada momento hay una "
        "**incumbente** (la mejor solución factible encontrada) y una **cota inferior** "
        "probada: ninguna solución cuesta menos. La **brecha** "
        "`(incumbente - cota) / incumbente` mide cuánto falta para demostrar optimalidad. "
        "Cada ejecución guarda un checkpoint en su directorio y se puede reanudar."
    )
    launch_tab, results_tab = streamlit.tabs(["Ejecutar", "Ejecuciones"])
    with launch_tab:
        _launcher()
    with results_tab:
        _results()


def _launcher() -> None:
    jobs = solver_jobs()
    resumable = [
        directory for directory in exact_run_directories() if directory.parent == PATHS.solver_runs
    ]
    # El nombre por defecto se renueva antes de crear el widget: después ya no se puede cambiar.
    if _NAME_KEY not in streamlit.session_state or streamlit.session_state.get(_ROTATE_NAME_KEY):
        streamlit.session_state[_NAME_KEY] = f"ui-{datetime.now():%Y%m%d-%H%M%S}"
        streamlit.session_state[_ROTATE_NAME_KEY] = False

    with streamlit.form("solver-form"):
        mode = streamlit.radio("Directorio de ejecución", [_NEW, _RESUME], horizontal=True)
        left, right = streamlit.columns(2)
        name = left.text_input(
            f"Nombre (se crea en `{PATHS.solver_runs}/`)",
            key=_NAME_KEY,
            help="Letras, dígitos, . _ -",
        )
        resume = right.selectbox(
            "Ejecución a reanudar",
            [directory.name for directory in resumable],
            index=None,
            placeholder="Ninguna disponible" if not resumable else "Elija una",
        )
        left, middle, right, last = streamlit.columns(4)
        gap_percent = left.number_input(
            "Brecha objetivo (%)",
            min_value=0.0,
            max_value=99.0,
            value=0.0,
            step=0.5,
            help="0 = seguir hasta demostrar optimalidad.",
        )
        time_limit = middle.number_input(
            "Límite de tiempo (s)", min_value=1.0, value=60.0, step=10.0
        )
        threads = right.number_input("Hilos (0 = automático)", min_value=0, value=0, step=1)
        seed = last.number_input("Semilla de Gurobi", min_value=0, value=0, step=1)
        unlimited = streamlit.checkbox(
            "Sin límite de tiempo (se detiene con el botón Detener o al alcanzar la brecha)"
        )
        submitted = streamlit.form_submit_button("Iniciar sesión del solver", type="primary")

    if submitted:
        run_name = name.strip() if mode == _NEW else (resume or "")
        if not _VALID_NAME.match(run_name):
            streamlit.error("Indique un nombre válido o elija una ejecución a reanudar.")
        else:
            run_dir = PATHS.solver_runs / run_name
            if mode == _NEW and run_dir.exists():
                streamlit.error(f"`{run_dir}` ya existe; use «{_RESUME}» o cambie el nombre.")
            else:
                request = SolverRequest(
                    run_dir=run_dir,
                    coverage=PATHS.coverage,
                    costs=PATHS.costs,
                    target_gap=float(gap_percent) / 100.0,
                    time_limit=None if unlimited else float(time_limit),
                    threads=int(threads),
                    seed=int(seed),
                )
                try:
                    jobs.start(run_dir, solver_command(request))
                except (SolverJobError, OSError):
                    streamlit.error(f"No se pudo iniciar el solver: {sys.exception()}")
                else:
                    streamlit.session_state[_MONITORED_KEY] = str(run_dir)
                    streamlit.session_state[_ROTATE_NAME_KEY] = True

    monitored = streamlit.session_state.get(_MONITORED_KEY)
    if isinstance(monitored, str):
        _monitor(monitored)
    else:
        streamlit.info(
            "Cada sesión corre `set-cover solve` en un proceso aparte: sigue aunque cierre la "
            "página, y «Detener» guarda el checkpoint. La licencia restringida incluida en "
            "`gurobipy` alcanza para esta instancia (500 x 500)."
        )


@streamlit.fragment(run_every=2)
def _monitor(run_dir_text: str) -> None:
    run_dir = Path(run_dir_text)
    jobs = solver_jobs()
    running = jobs.running(run_dir)
    streamlit.subheader(f"Sesión en `{run_dir}`")
    if running:
        streamlit.markdown(":orange[● En ejecución] — se actualiza cada 2 s.")
        stop = streamlit.button("Detener (guarda el checkpoint)", key="solver-stop")
        if stop and jobs.stop(run_dir):
            streamlit.toast("Señal enviada; el solver guarda y termina.")
    else:
        code = jobs.exit_code(run_dir)
        if code is None or code == 0:
            streamlit.markdown(":green[● Terminada]")
        else:
            streamlit.markdown(f":red[● Terminó con error (código {code})]; revise la salida.")

    try:
        run = exact_run(run_dir)
    except RESULT_ERRORS:
        run = None
    if run is not None:
        _metrics(run)
        if run.trajectory:
            streamlit.plotly_chart(charts.exact_trajectory([run]), key="solver-live-chart")
    with streamlit.expander("Salida del solver", expanded=run is None):
        streamlit.code(read_output(run_dir) or "(sin salida todavía)", language=None)


def _metrics(run: ExactRun) -> None:
    checkpoint = run.checkpoint
    gap = checkpoint.gap()
    columns = streamlit.columns(4)
    columns[0].metric(
        "Incumbente",
        "—" if checkpoint.incumbent is None else f"{checkpoint.incumbent.cost:,.0f}",
    )
    columns[1].metric(
        "Cota inferior", "—" if checkpoint.best_bound is None else f"{checkpoint.best_bound:,.0f}"
    )
    columns[2].metric("Brecha", "—" if gap is None else f"{gap:.2%}")
    columns[3].metric("Tiempo acumulado", duration(checkpoint.total_runtime_seconds))


def _status(run: ExactRun) -> str:
    if run.running:
        return "en ejecución"
    if not run.checkpoint.sessions:
        return "sin sesiones"
    status = run.checkpoint.sessions[-1].status
    # Sin lock, una sesión marcada "running" murió sin cerrar su checkpoint.
    return "abortada" if status == RUNNING else status


def _summary_row(run: ExactRun) -> dict[str, object]:
    checkpoint = run.checkpoint
    incumbent = checkpoint.incumbent
    gap = checkpoint.gap()
    return {
        "Ejecución": str(run.directory),
        "Estado": _status(run),
        "Incumbente": None if incumbent is None else incumbent.cost,
        "Antenas": None if incumbent is None else len(incumbent.antenna_ids),
        "Cota inferior": checkpoint.best_bound,
        "Brecha (%)": None if gap is None else 100 * gap,
        "Tiempo": duration(checkpoint.total_runtime_seconds),
        "Sesiones": len(checkpoint.sessions),
    }


def _results() -> None:
    directories = exact_run_directories()
    if not directories:
        streamlit.info(
            f"No hay ejecuciones en `{PATHS.saved_solver_runs}` ni en `{PATHS.solver_runs}`."
        )
        return
    by_label = {str(directory): directory for directory in directories}
    saved = [str(d) for d in directories if d.parent == PATHS.saved_solver_runs]
    chosen = streamlit.multiselect(
        "Ejecuciones a mostrar",
        list(by_label),
        default=saved or list(by_label)[:1],
        help=f"Guardadas en `{PATHS.saved_solver_runs}`; las nuevas quedan en "
        f"`{PATHS.solver_runs}`.",
    )
    runs: list[ExactRun] = []
    for label in chosen:
        try:
            runs.append(exact_run(by_label[label]))
        except RESULT_ERRORS:
            streamlit.warning(f"No se pudo leer `{label}`: {sys.exception()}")
    if not runs:
        return

    streamlit.dataframe(
        [_summary_row(run) for run in runs],
        hide_index=True,
        column_config={
            "Incumbente": streamlit.column_config.NumberColumn(format="localized"),
            "Cota inferior": streamlit.column_config.NumberColumn(format="localized"),
            "Brecha (%)": streamlit.column_config.NumberColumn(format="%.2f"),
        },
    )
    streamlit.plotly_chart(charts.exact_trajectory(runs), key="solver-results-chart")
    streamlit.caption(
        "Eje de tiempo logarítmico y acumulado entre sesiones. La cota es el máximo probado hasta "
        "ese instante: una sesión reanudada no la pierde aunque el log reinicie desde la LP."
    )

    detail_label = streamlit.selectbox("Detalle de", [str(run.directory) for run in runs])
    detail = next(run for run in runs if str(run.directory) == detail_label)
    incumbent = detail.checkpoint.incumbent
    if incumbent is not None:
        ids = incumbent.antenna_ids
        streamlit.markdown(
            f"**Antenas de la incumbente ({len(ids)}, base 1):** "
            + ", ".join(str(antenna) for antenna in ids)
        )
    streamlit.dataframe(
        [
            {
                "Inicio": session.started_at,
                "Fin": session.ended_at,
                "Estado": session.status,
                "Tiempo (s)": round(session.runtime_seconds, 1),
                "Nodos": int(session.nodes),
                "Brecha objetivo": session.target_gap,
                "Cota de la sesión": session.best_bound,
            }
            for session in detail.checkpoint.sessions
        ],
        hide_index=True,
    )
