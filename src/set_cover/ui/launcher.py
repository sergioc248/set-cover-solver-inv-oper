"""Punto de entrada ``set-cover-ui``: inicia el servidor de Streamlit con la aplicación."""

import os
import sys
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def command(arguments: list[str]) -> list[str]:
    """Comando de Streamlit; los argumentos extra pasan tal cual (p. ej. --server.port)."""
    return [sys.executable, "-m", "streamlit", "run", str(APP), *arguments]


def run() -> None:
    arguments = command(sys.argv[1:])
    # Reemplaza este proceso: Ctrl+C y las señales llegan directo a Streamlit.
    os.execv(arguments[0], arguments)
