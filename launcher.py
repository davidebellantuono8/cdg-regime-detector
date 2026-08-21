from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQ = ROOT / "requirements.txt"
LOG = ROOT / "avvio_log.txt"

REQUIRED_MODULES = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "yaml",
    "openpyxl",
    "requests",
]


def log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def missing_modules() -> list[str]:
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def install_requirements() -> None:
    log("\n[1/3] Installazione/aggiornamento dipendenze necessarie...")
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(REQ)]
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    try:
        LOG.write_text("CDG Macro Regime Detector - log di avvio\n", encoding="utf-8")
    except Exception:
        pass

    log(f"Python: {sys.version.split()[0]}")
    log(f"Eseguibile: {sys.executable}")
    log(f"Cartella app: {ROOT}")

    missing = missing_modules()
    if missing:
        log("Moduli mancanti: " + ", ".join(missing))
        try:
            install_requirements()
        except Exception as exc:
            log("\nERRORE durante l'installazione delle dipendenze:")
            log(str(exc))
            log("\nControlla la connessione Internet e riprova. Il dettaglio resta in avvio_log.txt")
            return 2
    else:
        log("[1/3] Dipendenze già presenti.")

    # Import check after installation.
    missing_after = missing_modules()
    if missing_after:
        log("ERRORE: moduli ancora mancanti: " + ", ".join(missing_after))
        return 3

    log("[2/3] Controllo sintassi e moduli dell'app...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "py_compile", "app.py", "src/data_sources.py", "src/technical.py", "src/regime_engine.py", "src/macro_data.py", "src/macro_engine.py"],
            cwd=ROOT,
        )
    except Exception as exc:
        log(f"ERRORE nel controllo dell'app: {exc}")
        return 4

    log("[3/3] Avvio Streamlit...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address=localhost",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    log("Comando: " + " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        log(f"ERRORE avvio Streamlit: {exc}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
