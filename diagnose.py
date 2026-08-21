from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
print("Python:", sys.version)
print("Sistema:", platform.platform())
print("Eseguibile:", sys.executable)
print("Cartella:", ROOT)
print()

mods = ["streamlit", "pandas", "numpy", "plotly", "yaml", "openpyxl", "requests"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"[OK] {m}: {getattr(mod, '__version__', 'installato')}")
    except Exception as e:
        print(f"[ERRORE] {m}: {e}")

print("\nControllo moduli CDG...")
try:
    from src.data_sources import REGIMES  # noqa: F401
    from src.regime_engine import build_market_engine  # noqa: F401
    from src.technical import technical_history  # noqa: F401
    from src.macro_data import load_fred_macro_data, test_fred_connection  # noqa: F401
    from src.macro_engine import build_macro_engine  # noqa: F401
    print("[OK] moduli src importati correttamente")
except Exception as e:
    print("[ERRORE] moduli src:", repr(e))

print("\nFile essenziali:")
for rel in ["app.py", "requirements.txt", "config/baskets.yaml", "src/data_sources.py", "src/technical.py", "src/regime_engine.py", "src/macro_data.py", "src/macro_engine.py"]:
    p = ROOT / rel
    print("[OK]" if p.exists() else "[MANCANTE]", rel)

print("\nTest FRED API (se key salvata):")
key_file = ROOT / "config" / "fred_api_key.txt"
if key_file.exists():
    try:
        key = key_file.read_text(encoding="utf-8").strip()
        ok, msg = test_fred_connection(key, timeout=10)
        print("[OK]" if ok else "[ERRORE]", msg)
    except Exception as e:
        print("[ERRORE] test FRED:", repr(e))
else:
    print("[INFO] Nessuna key salvata. Inseriscila dall'app e usa 'Salva key sul PC'.")
