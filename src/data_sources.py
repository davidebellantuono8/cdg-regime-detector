from __future__ import annotations

from pathlib import Path
import csv
import io
import re
import unicodedata
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import yaml

REGIMES = [
    "Recession",
    "Debasement",
    "Stagflation",
    "Reflation",
    "Dollar Weakness",
    "Goldilocks Economy",
    "Disinflation / Soft Landing",
    "Deflation",
]

REGIME_ALIASES = {
    "Recession": ["recession", "recessione"],
    "Debasement": ["debasement", "monetary debasement"],
    "Stagflation": ["stagflation", "stagflazione"],
    "Reflation": ["reflation", "reflazione"],
    "Dollar Weakness": [
        "dollar weakness",
        "usd weakness",
        "weak dollar",
        "global rebalancing",
        "dollaro debole",
    ],
    "Goldilocks Economy": ["goldilocks", "goldilocks economy"],
    "Disinflation / Soft Landing": [
        "disinflation",
        "soft landing",
        "disinflation soft landing",
        "disinflazione",
    ],
    "Deflation": ["deflation", "deflazione"],
}

DATE_ALIASES = {
    "date",
    "data",
    "datum",
    "fecha",
    "jour",
    "giorno",
    "time",
    "timestamp",
    "period",
    "periodo",
}


def load_basket_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_text(value: object) -> str:
    s = "" if value is None else str(value)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _dedupe_columns(cols) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        base = str(c).strip() if str(c).strip() else "Colonna"
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.append(base if n == 0 else f"{base} ({n + 1})")
    return out


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """Parse numbers exported by Excel/Quantalys/FIDA across common locales."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    def one(v):
        if pd.isna(v):
            return np.nan
        x = str(v).strip().replace("\u00a0", "").replace(" ", "").replace("'", "")
        if not x or x.lower() in {"nan", "na", "n/a", "null", "none", "-", "--"}:
            return np.nan
        neg = x.startswith("(") and x.endswith(")")
        if neg:
            x = x[1:-1]
        x = x.replace("%", "")

        if "," in x and "." in x:
            # Last separator is normally the decimal separator.
            if x.rfind(",") > x.rfind("."):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            # Excel exports in Italian commonly use decimal comma.
            parts = x.split(",")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 6:
                x = parts[0].replace(".", "") + "." + parts[1]
            else:
                x = x.replace(",", "")
        else:
            # If dots appear more than once and the last block has 3 digits, treat as thousands.
            if x.count(".") > 1:
                blocks = x.split(".")
                if all(len(b) == 3 for b in blocks[1:]):
                    x = "".join(blocks)
        try:
            y = float(x)
            return -y if neg else y
        except Exception:
            return np.nan

    return s.map(one).astype(float)


def _parse_dates(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    # Excel serial dates.
    num = pd.to_numeric(s, errors="coerce")
    numeric_share = float(num.notna().mean()) if len(s) else 0.0
    if numeric_share >= 0.75:
        med = float(num.dropna().median()) if num.notna().any() else np.nan
        if 15000 <= med <= 90000:
            d = pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce")
            return d

    text = s.astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = pd.to_datetime(text, errors="coerce", dayfirst=True)
        b = pd.to_datetime(text, errors="coerce", dayfirst=False)

    def valid_score(d: pd.Series) -> int:
        ok = d.notna() & d.dt.year.between(1980, 2100)
        return int(ok.sum())

    d = a if valid_score(a) >= valid_score(b) else b
    return d.where(d.dt.year.between(1980, 2100))


def _date_column_score(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    d = _parse_dates(s)
    valid = d.notna()
    if valid.sum() < max(3, int(len(s) * 0.35)):
        return 0.0
    ratio = float(valid.mean())
    uniq = float(d[valid].nunique() / max(1, valid.sum()))
    monotonic = float(d[valid].is_monotonic_increasing or d[valid].is_monotonic_decreasing)
    return 0.70 * ratio + 0.20 * uniq + 0.10 * monotonic


def infer_date_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    best = None
    best_score = 0.0
    for c in df.columns:
        name = _norm_text(c)
        score = _date_column_score(df[c])
        if name in DATE_ALIASES or any(name.startswith(x + " ") for x in DATE_ALIASES):
            score += 0.35
        if score > best_score:
            best, best_score = str(c), score
    return best if best_score >= 0.55 else None


def numeric_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    out = []
    for c in df.columns:
        if exclude is not None and str(c) == str(exclude):
            continue
        x = _coerce_numeric(df[c])
        n_valid = int(x.notna().sum())
        if n_valid >= 3 and n_valid / max(1, len(df)) >= 0.45:
            out.append(str(c))
    return out


def _table_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or df.shape[1] < 2:
        return -999.0
    date_col = infer_date_column(df)
    n_num = len(numeric_columns(df, date_col))
    unnamed = sum(_norm_text(c).startswith("unnamed") for c in df.columns)
    return (6.0 if date_col else 0.0) + min(n_num, 12) * 1.5 + min(df.shape[1], 20) * 0.1 - unnamed * 0.25


def _decode_csv(raw: bytes) -> tuple[str, str]:
    errors = []
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return raw.decode(enc), enc
        except Exception as e:
            errors.append(str(e))
    raise ValueError("Impossibile decodificare il CSV.")


def _detect_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get)


def _read_csv_flexible(raw: bytes, header_row: int | None = None) -> tuple[pd.DataFrame, dict]:
    text, enc = _decode_csv(raw)
    sep = _detect_separator(text)
    candidate_seps = [sep] + [x for x in [",", ";", "\t", "|"] if x != sep]
    best = None
    best_meta = None
    best_score = -999.0
    max_header = min(15, max(0, len(text.splitlines()) - 1))
    headers = [header_row] if header_row is not None else list(range(max_header + 1))

    for sp in candidate_seps:
        for h in headers:
            try:
                df = pd.read_csv(
                    io.StringIO(text), sep=sp, header=h, dtype=object,
                    engine="python", on_bad_lines="skip"
                )
                df.columns = _dedupe_columns(df.columns)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                sc = _table_score(df)
                if sc > best_score:
                    best, best_score = df, sc
                    best_meta = {"kind": "CSV", "encoding": enc, "separator": sp, "header_row": int(h)}
            except Exception:
                continue
        # If separator produced a clearly strong table, avoid needless alternatives.
        if best_score >= 15:
            break

    if best is None or best.shape[1] < 2:
        raise ValueError("Non riesco a riconoscere la struttura del CSV.")
    return best, best_meta or {}


def excel_sheet_names(raw: bytes, filename: str) -> list[str]:
    try:
        xf = pd.ExcelFile(io.BytesIO(raw))
        return list(xf.sheet_names)
    except Exception as e:
        raise ValueError(f"Non riesco ad aprire il file Excel: {e}") from e


def _read_excel_flexible(
    raw: bytes,
    filename: str,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    try:
        xf = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Non riesco ad aprire il file Excel: {e}") from e

    sheets = [sheet_name] if sheet_name is not None else list(xf.sheet_names)
    best = None
    best_meta = None
    best_score = -999.0
    headers = [header_row] if header_row is not None else list(range(0, 16))

    for sh in sheets:
        for h in headers:
            try:
                df = pd.read_excel(xf, sheet_name=sh, header=h, dtype=object)
                df.columns = _dedupe_columns(df.columns)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                sc = _table_score(df)
                if sc > best_score:
                    best, best_score = df, sc
                    best_meta = {"kind": "Excel", "sheet": str(sh), "header_row": int(h)}
            except Exception:
                continue

    if best is None or best.shape[1] < 2:
        raise ValueError("Non riesco a riconoscere una tabella di serie prezzi nel file Excel.")
    return best, best_meta or {}


def read_price_file(
    raw: bytes,
    filename: str,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Read a generic CSV/Excel export containing a date column and price series.

    The function auto-detects CSV encoding/separator/header row, or Excel sheet/header row,
    then infers the date column and numeric price columns. It deliberately does not require
    fixed column names.
    """
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        df, meta = _read_excel_flexible(raw, filename, sheet_name=sheet_name, header_row=header_row)
    else:
        df, meta = _read_csv_flexible(raw, header_row=header_row)

    date_col = infer_date_column(df)
    nums = numeric_columns(df, date_col)
    if date_col is None:
        raise ValueError("Non riesco a identificare automaticamente la colonna data.")
    if len(nums) < 2:
        raise ValueError("Ho trovato la colonna data, ma non almeno due serie numeriche di prezzo.")
    meta.update({"date_col": date_col, "numeric_columns": nums, "rows": int(len(df)), "columns": int(df.shape[1])})
    return df, meta


def _alias_match_score(regime: str, column: str) -> float:
    c = _norm_text(column)
    # Remove common labels that add no semantic information.
    c2 = re.sub(r"\b(ptf|portfolio|portafoglio|radar|index|indice|price|prezzo|nav|ew|equal weight)\b", " ", c)
    c2 = re.sub(r"\s+", " ", c2).strip()
    best = 0.0
    for alias in REGIME_ALIASES[regime]:
        a = _norm_text(alias)
        if a and a in c2:
            best = max(best, 1.0 if c2 == a else 0.95)
        best = max(best, 0.75 * SequenceMatcher(None, a, c2).ratio())
    return best


def auto_map_regimes(columns: list[str]) -> dict[str, str | None]:
    candidates = []
    for r in REGIMES:
        for c in columns:
            candidates.append((_alias_match_score(r, c), r, c))
    candidates.sort(reverse=True)
    assigned_r = set()
    assigned_c = set()
    mapping: dict[str, str | None] = {r: None for r in REGIMES}
    for score, r, c in candidates:
        if score < 0.56:
            continue
        if r in assigned_r or c in assigned_c:
            continue
        mapping[r] = c
        assigned_r.add(r)
        assigned_c.add(c)
    return mapping


def build_regime_prices(
    df: pd.DataFrame,
    date_col: str,
    mapping: dict[str, str | None],
) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Colonna data '{date_col}' non trovata.")
    dates = _parse_dates(df[date_col])
    out = pd.DataFrame(index=dates)
    out.index.name = "Date"

    used = set()
    for regime in REGIMES:
        col = mapping.get(regime)
        if not col:
            continue
        if col not in df.columns:
            raise ValueError(f"Colonna '{col}' non trovata per {regime}.")
        if col in used:
            raise ValueError(f"La colonna '{col}' è stata assegnata a più di un regime.")
        used.add(col)
        out[regime] = _coerce_numeric(df[col]).to_numpy()

    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(how="all")
    # Price series must be positive; zero/negative values are treated as missing.
    out = out.mask(out <= 0)
    if out.shape[1] < 2:
        raise ValueError("Mappa almeno due serie ai regimi per eseguire il Market Engine.")
    if len(out) < 20:
        raise ValueError("Storico troppo corto: servono almeno 20 osservazioni valide.")
    return out


def parse_wide_csv(raw: bytes) -> pd.DataFrame:
    """Backward-compatible auto parser for CSVs that already contain recognisable regime names."""
    df, meta = read_price_file(raw, "upload.csv")
    mapping = auto_map_regimes(meta["numeric_columns"])
    missing = [r for r, c in mapping.items() if c is None]
    if missing:
        raise ValueError("Non riconosco automaticamente tutte le serie: " + ", ".join(missing))
    return build_regime_prices(df, meta["date_col"], mapping)


def nav_to_ohlc(nav: pd.Series) -> pd.DataFrame:
    """Fallback when only a basket NAV/price series is available.

    High/Low are approximated from adjacent closes. For the Regime Engine this mode is
    intended for robust trend/persistence ranking, not intraday analytics.
    """
    c = nav.astype(float).dropna().sort_index()
    prev = c.shift(1)
    o = prev.fillna(c)
    h = pd.concat([o, c], axis=1).max(axis=1)
    l = pd.concat([o, c], axis=1).min(axis=1)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})


def download_constituents(config: dict, start: str) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("Installa yfinance per usare la modalità Yahoo.") from e

    baskets = {}
    for item in config.get("baskets", []):
        name = item["name"]
        tickers = [t.strip() for t in item.get("tickers", []) if t and t.strip()]
        if len(tickers) != 5:
            continue
        raw = yf.download(
            tickers,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw.empty:
            continue
        baskets[name] = build_equal_weight_basket_ohlc(raw, tickers)
    return baskets


def _extract_field(raw: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            x = raw[field].copy()
        elif field in raw.columns.get_level_values(1):
            x = raw.xs(field, axis=1, level=1).copy()
        else:
            raise KeyError(field)
        x = x.reindex(columns=tickers)
        return x
    if len(tickers) == 1 and field in raw.columns:
        return raw[[field]].rename(columns={field: tickers[0]})
    raise ValueError("Formato Yahoo non riconosciuto.")


def build_equal_weight_basket_ohlc(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Synthetic basket with 20% monthly rebalancing.

    The basket OHLC is constructed from fixed units inside each month. High/Low are an
    approximation because component intraday extrema need not occur simultaneously.
    """
    fields = {f: _extract_field(raw, f, tickers).ffill() for f in ["Open", "High", "Low", "Close"]}
    close = fields["Close"].dropna(how="all")
    valid = close.dropna().index
    if len(valid) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    idx = close.index

    values = pd.DataFrame(index=idx, columns=["Open", "High", "Low", "Close"], dtype=float)
    capital = 100.0
    units = None
    current_month = None

    for dt in idx:
        row_c = fields["Close"].loc[dt]
        if row_c.isna().any():
            continue
        month = (dt.year, dt.month)
        if units is None or month != current_month:
            row_o = fields["Open"].loc[dt].fillna(row_c)
            if row_o.isna().any() or (row_o <= 0).any():
                continue
            units = (capital / len(tickers)) / row_o
            current_month = month

        for f in ["Open", "High", "Low", "Close"]:
            px = fields[f].loc[dt].fillna(row_c)
            values.loc[dt, f] = float((units * px).sum())
        capital = float(values.loc[dt, "Close"])

    return values.dropna()
