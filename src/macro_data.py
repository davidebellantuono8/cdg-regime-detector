from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time

import numpy as np
import pandas as pd
import requests

# Official FRED API endpoint. Programmatic requests use a personal FRED API key.
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
_KEY_RE = re.compile(r"^[a-z0-9]{32}$")


@dataclass(frozen=True)
class SeriesInfo:
    series_id: str
    name: str
    source_group: str
    agg: str = "last"  # last | mean


# Conservative availability alignment for historical signals. FRED observation
# dates are reference-period dates, not release dates. Monthly indicators are
# shifted one month forward so a July CPI, for example, first enters the August
# signal. Revisions still require ALFRED for a fully vintage-safe backtest.
MONTHLY_RELEASE_LAG_1 = {
    "USALOLITOAASTSAM", "G7LOLITOAASTSAM", "CHNLOLITOAASTSAM",
    "PERMIT", "INDPRO", "TCU", "RSAFS", "UMCSENT", "PCEC96",
    "CPIAUCSL", "CPILFESL", "PCEPILFE", "WPSFD49116",
    "CES0500000003", "PALLFNFINDEXM", "PCOPPUSDM",
    "M2SL", "BOGMBASE",
}

FRED_SERIES: Dict[str, SeriesInfo] = {
    # Growth / activity
    "USALOLITOAASTSAM": SeriesInfo("USALOLITOAASTSAM", "OECD CLI USA", "Growth", "last"),
    "G7LOLITOAASTSAM": SeriesInfo("G7LOLITOAASTSAM", "OECD CLI G7", "Global Growth", "last"),
    "CHNLOLITOAASTSAM": SeriesInfo("CHNLOLITOAASTSAM", "OECD CLI Cina", "Global Growth", "last"),
    "IC4WSA": SeriesInfo("IC4WSA", "Initial Claims 4W average", "Growth", "mean"),
    "CCSA": SeriesInfo("CCSA", "Continued Claims", "Growth", "mean"),
    "PERMIT": SeriesInfo("PERMIT", "Building Permits", "Growth", "last"),
    "INDPRO": SeriesInfo("INDPRO", "Industrial Production", "Growth", "last"),
    "TCU": SeriesInfo("TCU", "Capacity Utilization", "Growth", "last"),
    "RSAFS": SeriesInfo("RSAFS", "Retail Sales", "Growth", "last"),
    "UMCSENT": SeriesInfo("UMCSENT", "Michigan Consumer Sentiment", "Growth", "last"),
    "PCEC96": SeriesInfo("PCEC96", "Real Personal Consumption", "Growth", "last"),

    # Inflation / price pressure
    "CPIAUCSL": SeriesInfo("CPIAUCSL", "Headline CPI", "Inflation", "last"),
    "CPILFESL": SeriesInfo("CPILFESL", "Core CPI", "Inflation", "last"),
    "PCEPILFE": SeriesInfo("PCEPILFE", "Core PCE", "Inflation", "last"),
    "WPSFD49116": SeriesInfo("WPSFD49116", "Core PPI Final Demand", "Inflation", "last"),
    "CES0500000003": SeriesInfo("CES0500000003", "Average Hourly Earnings", "Inflation", "last"),
    "T5YIFR": SeriesInfo("T5YIFR", "5Y5Y Inflation Expectations", "Inflation", "mean"),
    "T10YIE": SeriesInfo("T10YIE", "10Y Breakeven Inflation", "Inflation", "mean"),
    "PALLFNFINDEXM": SeriesInfo("PALLFNFINDEXM", "Global Commodity Price Index", "Inflation", "last"),
    "DCOILWTICO": SeriesInfo("DCOILWTICO", "WTI", "Inflation", "mean"),
    "PCOPPUSDM": SeriesInfo("PCOPPUSDM", "Copper", "Inflation", "last"),

    # Financial conditions / stress
    "NFCI": SeriesInfo("NFCI", "Chicago Fed NFCI", "Financial Stress", "mean"),
    "ANFCI": SeriesInfo("ANFCI", "Chicago Fed Adjusted NFCI", "Financial Stress", "mean"),
    "STLFSI4": SeriesInfo("STLFSI4", "St. Louis Fed Financial Stress", "Financial Stress", "mean"),
    "BAA10Y": SeriesInfo("BAA10Y", "Moody's Baa - Treasury Spread", "Financial Stress", "mean"),
    "T10Y3M": SeriesInfo("T10Y3M", "10Y-3M Treasury Curve", "Financial Stress", "mean"),
    "DFII10": SeriesInfo("DFII10", "10Y Real Yield", "Financial Stress", "mean"),
    "VIXCLS": SeriesInfo("VIXCLS", "VIX", "Financial Stress", "mean"),
    "DFF": SeriesInfo("DFF", "Effective Fed Funds Rate", "Financial Stress", "mean"),

    # Liquidity
    "WALCL": SeriesInfo("WALCL", "Fed Total Assets", "Liquidity", "last"),
    "WTREGEN": SeriesInfo("WTREGEN", "Treasury General Account", "Liquidity", "last"),
    "RRPONTSYD": SeriesInfo("RRPONTSYD", "ON Reverse Repo", "Liquidity", "mean"),
    "WRESBAL": SeriesInfo("WRESBAL", "Reserve Balances", "Liquidity", "last"),
    "M2SL": SeriesInfo("M2SL", "M2 Money Stock", "Liquidity", "last"),
    "BOGMBASE": SeriesInfo("BOGMBASE", "Monetary Base", "Liquidity", "last"),
    "TLAACBW027SBOG": SeriesInfo("TLAACBW027SBOG", "Commercial Bank Assets", "Liquidity", "last"),

    # USD / FX pressure
    "DTWEXBGS": SeriesInfo("DTWEXBGS", "Broad U.S. Dollar Index", "USD", "mean"),
    "DTWEXAFEGS": SeriesInfo("DTWEXAFEGS", "Advanced Foreign Economies USD Index", "USD", "mean"),
    "DGS2": SeriesInfo("DGS2", "2Y Treasury Yield", "USD", "mean"),
}


def validate_api_key(api_key: str | None) -> str:
    key = (api_key or "").strip()
    if not key:
        raise ValueError("FRED API key mancante")
    if not _KEY_RE.match(key):
        raise ValueError("FRED API key non valida: deve contenere 32 caratteri alfanumerici minuscoli")
    return key


def _cache_path(cache_dir: Path, series_id: str) -> Path:
    return cache_dir / f"{series_id}.csv"


def _read_cache(path: Path, series_id: str) -> pd.Series:
    df = pd.read_csv(path)
    if "DATE" not in df.columns or "VALUE" not in df.columns:
        raise ValueError("cache non riconosciuta")
    dates = pd.to_datetime(df["DATE"], errors="coerce")
    vals = pd.to_numeric(df["VALUE"], errors="coerce")
    s = pd.Series(vals.to_numpy(dtype=float), index=dates, name=series_id).dropna()
    return s[~s.index.duplicated(keep="last")].sort_index()


def _write_cache(path: Path, s: pd.Series) -> None:
    out = pd.DataFrame({"DATE": s.index.strftime("%Y-%m-%d"), "VALUE": s.to_numpy(dtype=float)})
    out.to_csv(path, index=False)


def _parse_api_json(payload: dict, series_id: str) -> pd.Series:
    obs = payload.get("observations")
    if not isinstance(obs, list):
        msg = payload.get("error_message") or payload.get("message") or "risposta FRED non riconosciuta"
        raise ValueError(str(msg))
    dates, vals = [], []
    for row in obs:
        dt = pd.to_datetime(row.get("date"), errors="coerce")
        val_raw = row.get("value")
        if val_raw in (None, ".", "") or pd.isna(dt):
            continue
        try:
            val = float(val_raw)
        except Exception:
            continue
        dates.append(dt)
        vals.append(val)
    if not vals:
        raise ValueError(f"FRED {series_id}: nessuna osservazione valida")
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=series_id, dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _fred_request(
    series_id: str,
    api_key: str,
    observation_start: str,
    timeout: int,
    limit: int | None = None,
    sort_order: str = "asc",
) -> pd.Series:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start,
        "sort_order": sort_order,
    }
    if limit is not None:
        params["limit"] = int(limit)
    r = requests.get(
        FRED_API_URL,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "CDG-Macro-Regime-Detector/2.2"},
    )
    try:
        payload = r.json()
    except Exception:
        payload = None
    if not r.ok:
        detail = ""
        if isinstance(payload, dict):
            detail = str(payload.get("error_message") or payload.get("message") or "")
        detail = detail.strip() or f"HTTP {r.status_code}"
        raise RuntimeError(detail)
    if not isinstance(payload, dict):
        raise RuntimeError("FRED ha restituito una risposta non JSON")
    return _parse_api_json(payload, series_id)


def test_fred_connection(api_key: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        key = validate_api_key(api_key)
        s = _fred_request("DGS2", key, "2026-01-01", timeout=timeout, limit=5, sort_order="desc")
        if s.empty:
            return False, "Connessione riuscita ma nessun dato ricevuto"
        return True, f"Connessione OK · DGS2 ultima osservazione {s.index.max().date()} = {s.iloc[-1]:.3f}"
    except Exception as exc:
        return False, str(exc)


def fetch_fred_series(
    series_id: str,
    api_key: str,
    cache_dir: str | Path,
    start: str = "1990-01-01",
    force_refresh: bool = False,
    timeout: int = 12,
    retries: int = 1,
    cache_ttl_hours: float = 12.0,
    revision_lookback_days: int = 550,
) -> tuple[pd.Series, str]:
    """Fetch one FRED series from the official API with local cache fallback.

    First run downloads the requested history. Later runs use a fresh cache; when
    the cache is stale only the recent revision window is refreshed and merged.
    """
    key = validate_api_key(api_key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, series_id)

    cached: pd.Series | None = None
    if cp.exists():
        try:
            cached = _read_cache(cp, series_id)
        except Exception:
            cached = None

    if cached is not None and not force_refresh:
        age_hours = max(0.0, (time.time() - cp.stat().st_mtime) / 3600.0)
        if age_hours <= cache_ttl_hours:
            return cached, f"CACHE FRESH ({age_hours:.1f}h)"

    request_start = start
    if cached is not None and not force_refresh and not cached.empty:
        recent_start = cached.index.max() - pd.Timedelta(days=int(revision_lookback_days))
        request_start = max(pd.Timestamp(start), recent_start).strftime("%Y-%m-%d")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            fresh = _fred_request(series_id, key, request_start, timeout=timeout)
            if cached is not None and not force_refresh:
                merged = pd.concat([cached[cached.index < fresh.index.min()], fresh]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
            else:
                merged = fresh
            _write_cache(cp, merged)
            tag = "DOWNLOAD" if cached is None or force_refresh else "REFRESH"
            if attempt:
                tag += f" retry {attempt}"
            return merged, tag
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))

    if cached is not None and not cached.empty:
        return cached, f"CACHE STALE (FRED error: {last_exc})"
    raise RuntimeError(f"FRED {series_id}: {last_exc}")


def _to_monthly(s: pd.Series, agg: str = "last", lag_months: int = 0) -> pd.Series:
    s = s.dropna().sort_index()
    if s.empty:
        return s
    groups = s.groupby(s.index.to_period("M"))
    out = groups.mean() if agg == "mean" else groups.last()
    out.index = out.index.to_timestamp("M")
    if lag_months:
        out.index = out.index + pd.offsets.MonthEnd(lag_months)
    out.name = s.name
    return out


def load_fred_macro_data(
    api_key: str,
    cache_dir: str | Path,
    start: str = "1990-01-01",
    force_refresh: bool = False,
    max_workers: int = 5,
    timeout: int = 12,
    retries: int = 1,
    cache_ttl_hours: float = 12.0,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all FRED series from the official API using bounded parallel calls."""
    key = validate_api_key(api_key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    monthly: dict[str, pd.Series] = {}
    status_rows: list[dict] = []
    total = len(FRED_SERIES)
    completed = 0

    def _one(sid: str, info: SeriesInfo):
        raw, status = fetch_fred_series(
            sid,
            api_key=key,
            cache_dir=cache_dir,
            start=start,
            force_refresh=force_refresh,
            timeout=timeout,
            retries=retries,
            cache_ttl_hours=cache_ttl_hours,
        )
        m = _to_monthly(raw, info.agg, 1 if sid in MONTHLY_RELEASE_LAG_1 else 0)
        row = {
            "Series": sid,
            "Name": info.name,
            "Group": info.source_group,
            "Status": status,
            "FirstDate": raw.index.min(),
            "LastDate": raw.index.max(),
            "LastValue": float(raw.iloc[-1]),
            "Observations": int(raw.notna().sum()),
            "Error": "",
        }
        return sid, m, row

    workers = max(1, min(int(max_workers), total))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fred") as pool:
        futures = {pool.submit(_one, sid, info): (sid, info) for sid, info in FRED_SERIES.items()}
        for fut in as_completed(futures):
            sid, info = futures[fut]
            try:
                sid_out, m, row = fut.result()
                monthly[sid_out] = m
                status_rows.append(row)
                status_text = str(row["Status"])
            except Exception as exc:
                status_text = f"ERROR: {exc}"
                status_rows.append({
                    "Series": sid,
                    "Name": info.name,
                    "Group": info.source_group,
                    "Status": "ERROR",
                    "FirstDate": pd.NaT,
                    "LastDate": pd.NaT,
                    "LastValue": np.nan,
                    "Observations": 0,
                    "Error": str(exc),
                })
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, sid, status_text)

    panel = pd.DataFrame(monthly).sort_index() if monthly else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    if not status.empty:
        order = {sid: i for i, sid in enumerate(FRED_SERIES)}
        status["_order"] = status["Series"].map(order)
        status = status.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return panel, status
