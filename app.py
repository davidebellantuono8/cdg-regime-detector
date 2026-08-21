from __future__ import annotations

from pathlib import Path
import io
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources import (
    REGIMES,
    auto_map_regimes,
    build_regime_prices,
    download_constituents,
    excel_sheet_names,
    load_basket_config,
    nav_to_ohlc,
    numeric_columns,
    read_price_file,
)
from src.regime_engine import build_market_engine
from src.macro_data import FRED_SERIES, load_fred_macro_data, test_fred_connection
from src.macro_engine import FACTOR_NAMES, build_macro_engine, build_decision_engine, decision_preview, decision_validation

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "baskets.yaml"
CACHE = ROOT / "data" / "basket_prices.csv"
FRED_CACHE = ROOT / "data" / "fred_cache"
FRED_KEY_FILE = ROOT / "config" / "fred_api_key.txt"

st.set_page_config(page_title="CDG Macro Regime Detector", layout="wide")
st.title("CDG — Macro Regime Detector")
st.caption("v2.5 · Market Engine + Macro Leading 2–3M + Decision Engine V1.1 · Market Proximity Filter")

with st.sidebar:
    st.header("Dati")
    mode = st.radio("Sorgente", ["File prezzi CSV / Excel", "Yahoo constituents"], index=0)
    st.caption("Il detector usa gli 8 basket Equal Weight, non i Momentum Tilt.")

    st.header("Pesi V1")
    st.caption("Current = 45% F + 30% Q + 20% RP + 5% A")
    st.caption("Emerging = 25% F + 10% Q + 25% RP + 40% A")
    show_advanced = st.checkbox("Mostra parametri avanzati", False)
    if show_advanced:
        normal_margin = st.number_input(
            "Margine switch normale", 0.0, 30.0, 7.0, 0.5,
            help="Vantaggio minimo di Current del nuovo leader rispetto al regime attivo per avviare la conferma."
        )
        confirm_months = st.number_input(
            "Conferme mensili", 1, 4, 2, 1,
            help="Numero di rilevazioni mensili consecutive richieste per uno switch normale."
        )
        min_current_switch = st.number_input(
            "Min Current per switch normale", 40.0, 90.0, 60.0, 1.0,
            help="Impedisce di cambiare regime quando il leader è semplicemente il meno debole. Default 60."
        )
        quick_current = st.number_input("Quick: Current", 50.0, 100.0, 75.0, 1.0)
        quick_accel = st.number_input("Quick: Accelerazione", 50.0, 100.0, 70.0, 1.0)
        quick_q = st.number_input("Quick: Persistenza", 40.0, 100.0, 60.0, 1.0)
        st.caption("Confidence: HIGH >15 · MEDIUM-HIGH 8–15 · MEDIUM 4–8 · LOW <4 punti di spread tra 1° e 2° Current.")
    else:
        normal_margin, confirm_months, min_current_switch = 7.0, 2, 60.0
        quick_current, quick_accel, quick_q = 75.0, 70.0, 60.0

    st.header("Macro Leading Engine")
    macro_start = st.text_input("Storico FRED da", "1990-01-01")

    saved_key = ""
    try:
        if FRED_KEY_FILE.exists():
            saved_key = FRED_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        saved_key = ""
    env_key = os.getenv("FRED_API_KEY", "").strip()
    default_key = saved_key or env_key
    fred_api_key = st.text_input(
        "FRED API key", value=default_key, type="password",
        help="Chiave personale gratuita FRED. Serve solo per il download automatico dei dati macro e resta sul tuo PC se scegli Salva."
    ).strip()
    k1, k2 = st.columns(2)
    save_key = k1.button("Salva key sul PC", disabled=not bool(fred_api_key))
    test_key = k2.button("Test FRED", disabled=not bool(fred_api_key))
    if save_key:
        try:
            FRED_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
            FRED_KEY_FILE.write_text(fred_api_key, encoding="utf-8")
            st.success("FRED API key salvata localmente.")
        except Exception as exc:
            st.error(f"Impossibile salvare la key: {exc}")
    if test_key:
        ok, msg = test_fred_connection(fred_api_key)
        if ok:
            st.success(msg)
        else:
            st.error("Test FRED fallito: " + msg)

    refresh_macro = st.button(
        "Aggiorna dati macro FRED",
        disabled=not bool(fred_api_key),
        help="Aggiorna tutte le serie tramite API ufficiale FRED. Le cache recenti sono riutilizzate; il refresh forzato riscarica lo storico."
    )
    if not fred_api_key:
        st.warning("Inserisci una FRED API key per attivare il Macro Leading Engine. Il Market Engine continua a funzionare anche senza key.")
    st.caption(f"{len(FRED_SERIES)} serie FRED automatiche · API ufficiale · download paralleli · cache locale incrementale.")

prices = None
basket_ohlc = None

if mode == "File prezzi CSV / Excel":
    st.info(
        "Carica direttamente un export di serie prezzi da Excel, FIDA, Quantalys o altra fonte. "
        "L'app prova a riconoscere automaticamente formato, intestazione, colonna data, separatori decimali "
        "e nomi degli 8 basket. Se i nomi sono diversi puoi mapparli manualmente."
    )
    up = st.file_uploader(
        "File con serie prezzi",
        type=["csv", "txt", "xlsx", "xls", "xlsm", "xlsb"],
        help="Sono accettate date giornaliere, settimanali o mensili e numeri con virgola o punto decimale.",
    )

    if up is None and CACHE.exists():
        try:
            prices = pd.read_csv(CACHE, parse_dates=["Date"]).set_index("Date").sort_index()
            st.success("Nessun nuovo file caricato: uso l'ultimo dataset normalizzato salvato in cache.")
        except Exception as e:
            st.warning(f"Cache non leggibile: {e}")

    elif up is not None:
        raw = up.getvalue()
        suffix = Path(up.name).suffix.lower()
        selected_sheet = None

        if suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
            try:
                sheets = excel_sheet_names(raw, up.name)
                if len(sheets) > 1:
                    selected_sheet = st.selectbox("Foglio Excel", sheets, index=0)
                elif sheets:
                    selected_sheet = sheets[0]
            except Exception as e:
                st.error(str(e))
                st.stop()

        with st.expander("Opzioni importazione", expanded=False):
            manual_header = st.checkbox("Imposta manualmente la riga di intestazione", False)
            header_human = st.number_input(
                "Riga intestazione (1 = prima riga)", min_value=1, max_value=100, value=1, step=1,
                disabled=not manual_header,
            )
            st.caption(
                "Normalmente lascia automatico. È utile solo se il file contiene titoli/note prima della tabella."
            )
        header_row = int(header_human) - 1 if manual_header else None

        try:
            raw_df, meta = read_price_file(
                raw,
                up.name,
                sheet_name=selected_sheet,
                header_row=header_row,
            )
        except Exception as e:
            st.error(f"Importazione non riuscita: {e}")
            st.stop()

        all_columns = [str(c) for c in raw_df.columns]
        detected_date = str(meta["date_col"])
        date_idx = all_columns.index(detected_date) if detected_date in all_columns else 0

        with st.expander("Controllo struttura e mappatura", expanded=True):
            left, right = st.columns([1, 1])
            with left:
                date_col = st.selectbox("Colonna data", all_columns, index=date_idx)
                price_cols = numeric_columns(raw_df, date_col)
                auto_map = auto_map_regimes(price_cols)
                st.caption(
                    f"Riconosciute {len(price_cols)} colonne numeriche. "
                    "La mappatura proposta può essere corretta manualmente."
                )
            with right:
                details = [
                    f"Formato: **{meta.get('kind', 'n.d.')}**",
                    f"Intestazione: **riga {int(meta.get('header_row', 0)) + 1}**",
                ]
                if meta.get("sheet") is not None:
                    details.append(f"Foglio: **{meta['sheet']}**")
                if meta.get("separator") is not None:
                    sep_label = {",": "virgola", ";": "punto e virgola", "\t": "TAB", "|": "pipe"}.get(meta["separator"], meta["separator"])
                    details.append(f"Separatore CSV: **{sep_label}**")
                if meta.get("encoding") is not None:
                    details.append(f"Encoding: **{meta['encoding']}**")
                st.markdown("  \n".join(details))

            mapping: dict[str, str | None] = {}
            options = ["— non usare —"] + price_cols
            c1, c2 = st.columns(2)
            for i, regime in enumerate(REGIMES):
                default = auto_map.get(regime)
                default_idx = options.index(default) if default in options else 0
                target = c1 if i % 2 == 0 else c2
                chosen = target.selectbox(regime, options, index=default_idx, key=f"map_{regime}")
                mapping[regime] = None if chosen == "— non usare —" else chosen

            if st.checkbox("Mostra anteprima del file originale", False):
                st.dataframe(raw_df.head(20), use_container_width=True)

        try:
            prices = build_regime_prices(raw_df, date_col, mapping)
            prices.reset_index().to_csv(CACHE, index=False)
            n_mapped = prices.shape[1]
            start_dt = prices.index.min().date()
            end_dt = prices.index.max().date()
            st.success(
                f"Importazione riuscita: {n_mapped} basket, {len(prices)} osservazioni, "
                f"dal {start_dt} al {end_dt}. Dataset normalizzato salvato in cache."
            )
            missing_map = [r for r in REGIMES if r not in prices.columns]
            if missing_map:
                st.warning("Basket non mappati: " + ", ".join(missing_map))
            with st.expander("Anteprima dataset normalizzato", expanded=False):
                st.dataframe(prices.tail(15), use_container_width=True)
        except Exception as e:
            st.error(f"Mappatura non valida: {e}")
            st.stop()

    if prices is not None:
        basket_ohlc = {name: nav_to_ohlc(prices[name]) for name in prices.columns}

else:
    cfg = load_basket_config(CONFIG)
    start = st.sidebar.text_input("Data inizio", cfg.get("settings", {}).get("start_date", "2010-01-01"))
    st.warning(
        "La modalità Yahoo richiede 5 ticker validi per ciascun basket nel file config/baskets.yaml. "
        "Il basket sintetico viene ribilanciato al 20% all'inizio di ogni mese."
    )
    if st.button("Scarica e costruisci i basket", type="primary"):
        try:
            basket_ohlc = download_constituents(cfg, start)
            if not basket_ohlc:
                raise ValueError("Nessun basket completo: inserisci 5 ticker Yahoo per ciascun regime.")
            prices = pd.DataFrame({k: v["Close"] for k, v in basket_ohlc.items()}).dropna(how="all")
            prices.reset_index(names="Date").to_csv(CACHE, index=False)
            st.success(f"Costruiti {len(basket_ohlc)} basket.")
        except Exception as e:
            st.error(str(e))

if prices is None or basket_ohlc is None:
    st.stop()

missing = [r for r in REGIMES if r not in prices.columns]
if missing:
    st.warning("Mancano alcuni basket: " + ", ".join(missing))

# ---------------- Market Regime Engine ----------------
try:
    hist, latest = build_market_engine(
        basket_ohlc,
        prices,
        normal_margin=float(normal_margin),
        confirm_months=int(confirm_months),
        quick_current=float(quick_current),
        quick_accel=float(quick_accel),
        quick_q=float(quick_q),
        min_current_switch=float(min_current_switch),
    )
except Exception as e:
    st.error(f"Market Engine non disponibile: {e}")
    st.stop()

latest = latest.sort_values("Current", ascending=False)
active = latest["ActiveRegime"].iloc[0]
active_score = float(latest["ActiveScore"].iloc[0])
emerging = latest["EmergingCandidate"].iloc[0]
emerging_score = float(latest["EmergingScore"].iloc[0])
last_date = pd.Timestamp(latest["Date"].iloc[0]).date()
leader = latest["MarketLeader"].iloc[0]
leader_score = float(latest["LeaderScore"].iloc[0])
runner_up = latest["RunnerUp"].iloc[0]
spread = float(latest["RegimeSpread"].iloc[0])
confidence = latest["Confidence"].iloc[0]
market_state = latest["MarketState"].iloc[0]
pending = latest["PendingCandidate"].iloc[0]
pending_streak = int(latest["PendingStreak"].iloc[0]) if pd.notna(latest["PendingStreak"].iloc[0]) else 0

# ---------------- Macro Leading Engine ----------------
# v2.3: API ufficiale FRED + progress reale + cache locale/sessione.
# Il Macro Engine non viene ricalcolato a ogni rerun di Streamlit: viene ricaricato
# soltanto al primo utilizzo, se cambia la data di inizio o se l'utente forza Aggiorna.
macro_bundle = None
macro_status = pd.DataFrame()
macro_error = None
macro_elapsed = None
macro_cache_note = ""

need_macro_reload = bool(fred_api_key) and (
    bool(refresh_macro)
    or st.session_state.get("_macro_start") != macro_start
    or st.session_state.get("_macro_key_fingerprint") != (fred_api_key[-6:] if fred_api_key else "")
    or st.session_state.get("_macro_basket_fingerprint") != f"{len(prices)}|{prices.index.min()}|{prices.index.max()}|{prices.shape[1]}"
    or "_macro_bundle" not in st.session_state
)

if not fred_api_key:
    macro_error = "FRED API key mancante"
    macro_bundle = None
    macro_status = pd.DataFrame()
elif need_macro_reload:
    progress = st.progress(0, text=f"FRED 0/{len(FRED_SERIES)} · avvio download paralleli...")
    status_line = st.empty()
    t0 = time.perf_counter()

    def _fred_progress(done: int, total: int, sid: str, status: str) -> None:
        pct = int(round(100 * done / max(total, 1)))
        short_status = "cache" if str(status).startswith("CACHE") else ("download" if str(status).startswith(("DOWNLOAD", "REFRESH")) else "errore")
        progress.progress(done / max(total, 1), text=f"FRED {done}/{total} ({pct}%) · {sid} · {short_status}")

    try:
        macro_panel, macro_status = load_fred_macro_data(
            api_key=fred_api_key,
            cache_dir=FRED_CACHE,
            start=macro_start,
            force_refresh=bool(refresh_macro),
            max_workers=5,
            timeout=12,
            retries=1,
            cache_ttl_hours=12.0,
            progress_callback=_fred_progress,
        )
        n_ok = int((macro_status["Status"] != "ERROR").sum()) if not macro_status.empty else 0
        n_err = int((macro_status["Status"] == "ERROR").sum()) if not macro_status.empty else len(FRED_SERIES)
        if macro_panel.empty:
            first_err = ""
            if not macro_status.empty and "Error" in macro_status.columns:
                errs = macro_status.loc[macro_status["Error"].astype(str).str.len() > 0, "Error"]
                first_err = str(errs.iloc[0]) if len(errs) else ""
            raise RuntimeError(f"0/{len(FRED_SERIES)} serie FRED disponibili" + (f" · primo errore: {first_err}" if first_err else ""))
        status_line.info(f"FRED disponibili {n_ok}/{len(FRED_SERIES)} · errori {n_err}. Calcolo 12 dimensioni, archetipi e calibrazione walk-forward...")
        macro_bundle = build_macro_engine(macro_panel, basket_prices=prices)
        macro_elapsed = time.perf_counter() - t0

        st.session_state["_macro_start"] = macro_start
        st.session_state["_macro_key_fingerprint"] = fred_api_key[-6:]
        st.session_state["_macro_basket_fingerprint"] = f"{len(prices)}|{prices.index.min()}|{prices.index.max()}|{prices.shape[1]}"
        st.session_state["_macro_bundle"] = macro_bundle
        st.session_state["_macro_status"] = macro_status
        st.session_state["_macro_error"] = None
        st.session_state["_macro_elapsed"] = macro_elapsed

        n_download = int(macro_status["Status"].astype(str).str.startswith(("DOWNLOAD", "REFRESH")).sum()) if not macro_status.empty else 0
        n_cache = int(macro_status["Status"].astype(str).str.startswith("CACHE").sum()) if not macro_status.empty else 0
        n_error = int((macro_status["Status"] == "ERROR").sum()) if not macro_status.empty else 0
        progress.progress(1.0, text=f"FRED completato: {len(FRED_SERIES)}/{len(FRED_SERIES)}")
        status_line.success(
            f"Macro Engine pronto in {macro_elapsed:.1f}s · download {n_download} · cache {n_cache} · errori {n_error}."
        )
    except Exception as e:
        macro_error = str(e)
        st.session_state["_macro_error"] = macro_error
        st.session_state["_macro_status"] = macro_status
        progress.empty()
        status_line.error(f"Macro Engine non disponibile: {macro_error}")
        if not macro_status.empty:
            with st.expander("Dettaglio errori FRED", expanded=True):
                err_view = macro_status.copy()
                st.dataframe(err_view, use_container_width=True, hide_index=True)
else:
    macro_bundle = st.session_state.get("_macro_bundle")
    macro_status = st.session_state.get("_macro_status", pd.DataFrame())
    macro_error = st.session_state.get("_macro_error")
    macro_elapsed = st.session_state.get("_macro_elapsed")
    if macro_bundle is not None:
        macro_cache_note = "Dati macro già caricati in questa sessione: nessun nuovo download FRED."

macro_latest = macro_bundle["latest_regimes"] if macro_bundle is not None else pd.DataFrame()
decision_history = build_decision_engine(hist, macro_bundle["regimes"]) if macro_bundle is not None else pd.DataFrame()
decision = decision_preview(latest, macro_latest, decision_history) if macro_bundle is not None else {}
decision_val = decision_validation(decision_history, prices) if macro_bundle is not None else pd.DataFrame()

# ---------------- Main navigation ----------------
tab_dash, tab_market, tab_macro, tab_decision, tab_history = st.tabs(
    ["Dashboard", "Market Engine", "Macro Leading 2–3M", "Decision Engine", "History / Export"]
)

with tab_dash:
    st.subheader("Quadro operativo")
    if macro_bundle is not None and not macro_latest.empty:
        macro_top = macro_latest.iloc[0]
        macro_next = str(macro_top["Regime"])
        macro_score = float(macro_top["MacroScore"])
        macro_conf = str(macro_top["MacroConfidence"])
        macro_period = pd.Timestamp(macro_bundle["last_date"]).strftime("%m/%Y")
    else:
        macro_next, macro_score, macro_conf, macro_period = "N.D.", np.nan, "N.D.", "N.D."

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("REGIME OPERATIVO", active, f"Market score {active_score:.1f}")
    c2.metric("MARKET STATE", market_state, f"Leader {leader} {leader_score:.1f}")
    c3.metric("MARKET CONFIDENCE", confidence, f"Spread {spread:.1f}")
    c4.metric("MACRO NEXT 1–3M", macro_next, f"Score {macro_score:.1f} · {macro_conf}" if np.isfinite(macro_score) else "")
    c5.metric("MARKET EMERGING", emerging or "—", f"{emerging_score:.1f}" if np.isfinite(emerging_score) else "")
    c6.metric("AZIONE V2.5", decision.get("Action", "N.D."), decision.get("TransitionText", ""))

    st.caption(f"Market signal: {last_date} · Macro period: {macro_period}. Il Macro Engine è anticipatore e in v2.3 NON forza ancora lo switch del regime operativo.")
    if macro_cache_note:
        st.caption("⚡ " + macro_cache_note)

    if decision:
        if decision["Action"] == "PREPARE":
            st.warning(f"PREPARE: {active} → {decision['MacroNext']} · {decision['TransitionText']}")
        elif decision["Action"] == "WATCH":
            st.info(f"WATCH: mantieni {active}. {decision['TransitionText']}")
        else:
            st.success(f"HOLD: {active}. {decision['TransitionText']}")
    elif macro_error:
        st.warning(f"Macro Engine non disponibile: {macro_error}. Il Market Engine continua a funzionare normalmente.")

    st.markdown("#### Top Macro Regimes")
    if macro_bundle is not None:
        top_macro = macro_latest[["Regime", "MacroScore", "EconomicScore", "EmpiricalScore", "Distance12D"]].copy()
        for c in ["MacroScore", "EconomicScore", "EmpiricalScore"]:
            top_macro[c] = pd.to_numeric(top_macro[c], errors="coerce").round(1)
        top_macro["Distance12D"] = pd.to_numeric(top_macro["Distance12D"], errors="coerce").round(2)
        st.dataframe(top_macro.set_index("Regime"), use_container_width=True)

        fshow = macro_bundle["latest_factors"][["Factor", "LevelScore", "ImpulseScore", "Coverage"]].copy()
        for c in ["LevelScore", "ImpulseScore"]:
            fshow[c] = fshow[c].round(1)
        fshow["Coverage"] = (100 * fshow["Coverage"]).round(0).astype("Int64")
        st.markdown("#### Sei fattori macro")
        st.dataframe(fshow.set_index("Factor"), use_container_width=True)

with tab_market:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ACTIVE REGIME", active, f"Score {active_score:.1f}")
    c2.metric("MARKET LEADER", leader, f"Current {leader_score:.1f}")
    c3.metric("REGIME CONFIDENCE", confidence, f"Spread {spread:.1f} vs {runner_up}")
    c4.metric("EMERGING CANDIDATE", emerging or "—", f"Score {emerging_score:.1f}" if np.isfinite(emerging_score) else "")
    c5.metric("Data segnale", str(last_date))

    if market_state == "TRANSITION":
        if pending:
            st.warning(
                f"TRANSITION: regime operativo sempre {active} · candidato {pending} · "
                f"conferma {pending_streak}/{int(confirm_months)} · spread {spread:.1f} ({confidence})."
            )
        else:
            st.warning(f"TRANSITION: regime operativo sempre {active}, ma ActiveScore {active_score:.1f} < 50. Attendo conferma del successore.")
    elif market_state == "WATCH":
        st.info(f"WATCH: regime operativo {active} ancora mantenuto, ActiveScore {active_score:.1f} nella fascia 50–60.")
    else:
        emerging_transition = (
            emerging is not None and np.isfinite(emerging_score) and emerging_score >= 65
            and float(latest.loc[latest["Regime"] == emerging, "A"].iloc[0]) >= 60
        )
        if emerging_transition:
            st.info(f"HOLD: {active} · Emerging watch: {emerging} ({emerging_score:.1f})")
        else:
            st.success(f"HOLD: {active} · Confidence {confidence} · spread {spread:.1f}")

    st.subheader("Mappa dei regimi")
    prev_date = sorted(hist["Date"].unique())[-2] if hist["Date"].nunique() > 1 else None
    prev = hist[hist["Date"] == prev_date].set_index("Regime") if prev_date is not None else pd.DataFrame()

    fig = go.Figure()
    for _, r in latest.iterrows():
        name = r["Regime"]
        if not prev.empty and name in prev.index:
            p = prev.loc[name]
            fig.add_trace(go.Scatter(
                x=[p["Q"], r["Q"]], y=[p["F"], r["F"]], mode="lines",
                line=dict(width=1), showlegend=False, hoverinfo="skip", opacity=0.45,
            ))
        fig.add_trace(go.Scatter(
            x=[r["Q"]], y=[r["F"]], mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=max(12, min(28, r["Current"] / 3))),
            name=name,
            customdata=[[r["Current"], r["Emerging"], r["RP"], r["A"], r["State"]]],
            hovertemplate=(
                "<b>%{text}</b><br>Forza %{y:.1f}<br>Persistenza %{x:.1f}"
                "<br>Current %{customdata[0]:.1f}<br>Emerging %{customdata[1]:.1f}"
                "<br>Relative Perf %{customdata[2]:.1f}<br>Acceleration %{customdata[3]:.1f}"
                "<br>State %{customdata[4]}<extra></extra>"
            ),
        ))
    fig.update_xaxes(range=[0, 100], title="QUALITÀ / PERSISTENZA DEL TREND")
    fig.update_yaxes(range=[0, 100], title="FORZA DEL TREND")
    fig.update_layout(height=650, showlegend=False, margin=dict(l=30, r=30, t=20, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classifica Market Regime Engine")
    show = latest[["Regime", "F", "Q", "RP", "A", "Current", "Emerging", "State"]].copy()
    for c in ["F", "Q", "RP", "A", "Current", "Emerging"]:
        show[c] = show[c].round(1)
    st.dataframe(show.set_index("Regime"), use_container_width=True)

    st.subheader("Score nel tempo")
    metric = st.selectbox("Metrica Market", ["Current", "Emerging", "F", "Q", "RP", "A"], index=0)
    pivot = hist.pivot(index="Date", columns="Regime", values=metric)
    st.line_chart(pivot.tail(60))

    with st.expander("Metodologia Market Engine"):
        st.markdown(
            """
**Current** = 45% Forza + 30% Persistenza + 20% Relative Performance + 5% Accelerazione.  
**Emerging** = 25% Forza + 10% Persistenza + 25% Relative Performance + 40% Accelerazione.  

Lo switch normale richiede: nuovo leader Current ≥60, vantaggio ≥7 punti e 2 conferme mensili.  
Quick switch: Current ≥75, Accelerazione ≥70, Persistenza ≥60.  
Il regime operativo è **sempre valorizzato**: STABLE / WATCH / TRANSITION indicano soltanto lo stato di affidabilità/transizione.
            """
        )

with tab_macro:
    st.subheader("Macro Leading Engine 2–3 mesi")
    st.caption("Dati automatici FRED. Debasement e Dollar Weakness competono come regimi first-class, non come overlay.")
    if macro_bundle is None:
        st.error(f"Macro Engine non disponibile: {macro_error}")
        if not macro_status.empty:
            st.markdown("#### Diagnostica FRED")
            ms = macro_status.copy()
            st.dataframe(ms, use_container_width=True, hide_index=True)
        elif not fred_api_key:
            st.info("Configura la FRED API key nella barra laterale e premi Test FRED. Poi il Macro Engine partirà automaticamente.")
    else:
        ml = macro_latest.copy()
        top = ml.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MACRO NEXT", top["Regime"], f"Score {top['MacroScore']:.1f}")
        c2.metric("MACRO CONFIDENCE", top["MacroConfidence"], f"Gap score {top['MacroSpread']:.1f}")
        c3.metric("RUNNER-UP", top["MacroRunnerUp"], f"Score {ml.iloc[1]['MacroScore']:.1f}" if len(ml) > 1 else "")
        c4.metric("Periodo macro", pd.Timestamp(macro_bundle["last_date"]).strftime("%m/%Y"))

        st.markdown("#### Classifica degli 8 regimi macro")
        tbl = ml[["Regime", "MacroScore", "EconomicScore", "EmpiricalScore", "PredictedRelativeReturn", "Distance12D"]].copy()
        tbl.columns = ["Regime", "Macro Score", "Economic 70%", "Empirical 30%", "Pred. rel. 1-3M", "Distance 12D"]
        for c in ["Macro Score", "Economic 70%", "Empirical 30%"]:
            tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(1)
        tbl["Pred. rel. 1-3M"] = (100 * pd.to_numeric(tbl["Pred. rel. 1-3M"], errors="coerce")).round(2)
        tbl["Distance 12D"] = pd.to_numeric(tbl["Distance 12D"], errors="coerce").round(2)
        st.dataframe(tbl.set_index("Regime"), use_container_width=True)

        st.markdown("#### Fattori macro: Level + Impulse")
        fshow = macro_bundle["latest_factors"][["Factor", "LevelScore", "ImpulseScore", "Coverage"]].copy()
        fshow.columns = ["Factor", "Level", "Impulse", "Coverage"]
        for c in ["Level", "Impulse"]:
            fshow[c] = fshow[c].round(1)
        fshow["Coverage"] = (100 * fshow["Coverage"]).round(0).astype("Int64")
        st.dataframe(fshow.set_index("Factor"), use_container_width=True)

        st.markdown("#### Macro Score nel tempo")
        mp = macro_bundle["regimes"].pivot(index="Date", columns="Regime", values="MacroScore")
        st.line_chart(mp.tail(60))

        validation = macro_bundle.get("validation", pd.DataFrame())
        if validation is not None and not validation.empty:
            st.markdown("#### Validazione walk-forward sui basket EW")
            v = validation.copy()
            for c in ["Top1HitPct", "Top3HitPct"]:
                v[c] = pd.to_numeric(v[c], errors="coerce").round(1)
            v["RankCorrelation"] = pd.to_numeric(v["RankCorrelation"], errors="coerce").round(3)
            v.columns = ["Orizzonte", "Osservazioni", "Top-1 %", "Top-3 %", "Corr. rank"]
            st.dataframe(v.set_index("Orizzonte"), use_container_width=True)
            st.caption("Riferimenti casuali con 8 basket: Top-1 12,5% · Top-3 37,5%. La calibrazione empirica è rigorosamente walk-forward.")

        st.markdown("#### Dettaglio indicatori")
        indicators = macro_bundle["latest_indicators"].copy()
        if not indicators.empty:
            indicators["Level score"] = np.clip(50 + 18 * indicators["LevelZ"], 0, 100)
            indicators["Impulse score"] = np.clip(50 + 18 * indicators["ImpulseZ"], 0, 100)
            det = indicators[["Factor", "Indicator", "Raw", "Transformed", "Level score", "Impulse score", "Weight"]].copy()
            det["Raw"] = det["Raw"].round(4)
            det["Transformed"] = det["Transformed"].round(3)
            det["Level score"] = det["Level score"].round(1)
            det["Impulse score"] = det["Impulse score"].round(1)
            det["Weight"] = (100 * det["Weight"]).round(1)
            st.dataframe(det, use_container_width=True, hide_index=True)

        with st.expander("Stato download FRED / freschezza dati"):
            ms = macro_status.copy()
            if not ms.empty:
                for c in ["FirstDate", "LastDate"]:
                    ms[c] = pd.to_datetime(ms[c], errors="coerce").dt.date
                ms["LastValue"] = pd.to_numeric(ms["LastValue"], errors="coerce").round(4)
                st.dataframe(ms, use_container_width=True, hide_index=True)

        with st.expander("Metodologia Macro Leading V2.3"):
            st.markdown(
                """
Il motore usa **6 fattori × 2 dimensioni = 12 segnali**: Level e Impulse restano separati per US Growth, Global Growth, Inflation, Financial Stress, Liquidity e USD.  
Ogni indicatore produce un **Level robust Z-score** e un **Impulse** (35% variazione 1M + 65% slope 3M). Non vengono più compressi in un unico State: la direzione del cambiamento resta esplicita.  

Gli 8 regimi, inclusi **Debasement** e **Dollar Weakness**, sono archetipi economici 12D first-class. **Economic Score = 70%** del segnale finale. Il restante **30%** è una calibrazione empirica ridge walk-forward sui rendimenti futuri 1–3M degli 8 basket Equal Weight: a ogni data usa soltanto target già completamente osservabili (training fermo a t-3).  

L'output è un **Macro Regime Score 0–100**, non una probabilità non calibrata. La confidence dipende dal gap di score fra primo e secondo. Una EWM unidirezionale riduce l'erraticità.  

Il Macro Engine è interpretato operativamente soprattutto come **anticipatore 2–3M**; la validazione 1M resta visibile come controllo. In v2.5 alimenta il Decision Engine, che può proporre uno switch anticipato solo con convergenza robusta, confermata e vicina alla leadership Current di mercato.
                """
            )

with tab_decision:
    st.subheader("Decision Engine V1.1 — Market + Macro 2–3M")
    st.caption(
        "Il Market Engine resta l'ancora robusta. Il Decision Engine mantiene sempre un regime operativo e "
        "consente un early switch soltanto quando Macro, Economic/Empirical Agreement e Market Emerging convergono e il candidato è già vicino alla leadership Current."
    )
    if not decision:
        st.warning("Macro Engine non disponibile: nessuna decisione combinata.")
    else:
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        d1.metric("REGIME OPERATIVO", decision["OperationalRegime"], f"Market: {decision.get('MarketActive','—')}")
        d2.metric("MACRO NEXT 2–3M", decision["MacroNext"], f"Score {decision['MacroScore']:.1f} · {decision['MacroConfidence']}")
        d3.metric("MACRO AGREEMENT", decision.get("MacroAgreement", "N.D."), f"Eco {decision.get('EconomicLeader') or '—'} · Emp {decision.get('EmpiricalLeader') or '—'}")
        rank = decision.get("MacroCandidateCurrentRank")
        gap = decision.get("MacroCandidateGapToLeader", np.nan)
        prox_txt = f"#{rank}" if rank is not None else "N.D."
        prox_delta = f"Gap {gap:.1f}" if np.isfinite(gap) else ""
        d4.metric("MARKET PROXIMITY", prox_txt, prox_delta)
        d5.metric("TRANSITION SCORE", f"{decision.get('TransitionScore', np.nan):.0f}/100" if np.isfinite(decision.get('TransitionScore', np.nan)) else "N.D.")
        d6.metric("ACTION", decision["Action"], decision.get("SwitchType", ""))

        if decision["Action"] == "SWITCH":
            st.error(f"SWITCH → **{decision['OperationalRegime']}** · {decision['TransitionText']}")
        elif decision["Action"] == "PREPARE":
            st.warning(f"PREPARE · {decision['TransitionText']}")
        elif decision["Action"] == "WATCH":
            st.info(f"WATCH · {decision['TransitionText']}")
        else:
            st.success(f"HOLD · {decision['TransitionText']}")

        st.markdown("#### Regole Decision Engine V1.1")
        st.markdown(
            """
- **HOLD**: Macro 2–3M allineato al regime operativo oppure early regime già assunto e ancora supportato.  
- **WATCH**: divergenza Macro/Market senza convergenza sufficiente oppure confidence/agreement bassi.  
- **PREPARE**: Macro e Market Emerging/Leader convergono, ma manca ancora una conferma piena.  
- **SWITCH EARLY 2M**: stesso candidato per **2 mesi**, Market Emerging ≥60, Macro confidence ≥MEDIUM-HIGH, Macro Agreement ≥MEDIUM, Current candidato ≥60 e candidato **Top-3 Current oppure entro 10 punti dal Market Leader**.  
- **SWITCH QUICK EARLY**: caso raro, con confidence HIGH, agreement HIGH, Macro Score ≥75, Emerging ≥70, Current candidato ≥60 e candidato **Top-2 Current oppure entro 7 punti dal Market Leader**.  
- Un early regime non viene abbandonato per un solo mese rumoroso: servono **2 mesi di perdita congiunta di conferme** per tornare al Market Regime validato.
            """
        )

        if decision_val is not None and not decision_val.empty:
            st.markdown("#### Validazione 1M forward della decisione")
            vv = decision_val.copy()
            for c in ["DecisionAvg1M", "MarketAvg1M", "AvgDeltaBp", "DecisionBeatMarketPct"]:
                vv[c] = pd.to_numeric(vv[c], errors="coerce").round(2)
            vv.columns = ["Osservazioni", "Decision avg 1M %", "Market avg 1M %", "Delta medio bp", "Decision > Market %", "Mesi early"]
            st.dataframe(vv, use_container_width=True, hide_index=True)
            st.caption("Diagnostica ex-post: non alimenta né ottimizza le regole del Decision Engine.")

        st.markdown("#### Confronto Market / Macro")
        market_comp = latest.set_index("Regime")[["Current", "Emerging", "F", "Q", "RP", "A"]]
        macro_comp = macro_latest.set_index("Regime")[["MacroScore", "EconomicScore", "EmpiricalScore", "PredictedRelativeReturn", "Distance12D"]]
        comp = market_comp.join(macro_comp, how="left").reset_index()
        comp = comp.sort_values(["MacroScore", "Current"], ascending=False)
        for c in ["Current", "Emerging", "F", "Q", "RP", "A", "MacroScore", "EconomicScore", "EmpiricalScore"]:
            comp[c] = pd.to_numeric(comp[c], errors="coerce").round(1)
        comp["PredictedRelativeReturn"] = (100 * pd.to_numeric(comp["PredictedRelativeReturn"], errors="coerce")).round(2)
        comp["Distance12D"] = pd.to_numeric(comp["Distance12D"], errors="coerce").round(2)
        st.dataframe(comp.set_index("Regime"), use_container_width=True)

        if decision_history is not None and not decision_history.empty:
            st.markdown("#### Ultimi segnali Decision Engine")
            dh = decision_history.copy().sort_values("Date")
            cols = ["Date", "DecisionRegime", "MarketActive", "MacroNext", "MacroConfidence", "MacroAgreement", "MacroCandidateCurrent", "MacroCandidateCurrentRank", "MacroCandidateGapToLeader", "TransitionScore", "Action", "SwitchType", "Reason"]
            show = dh[cols].tail(24).copy()
            show["TransitionScore"] = pd.to_numeric(show["TransitionScore"], errors="coerce").round(0)
            st.dataframe(show.set_index("Date"), use_container_width=True)

with tab_history:
    st.subheader("Storia del regime operativo")
    path_cols = [
        "Date", "ActiveRegime", "ActiveScore", "MarketLeader", "LeaderScore",
        "RunnerUp", "RegimeSpread", "Confidence", "MarketState",
        "EmergingCandidate", "EmergingScore", "PendingCandidate", "PendingStreak"
    ]
    path = hist[path_cols].drop_duplicates("Date").sort_values("Date")
    path_display = path.copy()
    for c in ["ActiveScore", "LeaderScore", "RegimeSpread", "EmergingScore"]:
        path_display[c] = path_display[c].round(1)
    st.dataframe(path_display.tail(48).set_index("Date"), use_container_width=True)

    if macro_bundle is not None:
        st.subheader("Storico Macro Next")
        mh = macro_bundle["regimes"].copy()
        macro_path = (
            mh.sort_values(["Date", "MacroScore"], ascending=[True, False])
            .groupby("Date", as_index=False).first()
        )
        mcols = ["Date", "Regime", "MacroScore", "MacroRunnerUp", "MacroSpread", "MacroConfidence", "EmpiricalActive"]
        st.dataframe(macro_path[mcols].tail(48).set_index("Date"), use_container_width=True)

    # Excel-safe exports.
    export_hist = hist.copy()
    num_cols = export_hist.select_dtypes(include=[np.number]).columns
    export_hist[num_cols] = export_hist[num_cols].round(4)

    col_csv, col_xlsx = st.columns(2)
    with col_csv:
        csv_out = export_hist.to_csv(
            index=False, sep=";", decimal=",", float_format="%.4f",
            date_format="%d/%m/%Y", lineterminator="\n"
        ).encode("utf-8-sig")
        st.download_button(
            "Scarica regime_history.csv (Excel IT)", csv_out,
            "regime_history.csv", "text/csv"
        )
    with col_xlsx:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl", datetime_format="DD/MM/YYYY") as writer:
            export_hist.to_excel(writer, index=False, sheet_name="Market Regime History")
            export_path = path.copy()
            num_path = export_path.select_dtypes(include=[np.number]).columns
            export_path[num_path] = export_path[num_path].round(4)
            export_path.to_excel(writer, index=False, sheet_name="Market Active Path")
            if macro_bundle is not None:
                macro_bundle["regimes"].to_excel(writer, index=False, sheet_name="Macro Regime History")
                macro_bundle["factors"].to_excel(writer, index=False, sheet_name="Macro Factors")
                macro_bundle.get("validation", pd.DataFrame()).to_excel(writer, index=False, sheet_name="Macro Validation")
                macro_bundle.get("empirical_predictions", pd.DataFrame()).reset_index(names="Date").to_excel(writer, index=False, sheet_name="Empirical Predictions")
                if decision_history is not None and not decision_history.empty:
                    decision_history.to_excel(writer, index=False, sheet_name="Decision History")
                if decision_val is not None and not decision_val.empty:
                    decision_val.to_excel(writer, index=False, sheet_name="Decision Validation")
                macro_status.to_excel(writer, index=False, sheet_name="FRED Status")
        st.download_button(
            "Scarica detector_history.xlsx", xlsx_buf.getvalue(),
            "detector_history.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.caption(
        "CSV: ';' come separatore e ',' come decimale per Excel italiano. "
        "L'XLSX contiene Market history, Macro 12D history, fattori, validazione walk-forward, Decision History/Validation, previsioni empiriche e stato FRED."
    )
