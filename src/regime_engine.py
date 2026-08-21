from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import technical_history


def _robust_z(s: pd.Series, window: int = 60, min_periods: int = 18) -> pd.Series:
    def one(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < max(6, min_periods // 2):
            return np.nan
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad <= 1e-12:
            sd = np.std(x)
            return 0.0 if sd <= 1e-12 else (x[-1] - med) / sd
        return (x[-1] - med) / (1.4826 * mad)

    return s.rolling(window, min_periods=min_periods).apply(one, raw=True).clip(-2.5, 2.5)


def _z_to_score(z: pd.Series) -> pd.Series:
    return (50 + 20 * z).clip(0, 100)


def _relative_performance_scores(prices: pd.DataFrame) -> pd.DataFrame:
    # Monthly observations are indexed by the actual last date present in the file,
    # not by the calendar month-end. This is essential for partial current months.
    p = prices.sort_index().ffill()
    monthly_rows = [g.iloc[[-1]] for _, g in p.groupby(p.index.to_period("M"))]
    m = pd.concat(monthly_rows).sort_index() if monthly_rows else p.iloc[0:0].copy()
    scores = {}
    horizon_weights = {1: 0.20, 3: 0.50, 6: 0.30}
    for col in m.columns:
        total = None
        for h, w in horizon_weights.items():
            ret = m[col].pct_change(h)
            universe_ret = m.pct_change(h).mean(axis=1)
            excess = ret - universe_ret
            sc = _z_to_score(_robust_z(excess))
            total = w * sc if total is None else total + w * sc
        scores[col] = total
    return pd.DataFrame(scores)


def _acceleration_score(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index, columns=f.columns, dtype=float)
    for col in f.columns:
        d1 = f[col].diff(1)
        d3 = (f[col] - f[col].shift(3)) / 3.0
        s1 = _z_to_score(_robust_z(d1, 36, 12))
        s3 = _z_to_score(_robust_z(d3, 36, 12))
        out[col] = 0.40 * s1 + 0.60 * s3
    return out


def build_market_engine(
    basket_ohlc: dict[str, pd.DataFrame],
    basket_prices: pd.DataFrame,
    current_weights=(0.45, 0.30, 0.20, 0.05),
    emerging_weights=(0.25, 0.10, 0.25, 0.40),
    normal_margin: float = 7.0,
    confirm_months: int = 2,
    quick_current: float = 75.0,
    quick_accel: float = 70.0,
    quick_q: float = 60.0,
    min_current_switch: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-form regime history and latest snapshot."""
    tech = {}
    for name, df in basket_ohlc.items():
        h = technical_history(df)
        if not h.empty:
            tech[name] = h
    if not tech:
        raise ValueError("Nessuna serie con storico sufficiente per il calcolo tecnico.")

    common_names = [c for c in basket_prices.columns if c in tech]
    if len(common_names) < 2:
        raise ValueError("Servono almeno due basket con dati validi.")

    idx = sorted(set().union(*[set(tech[n].index) for n in common_names]))
    idx = pd.DatetimeIndex(idx)
    F = pd.DataFrame(index=idx, columns=common_names, dtype=float)
    Q = pd.DataFrame(index=idx, columns=common_names, dtype=float)
    for n in common_names:
        F[n] = tech[n]["F"].reindex(idx).ffill()
        Q[n] = tech[n]["Q"].reindex(idx).ffill()

    RP = _relative_performance_scores(basket_prices[common_names]).reindex(idx).ffill()
    A = _acceleration_score(F)

    cw = current_weights
    ew = emerging_weights
    current = cw[0] * F + cw[1] * Q + cw[2] * RP + cw[3] * A
    emerging = ew[0] * F + ew[1] * Q + ew[2] * RP + ew[3] * A

    rows = []
    for dt in idx:
        for n in common_names:
            vals = [F.at[dt, n], Q.at[dt, n], RP.at[dt, n], A.at[dt, n], current.at[dt, n], emerging.at[dt, n]]
            if not all(np.isfinite(v) for v in vals):
                continue
            cur, em, acc, q = current.at[dt, n], emerging.at[dt, n], A.at[dt, n], Q.at[dt, n]
            if cur >= 65 and q >= 60:
                state = "DOMINANT"
            elif cur >= 60 and acc < 45:
                state = "MATURE"
            elif em >= 65 and acc >= 60:
                state = "EMERGING"
            elif cur < 45 and acc < 50:
                state = "WEAK"
            else:
                state = "NEUTRAL"
            rows.append(
                {
                    "Date": dt,
                    "Regime": n,
                    "F": F.at[dt, n],
                    "Q": Q.at[dt, n],
                    "RP": RP.at[dt, n],
                    "A": A.at[dt, n],
                    "Current": cur,
                    "Emerging": em,
                    "State": state,
                }
            )

    hist = pd.DataFrame(rows)
    if hist.empty:
        raise ValueError("Storico insufficiente dopo la normalizzazione robusta.")
    hist = apply_hysteresis(
        hist,
        normal_margin=normal_margin,
        confirm_months=confirm_months,
        quick_current=quick_current,
        quick_accel=quick_accel,
        quick_q=quick_q,
        min_current_switch=min_current_switch,
    )
    last = hist[hist["Date"] == hist["Date"].max()].copy()
    return hist, last


def _confidence_from_spread(spread: float) -> str:
    """Qualitative confidence based on Current-score separation of top two regimes."""
    if not np.isfinite(spread):
        return "N.D."
    if spread > 15:
        return "HIGH"
    if spread >= 8:
        return "MEDIUM-HIGH"
    if spread >= 4:
        return "MEDIUM"
    return "LOW"


def apply_hysteresis(
    hist: pd.DataFrame,
    normal_margin: float = 7.0,
    confirm_months: int = 2,
    quick_current: float = 75.0,
    quick_accel: float = 70.0,
    quick_q: float = 60.0,
    min_current_switch: float = 60.0,
) -> pd.DataFrame:
    """Apply sticky regime selection and attach confidence / transition diagnostics.

    Normal switches require all of the following:
      * the challenger is the Current-score market leader;
      * challenger Current >= ``min_current_switch``;
      * challenger leads the active regime by ``normal_margin`` points;
      * the condition persists for ``confirm_months`` observations.

    Quick switches preserve the V1 thresholds and do not require the normal
    confirmation streak.  The top-2 Current spread is retained as a separate
    confidence diagnostic and does not itself force a switch.
    """
    h = hist.copy().sort_values(["Date", "Regime"])
    dates = sorted(h["Date"].unique())
    active = None
    candidate = None
    streak = 0
    records = []

    for dt in dates:
        x = h[h["Date"] == dt].copy().sort_values("Current", ascending=False)
        if x.empty:
            continue

        top = x.iloc[0]
        runner = x.iloc[1] if len(x) > 1 else None
        leader = str(top["Regime"])
        leader_score = float(top["Current"])
        runner_name = str(runner["Regime"]) if runner is not None else None
        runner_score = float(runner["Current"]) if runner is not None else np.nan
        spread = leader_score - runner_score if runner is not None else np.nan
        confidence = _confidence_from_spread(spread)

        if active is None:
            # Bootstrap on the first usable observation.  Low-conviction starts
            # are flagged explicitly rather than suppressed, otherwise no active
            # path could ever be formed.
            active = leader

        elif leader != active:
            active_row = x[x["Regime"] == active]
            active_score = float(active_row["Current"].iloc[0]) if not active_row.empty else -np.inf
            margin = leader_score - active_score
            quick = (
                leader_score >= quick_current
                and float(top["A"]) >= quick_accel
                and float(top["Q"]) >= quick_q
            )
            normal_eligible = leader_score >= min_current_switch

            if quick:
                active = leader
                candidate, streak = None, 0
            elif normal_eligible and margin >= normal_margin:
                if candidate == leader:
                    streak += 1
                else:
                    candidate, streak = leader, 1
                if streak >= confirm_months:
                    active = candidate
                    candidate, streak = None, 0
            else:
                # A weak leader (e.g. Current 47) must not become the active
                # regime just because everything else is weaker.
                candidate, streak = None, 0
        else:
            candidate, streak = None, 0

        em = x[x["Regime"] != active].sort_values("Emerging", ascending=False)
        emerging = em.iloc[0]["Regime"] if not em.empty else None
        emerging_score = float(em.iloc[0]["Emerging"]) if not em.empty else np.nan
        active_row_now = x[x["Regime"] == active]
        active_score = float(active_row_now["Current"].iloc[0]) if not active_row_now.empty else np.nan

        # Operational regime is ALWAYS retained.  State is a validity / transition
        # diagnostic, never an empty-regime state.
        if candidate is not None and streak > 0:
            market_state = "TRANSITION"
        elif np.isfinite(active_score) and active_score < 50:
            market_state = "TRANSITION"
        elif np.isfinite(active_score) and active_score < 60:
            market_state = "WATCH"
        else:
            market_state = "STABLE"

        records.append(
            {
                "Date": dt,
                "ActiveRegime": active,
                "ActiveScore": active_score,
                "MarketLeader": leader,
                "LeaderScore": leader_score,
                "RunnerUp": runner_name,
                "RunnerUpScore": runner_score,
                "RegimeSpread": spread,
                "Confidence": confidence,
                "MarketState": market_state,
                "EmergingCandidate": emerging,
                "EmergingScore": emerging_score,
                "PendingCandidate": candidate,
                "PendingStreak": streak,
            }
        )

    path = pd.DataFrame(records)
    h = h.merge(path, on="Date", how="left")
    return h
