from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .data_sources import REGIMES


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    label: str
    factor: str
    transform: str
    direction: float
    weight: float


FACTOR_NAMES = [
    "US Growth",
    "Global Growth",
    "Inflation",
    "Financial Stress",
    "Liquidity",
    "USD",
]

# Higher signed values always mean MORE of the factor named above:
# more growth, more inflation, more stress, more liquidity, stronger USD.
INDICATORS = [
    # US Growth
    IndicatorSpec("USALOLITOAASTSAM", "OECD CLI USA", "US Growth", "cli", +1, 0.18),
    IndicatorSpec("IC4WSA", "Initial Claims 4W", "US Growth", "pct3", -1, 0.12),
    IndicatorSpec("CCSA", "Continued Claims", "US Growth", "pct3", -1, 0.08),
    IndicatorSpec("PERMIT", "Building Permits", "US Growth", "pct6", +1, 0.12),
    IndicatorSpec("INDPRO", "Industrial Production", "US Growth", "ann3", +1, 0.14),
    IndicatorSpec("TCU", "Capacity Utilization", "US Growth", "level", +1, 0.10),
    IndicatorSpec("RSAFS", "Retail Sales", "US Growth", "ann3", +1, 0.10),
    IndicatorSpec("UMCSENT", "Consumer Sentiment", "US Growth", "level", +1, 0.08),
    IndicatorSpec("PCEC96", "Real Personal Consumption", "US Growth", "ann3", +1, 0.08),

    # Global Growth
    IndicatorSpec("G7LOLITOAASTSAM", "OECD CLI G7", "Global Growth", "cli", +1, 0.45),
    IndicatorSpec("CHNLOLITOAASTSAM", "OECD CLI Cina", "Global Growth", "cli", +1, 0.35),
    IndicatorSpec("PCOPPUSDM", "Copper Momentum", "Global Growth", "pct3", +1, 0.20),

    # Inflation
    IndicatorSpec("CPIAUCSL", "Headline CPI 3M ann.", "Inflation", "ann3", +1, 0.12),
    IndicatorSpec("CPILFESL", "Core CPI 3M ann.", "Inflation", "ann3", +1, 0.17),
    IndicatorSpec("PCEPILFE", "Core PCE 3M ann.", "Inflation", "ann3", +1, 0.16),
    IndicatorSpec("WPSFD49116", "Core PPI 3M ann.", "Inflation", "ann3", +1, 0.10),
    IndicatorSpec("CES0500000003", "Wage Growth", "Inflation", "yoy", +1, 0.11),
    IndicatorSpec("T5YIFR", "5Y5Y Inflation Expectations", "Inflation", "level", +1, 0.10),
    IndicatorSpec("T10YIE", "10Y Breakeven", "Inflation", "level", +1, 0.07),
    IndicatorSpec("PALLFNFINDEXM", "Commodity Index", "Inflation", "pct3", +1, 0.07),
    IndicatorSpec("DCOILWTICO", "WTI Momentum", "Inflation", "pct3", +1, 0.05),
    IndicatorSpec("PCOPPUSDM", "Copper Momentum", "Inflation", "pct3", +1, 0.05),

    # Financial Stress
    IndicatorSpec("NFCI", "NFCI", "Financial Stress", "level", +1, 0.17),
    IndicatorSpec("ANFCI", "Adjusted NFCI", "Financial Stress", "level", +1, 0.13),
    IndicatorSpec("STLFSI4", "St. Louis Financial Stress", "Financial Stress", "level", +1, 0.13),
    IndicatorSpec("BAA10Y", "Baa-Treasury Spread", "Financial Stress", "level", +1, 0.17),
    IndicatorSpec("T10Y3M", "10Y-3M Curve (inverted)", "Financial Stress", "level", -1, 0.10),
    IndicatorSpec("DFII10", "10Y Real Yield", "Financial Stress", "level", +1, 0.10),
    IndicatorSpec("VIXCLS", "VIX", "Financial Stress", "level", +1, 0.08),
    IndicatorSpec("REAL_POLICY", "Real Policy Rate", "Financial Stress", "level", +1, 0.12),

    # Liquidity
    IndicatorSpec("NET_LIQ", "Fed Net Liquidity", "Liquidity", "pct3", +1, 0.28),
    IndicatorSpec("WRESBAL", "Reserve Balances", "Liquidity", "pct3", +1, 0.18),
    IndicatorSpec("M2SL", "M2", "Liquidity", "pct6", +1, 0.22),
    IndicatorSpec("BOGMBASE", "Monetary Base", "Liquidity", "pct3", +1, 0.14),
    IndicatorSpec("TLAACBW027SBOG", "Commercial Bank Assets", "Liquidity", "pct6", +1, 0.18),

    # USD strength
    IndicatorSpec("DTWEXBGS", "Broad USD 3M", "USD", "pct3", +1, 0.35),
    IndicatorSpec("DTWEXAFEGS", "Advanced FX USD 3M", "USD", "pct3", +1, 0.20),
    IndicatorSpec("DGS2", "US 2Y Yield", "USD", "level", +1, 0.15),
    IndicatorSpec("REL_G7_US", "G7 vs US Growth", "USD", "level", -1, 0.18),
    IndicatorSpec("REL_CHINA_US", "China vs US Growth", "USD", "level", -1, 0.12),
]

# V2.3: Level and Impulse stay separate.  These splits control only the
# distance weighting, NOT an averaging of the two signals.
FACTOR_DIMENSION_SPLIT = {
    "US Growth": (0.35, 0.65),
    "Global Growth": (0.35, 0.65),
    "Inflation": (0.35, 0.65),
    "Financial Stress": (0.45, 0.55),
    "Liquidity": (0.35, 0.65),
    "USD": (0.40, 0.60),
}

# 12-dimensional archetypes: (Level target, Impulse target) for every factor.
# These are economic priors, intentionally not optimized on 2021-2026 basket returns.
REGIME_ARCHETYPES_12D: Dict[str, dict] = {
    "Recession": {
        "target": {
            "US Growth": (-1.00, -1.35), "Global Growth": (-0.75, -1.00),
            "Inflation": (-0.10, -0.35), "Financial Stress": (0.90, 1.25),
            "Liquidity": (-0.20, -0.35), "USD": (0.35, 0.45),
        },
        "weight": {"US Growth": .28, "Global Growth": .14, "Inflation": .12, "Financial Stress": .30, "Liquidity": .10, "USD": .06},
    },
    "Debasement": {
        "target": {
            "US Growth": (0.05, 0.15), "Global Growth": (0.10, 0.20),
            "Inflation": (0.65, 0.80), "Financial Stress": (0.00, -0.10),
            "Liquidity": (0.95, 1.25), "USD": (-0.65, -0.95),
        },
        "weight": {"US Growth": .08, "Global Growth": .08, "Inflation": .22, "Financial Stress": .08, "Liquidity": .32, "USD": .22},
    },
    "Stagflation": {
        "target": {
            "US Growth": (-0.55, -0.90), "Global Growth": (-0.45, -0.70),
            "Inflation": (1.00, 1.25), "Financial Stress": (0.45, 0.65),
            "Liquidity": (-0.10, -0.25), "USD": (0.15, 0.20),
        },
        "weight": {"US Growth": .24, "Global Growth": .12, "Inflation": .32, "Financial Stress": .18, "Liquidity": .08, "USD": .06},
    },
    "Reflation": {
        "target": {
            "US Growth": (0.55, 1.00), "Global Growth": (0.50, 0.85),
            "Inflation": (0.40, 0.80), "Financial Stress": (-0.40, -0.55),
            "Liquidity": (0.45, 0.80), "USD": (-0.15, -0.40),
        },
        "weight": {"US Growth": .24, "Global Growth": .19, "Inflation": .20, "Financial Stress": .14, "Liquidity": .15, "USD": .08},
    },
    "Dollar Weakness": {
        "target": {
            "US Growth": (0.05, -0.20), "Global Growth": (0.55, 0.80),
            "Inflation": (0.05, 0.10), "Financial Stress": (-0.30, -0.35),
            "Liquidity": (0.40, 0.65), "USD": (-0.95, -1.30),
        },
        "weight": {"US Growth": .08, "Global Growth": .20, "Inflation": .07, "Financial Stress": .09, "Liquidity": .20, "USD": .36},
    },
    "Goldilocks Economy": {
        "target": {
            "US Growth": (0.70, 0.35), "Global Growth": (0.55, 0.25),
            "Inflation": (-0.30, -0.50), "Financial Stress": (-0.65, -0.45),
            "Liquidity": (0.20, 0.15), "USD": (-0.05, -0.10),
        },
        "weight": {"US Growth": .29, "Global Growth": .18, "Inflation": .20, "Financial Stress": .21, "Liquidity": .07, "USD": .05},
    },
    "Disinflation / Soft Landing": {
        "target": {
            "US Growth": (0.30, -0.15), "Global Growth": (0.20, -0.10),
            "Inflation": (-0.65, -1.15), "Financial Stress": (-0.35, -0.15),
            "Liquidity": (0.05, 0.00), "USD": (0.05, 0.00),
        },
        "weight": {"US Growth": .25, "Global Growth": .11, "Inflation": .34, "Financial Stress": .20, "Liquidity": .06, "USD": .04},
    },
    "Deflation": {
        "target": {
            "US Growth": (-0.85, -1.15), "Global Growth": (-0.80, -1.05),
            "Inflation": (-1.00, -1.30), "Financial Stress": (0.65, 0.90),
            "Liquidity": (-0.20, -0.30), "USD": (0.35, 0.45),
        },
        "weight": {"US Growth": .25, "Global Growth": .18, "Inflation": .27, "Financial Stress": .20, "Liquidity": .06, "USD": .04},
    },
}


def _robust_z(s: pd.Series, window: int = 120, min_periods: int = 36) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")

    def one(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < max(8, min_periods // 2):
            return np.nan
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad <= 1e-12:
            sd = np.std(x)
            return 0.0 if sd <= 1e-12 else (x[-1] - med) / sd
        return (x[-1] - med) / (1.4826 * mad)

    return s.rolling(window, min_periods=min_periods).apply(one, raw=True).clip(-3.0, 3.0)


def _transform(s: pd.Series, kind: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if kind == "level":
        return s
    if kind == "cli":
        return s - 100.0
    if kind == "pct3":
        return 100.0 * s.pct_change(3)
    if kind == "pct6":
        return 100.0 * s.pct_change(6)
    if kind == "yoy":
        return 100.0 * s.pct_change(12)
    if kind == "ann3":
        ratio = s / s.shift(3)
        return 100.0 * (ratio.pow(4.0) - 1.0)
    raise ValueError(f"Transform non riconosciuta: {kind}")


def _weighted_mean_frame(values: list[pd.Series], weights: list[float]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    df = pd.concat(values, axis=1)
    w = np.asarray(weights, dtype=float)
    arr = df.to_numpy(dtype=float)
    valid = np.isfinite(arr)
    numer = np.nansum(arr * w[None, :], axis=1)
    denom = np.sum(valid * w[None, :], axis=1)
    out = np.divide(numer, denom, out=np.full(len(df), np.nan), where=denom > 0)
    return pd.Series(out, index=df.index)


def _add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy().sort_index()
    if {"WALCL", "WTREGEN", "RRPONTSYD"}.issubset(x.columns):
        x["NET_LIQ"] = x["WALCL"] - x["WTREGEN"] - 1000.0 * x["RRPONTSYD"]
    if {"DFF", "CPILFESL"}.issubset(x.columns):
        core_yoy = 100.0 * x["CPILFESL"].pct_change(12)
        x["REAL_POLICY"] = x["DFF"] - core_yoy
    if {"G7LOLITOAASTSAM", "USALOLITOAASTSAM"}.issubset(x.columns):
        x["REL_G7_US"] = (x["G7LOLITOAASTSAM"] - 100.0) - (x["USALOLITOAASTSAM"] - 100.0)
    if {"CHNLOLITOAASTSAM", "USALOLITOAASTSAM"}.issubset(x.columns):
        x["REL_CHINA_US"] = (x["CHNLOLITOAASTSAM"] - 100.0) - (x["USALOLITOAASTSAM"] - 100.0)
    return x


def _make_dimension_frame(level: pd.DataFrame, impulse: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=level.index)
    for f in FACTOR_NAMES:
        out[f"{f}|L"] = level[f]
        out[f"{f}|I"] = impulse[f]
    return out


def _economic_scores(level: pd.DataFrame, impulse: pd.DataFrame, sigma: float = 1.10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist = pd.DataFrame(index=level.index, columns=REGIMES, dtype=float)
    similarity = pd.DataFrame(index=level.index, columns=REGIMES, dtype=float)

    for regime in REGIMES:
        arch = REGIME_ARCHETYPES_12D[regime]
        vals: list[pd.Series] = []
        wts: list[float] = []
        for f in FACTOR_NAMES:
            base_w = float(arch["weight"][f])
            wl, wi = FACTOR_DIMENSION_SPLIT[f]
            tgt_l, tgt_i = arch["target"][f]
            vals.append((level[f] - float(tgt_l)).pow(2))
            wts.append(base_w * wl)
            vals.append((impulse[f] - float(tgt_i)).pow(2))
            wts.append(base_w * wi)
        d2 = _weighted_mean_frame(vals, wts)
        dist[regime] = np.sqrt(d2)
        similarity[regime] = np.exp(-0.5 * d2 / (sigma * sigma))

    score = 100.0 * similarity.div(similarity.max(axis=1), axis=0)
    return score, dist, similarity


def _monthly_basket_target(basket_prices: pd.DataFrame) -> pd.DataFrame:
    """Future 1-3M monthly-equivalent relative return target.

    30% next month + 40% two-month monthly-equivalent + 30% three-month
    monthly-equivalent.  The cross-sectional mean is removed each month so the
    empirical layer learns which regime basket should outperform the other 7.
    """
    p = basket_prices.copy().sort_index()
    p.index = pd.to_datetime(p.index)
    p = p[[c for c in REGIMES if c in p.columns]].resample("ME").last()
    if len(p.columns) < 2:
        return pd.DataFrame()
    r1 = p.shift(-1) / p - 1.0
    r2c = p.shift(-2) / p - 1.0
    r3c = p.shift(-3) / p - 1.0
    r2 = (1.0 + r2c).pow(1.0 / 2.0) - 1.0
    r3 = (1.0 + r3c).pow(1.0 / 3.0) - 1.0
    y = 0.30 * r1 + 0.40 * r2 + 0.30 * r3
    return y.sub(y.mean(axis=1), axis=0)


def _ridge_walk_forward(
    features: pd.DataFrame,
    basket_prices: pd.DataFrame | None,
    ridge_lambda: float = 8.0,
    min_train: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict walk-forward ridge prediction of future relative basket returns.

    At month t, training rows stop at t-3 so all 1/2/3M targets used for fitting
    were already observable.  No current/future basket return enters the fit.
    """
    empty_pred = pd.DataFrame(index=features.index, columns=REGIMES, dtype=float)
    empty_count = pd.DataFrame(index=features.index, columns=REGIMES, dtype=float)
    if basket_prices is None or basket_prices.empty:
        return empty_pred, empty_count

    target = _monthly_basket_target(basket_prices)
    if target.empty:
        return empty_pred, empty_count

    X = features.copy()
    X.index = pd.to_datetime(X.index).to_period("M")
    pred = pd.DataFrame(index=X.index, columns=REGIMES, dtype=float)
    train_count = pd.DataFrame(index=X.index, columns=REGIMES, dtype=float)
    Y = target.copy()
    Y.index = pd.to_datetime(Y.index).to_period("M")

    for period in X.index:
        cutoff = period - 3
        train_periods = X.index[(X.index <= cutoff) & X.index.isin(Y.index)]
        if len(train_periods) < min_train:
            continue
        x_now = X.loc[period]
        if not np.isfinite(x_now.to_numpy(dtype=float)).all():
            continue

        for regime in REGIMES:
            if regime not in Y.columns:
                continue
            y = Y[regime].reindex(train_periods)
            x = X.loc[train_periods]
            ok = y.notna() & x.notna().all(axis=1)
            n = int(ok.sum())
            train_count.loc[X.index == period, regime] = n
            if n < min_train:
                continue

            A = x.loc[ok].to_numpy(dtype=float)
            b = y.loc[ok].to_numpy(dtype=float)
            mu = A.mean(axis=0)
            sd = A.std(axis=0)
            sd[sd < 1e-6] = 1.0
            Z = (A - mu) / sd
            b_mean = float(b.mean())
            bc = b - b_mean
            lhs = Z.T @ Z + float(ridge_lambda) * np.eye(Z.shape[1])
            beta = np.linalg.solve(lhs, Z.T @ bc)
            z_now = (x_now.to_numpy(dtype=float) - mu) / sd
            pred_value = b_mean + float(z_now @ beta)
            pred.loc[X.index == period, regime] = pred_value

    # Convert period index back to month-end timestamps matching factor history.
    pred.index = pred.index.to_timestamp("M")
    train_count.index = train_count.index.to_timestamp("M")
    pred = pred.groupby(pred.index).last().reindex(features.index)
    train_count = train_count.groupby(train_count.index).last().reindex(features.index)
    return pred, train_count


def _empirical_score(pred: pd.DataFrame, scale: float = 0.02) -> pd.DataFrame:
    # 2% expected monthly-equivalent relative return is intentionally a strong
    # signal. tanh keeps the layer bounded without rank-forcing separation.
    return 50.0 + 50.0 * np.tanh(pred / float(scale))


def _confidence_from_gap(gap: float) -> str:
    if not np.isfinite(gap):
        return "N.D."
    if gap >= 10.0:
        return "HIGH"
    if gap >= 6.0:
        return "MEDIUM-HIGH"
    if gap >= 3.0:
        return "MEDIUM"
    return "LOW"


def _validation_table(score: pd.DataFrame, basket_prices: pd.DataFrame | None) -> pd.DataFrame:
    if basket_prices is None or basket_prices.empty:
        return pd.DataFrame()
    p = basket_prices.copy().sort_index()
    p.index = pd.to_datetime(p.index)
    cols = [c for c in REGIMES if c in p.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    p = p[cols].resample("ME").last()
    score_m = score.copy()
    score_m.index = pd.to_datetime(score_m.index).to_period("M")
    p.index = p.index.to_period("M")

    rows = []
    for h in (1, 2, 3):
        future = p.shift(-h) / p - 1.0
        common = score_m.index.intersection(future.index)
        n = hit1 = hit3 = 0
        rho_vals = []
        for dt in common:
            s = score_m.loc[dt, cols].dropna()
            r = future.loc[dt, cols].dropna()
            ci = s.index.intersection(r.index)
            if len(ci) < len(cols):
                continue
            leader = s[ci].idxmax()
            ranks = r[ci].rank(ascending=False, method="average")
            hit1 += int(ranks[leader] == 1)
            hit3 += int(ranks[leader] <= 3)
            rho = s[ci].rank().corr(r[ci].rank())
            if np.isfinite(rho):
                rho_vals.append(float(rho))
            n += 1
        rows.append({
            "Horizon": f"{h}M",
            "Observations": n,
            "Top1HitPct": 100.0 * hit1 / n if n else np.nan,
            "Top3HitPct": 100.0 * hit3 / n if n else np.nan,
            "RankCorrelation": float(np.mean(rho_vals)) if rho_vals else np.nan,
        })
    return pd.DataFrame(rows)


def build_macro_engine(
    panel: pd.DataFrame,
    basket_prices: pd.DataFrame | None = None,
    empirical_weight: float = 0.30,
    ridge_lambda: float = 8.0,
    min_train: int = 36,
    score_alpha: float = 0.55,
) -> dict:
    """Macro Leading Engine V2.3.

    Core changes vs V2.2:
    - 12 dimensions: Level and Impulse remain separate for all six factors.
    - Output is a Macro Regime Score 0-100, not an uncalibrated probability.
    - Optional 30% empirical layer: strictly walk-forward ridge calibration on
      the eight Equal Weight regime baskets.  Economic priors always remain 70%.
    - Confidence uses the score gap between first and second regime.
    """
    if panel is None or panel.empty:
        raise ValueError("Nessun dato macro disponibile.")

    data = _add_derived(panel).sort_index().ffill()

    indicator_rows = []
    factor_level: dict[str, pd.Series] = {}
    factor_impulse: dict[str, pd.Series] = {}
    factor_coverage: dict[str, pd.Series] = {}

    for factor in FACTOR_NAMES:
        specs = [s for s in INDICATORS if s.factor == factor and s.key in data.columns]
        level_parts, impulse_parts, weights = [], [], []
        for spec in specs:
            raw = data[spec.key]
            transformed = _transform(raw, spec.transform)
            level_z = spec.direction * _robust_z(transformed, 120, 36)
            d1 = transformed.diff(1)
            d3 = (transformed - transformed.shift(3)) / 3.0
            imp1 = _robust_z(d1, 60, 18)
            imp3 = _robust_z(d3, 60, 18)
            impulse_z = spec.direction * (0.35 * imp1 + 0.65 * imp3)

            level_parts.append(level_z.rename(spec.key))
            impulse_parts.append(impulse_z.rename(spec.key))
            weights.append(spec.weight)

            indicator_rows.append(pd.DataFrame({
                "Date": data.index,
                "Indicator": spec.label,
                "Key": spec.key,
                "Factor": factor,
                "Raw": raw.to_numpy(),
                "Transformed": transformed.to_numpy(),
                "LevelZ": level_z.to_numpy(),
                "ImpulseZ": impulse_z.to_numpy(),
                "Weight": spec.weight,
            }))

        if level_parts:
            factor_level[factor] = _weighted_mean_frame(level_parts, weights)
            factor_impulse[factor] = _weighted_mean_frame(impulse_parts, weights)
            avail = pd.concat(level_parts, axis=1).notna().astype(float)
            w = np.asarray(weights, dtype=float)
            cov = (avail.to_numpy() * w[None, :]).sum(axis=1) / w.sum()
            factor_coverage[factor] = pd.Series(cov, index=avail.index)
        else:
            factor_level[factor] = pd.Series(index=data.index, dtype=float)
            factor_impulse[factor] = pd.Series(index=data.index, dtype=float)
            factor_coverage[factor] = pd.Series(0.0, index=data.index)

    L = pd.DataFrame(factor_level).reindex(data.index)
    I = pd.DataFrame(factor_impulse).reindex(data.index)
    COV = pd.DataFrame(factor_coverage).reindex(data.index)
    X12 = _make_dimension_frame(L, I)

    economic_score, dist, similarity = _economic_scores(L, I)
    pred_excess, train_count = _ridge_walk_forward(X12, basket_prices, ridge_lambda=ridge_lambda, min_train=min_train)
    empirical_score = _empirical_score(pred_excess)

    ew = float(np.clip(empirical_weight, 0.0, 0.50))
    combined_raw = economic_score.copy()
    empirical_available = empirical_score.notna()
    for r in REGIMES:
        mask = empirical_available[r]
        combined_raw.loc[mask, r] = (1.0 - ew) * economic_score.loc[mask, r] + ew * empirical_score.loc[mask, r]

    # One-sided smoothing only; no future information.
    macro_score = combined_raw.ewm(alpha=float(score_alpha), adjust=False, min_periods=1).mean()

    factor_rows = []
    for dt in L.index:
        for f in FACTOR_NAMES:
            factor_rows.append({
                "Date": dt,
                "Factor": f,
                "LevelZ": L.at[dt, f],
                "ImpulseZ": I.at[dt, f],
                "LevelScore": np.clip(50 + 18 * L.at[dt, f], 0, 100) if np.isfinite(L.at[dt, f]) else np.nan,
                "ImpulseScore": np.clip(50 + 18 * I.at[dt, f], 0, 100) if np.isfinite(I.at[dt, f]) else np.nan,
                "Coverage": COV.at[dt, f],
            })

    regime_rows = []
    for dt in macro_score.index:
        row = macro_score.loc[dt].dropna().sort_values(ascending=False)
        leader = row.index[0] if len(row) else None
        runner = row.index[1] if len(row) > 1 else None
        spread = float(row.iloc[0] - row.iloc[1]) if len(row) > 1 else np.nan
        conf = _confidence_from_gap(spread)
        for r in REGIMES:
            pred = pred_excess.at[dt, r] if dt in pred_excess.index and r in pred_excess.columns else np.nan
            regime_rows.append({
                "Date": dt,
                "Regime": r,
                "MacroScore": macro_score.at[dt, r],
                "MacroScoreRaw": combined_raw.at[dt, r],
                "EconomicScore": economic_score.at[dt, r],
                "EmpiricalScore": empirical_score.at[dt, r] if dt in empirical_score.index else np.nan,
                "PredictedRelativeReturn": pred,
                "EmpiricalTrainN": train_count.at[dt, r] if dt in train_count.index else np.nan,
                "EmpiricalActive": bool(np.isfinite(pred)),
                "Distance12D": dist.at[dt, r],
                "MacroLeader": leader,
                "MacroRunnerUp": runner,
                "MacroSpread": spread,
                "MacroConfidence": conf,
            })

    indicators_long = pd.concat(indicator_rows, ignore_index=True) if indicator_rows else pd.DataFrame()
    factors_long = pd.DataFrame(factor_rows)
    regimes_long = pd.DataFrame(regime_rows)

    valid_dates = regimes_long.dropna(subset=["MacroScore"])["Date"]
    if valid_dates.empty:
        raise ValueError("Storico macro insufficiente per normalizzazione e scoring.")
    last_date = pd.Timestamp(valid_dates.max())
    latest_regimes = regimes_long[regimes_long["Date"] == last_date].copy().sort_values("MacroScore", ascending=False)
    latest_factors = factors_long[factors_long["Date"] == last_date].copy()
    latest_indicators = indicators_long[indicators_long["Date"] == last_date].copy() if not indicators_long.empty else pd.DataFrame()
    validation = _validation_table(macro_score, basket_prices)

    return {
        "panel": data,
        "indicators": indicators_long,
        "factors": factors_long,
        "regimes": regimes_long,
        "latest_regimes": latest_regimes,
        "latest_factors": latest_factors,
        "latest_indicators": latest_indicators,
        "last_date": last_date,
        "economic_score": economic_score,
        "empirical_score": empirical_score,
        "macro_score": macro_score,
        "empirical_predictions": pred_excess,
        "validation": validation,
        "empirical_weight": ew,
    }




_CONF_RANK = {"N.D.": 0, "LOW": 1, "MEDIUM": 2, "MEDIUM-HIGH": 3, "HIGH": 4}
_AGR_RANK = {"N.D.": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def _rank_label(value: str, mapping: dict[str, int]) -> int:
    return int(mapping.get(str(value), 0))


def _macro_agreement_for_date(macro_date: pd.DataFrame) -> dict:
    """Economic vs empirical agreement without using future information.

    HIGH   : same leader.
    MEDIUM : reciprocal Top-3 membership.
    LOW    : both layers are available but point to materially different regimes.
    N.D.   : empirical layer not available yet.
    """
    if macro_date is None or macro_date.empty:
        return {
            "EconomicLeader": None, "EmpiricalLeader": None,
            "MacroAgreement": "N.D.", "AgreementScore": 0.0,
        }

    eco = macro_date.dropna(subset=["EconomicScore"]).sort_values("EconomicScore", ascending=False)
    emp = macro_date.dropna(subset=["EmpiricalScore"]).sort_values("EmpiricalScore", ascending=False)
    eco_leader = str(eco.iloc[0]["Regime"]) if len(eco) else None
    emp_leader = str(emp.iloc[0]["Regime"]) if len(emp) else None

    if eco_leader is None or emp_leader is None or len(emp) < 3:
        label = "N.D."
        score = 0.0
    elif eco_leader == emp_leader:
        label = "HIGH"
        score = 100.0
    else:
        eco_top3 = set(eco.head(3)["Regime"].astype(str))
        emp_top3 = set(emp.head(3)["Regime"].astype(str))
        if eco_leader in emp_top3 and emp_leader in eco_top3:
            label = "MEDIUM"
            score = 60.0
        else:
            label = "LOW"
            score = 25.0

    return {
        "EconomicLeader": eco_leader,
        "EmpiricalLeader": emp_leader,
        "MacroAgreement": label,
        "AgreementScore": score,
    }


def _decision_market_snapshot(market_date: pd.DataFrame) -> dict:
    """Extract one monthly Market Engine snapshot from the long regime table."""
    if market_date is None or market_date.empty:
        return {}
    row0 = market_date.iloc[0]
    active = str(row0.get("ActiveRegime"))
    leader = str(row0.get("MarketLeader"))
    emerging = row0.get("EmergingCandidate")
    emerging = None if pd.isna(emerging) else str(emerging)

    def regime_value(regime: str | None, col: str) -> float:
        if not regime:
            return np.nan
        z = market_date.loc[market_date["Regime"].astype(str) == str(regime), col]
        return float(z.iloc[0]) if len(z) and pd.notna(z.iloc[0]) else np.nan

    return {
        "MarketActive": active,
        "MarketActiveScore": float(row0.get("ActiveScore", np.nan)),
        "MarketState": str(row0.get("MarketState", "STABLE")),
        "MarketConfidence": str(row0.get("Confidence", "N.D.")),
        "MarketLeader": leader,
        "MarketLeaderScore": float(row0.get("LeaderScore", np.nan)),
        "MarketEmerging": emerging,
        "MarketEmergingScore": float(row0.get("EmergingScore", np.nan)),
        "CandidateCurrent": regime_value(emerging, "Current"),
        "CandidateAcceleration": regime_value(emerging, "A"),
    }


def _decision_macro_snapshot(macro_date: pd.DataFrame) -> dict:
    if macro_date is None or macro_date.empty:
        return {}
    z = macro_date.dropna(subset=["MacroScore"]).sort_values("MacroScore", ascending=False)
    if z.empty:
        return {}
    top = z.iloc[0]
    agree = _macro_agreement_for_date(z)
    return {
        "MacroNext": str(top["Regime"]),
        "MacroScore": float(top["MacroScore"]),
        "MacroRunnerUp": str(top.get("MacroRunnerUp", z.iloc[1]["Regime"] if len(z) > 1 else "")),
        "MacroSpread": float(top.get("MacroSpread", np.nan)),
        "MacroConfidence": str(top.get("MacroConfidence", "N.D.")),
        **agree,
    }


def _transition_score(snapshot: dict) -> float:
    """Transparent 0-100 diagnostic score; rules, not this score, trigger SWITCH."""
    conf_score = {"N.D.": 0.0, "LOW": 25.0, "MEDIUM": 50.0, "MEDIUM-HIGH": 75.0, "HIGH": 100.0}.get(snapshot.get("MacroConfidence"), 0.0)
    agr_score = float(snapshot.get("AgreementScore", 0.0))
    macro_next = snapshot.get("MacroNext")
    emerging = snapshot.get("MarketEmerging")
    leader = snapshot.get("MarketLeader")
    e_score = float(snapshot.get("MarketEmergingScore", np.nan))
    l_score = float(snapshot.get("MarketLeaderScore", np.nan))
    if macro_next == emerging and np.isfinite(e_score):
        market_confirm = float(np.clip(e_score, 0, 100))
    elif macro_next == leader and np.isfinite(l_score):
        market_confirm = float(np.clip(0.80 * l_score, 0, 100))
    else:
        market_confirm = 15.0
    return float(np.clip(0.35 * conf_score + 0.30 * agr_score + 0.35 * market_confirm, 0, 100))


def build_decision_engine(market_history: pd.DataFrame, macro_history: pd.DataFrame) -> pd.DataFrame:
    """Decision Engine V1, strictly chronological.

    The Market Engine remains the robust anchor.  Macro may recommend an early
    transition only when Economic/Empirical information and Market Emerging
    converge.  A regime is always present.

    Early SWITCH normal:
      - MacroNext == MarketEmerging
      - Emerging >= 60
      - Macro confidence >= MEDIUM-HIGH
      - Macro agreement >= MEDIUM
      - candidate Current >= 60
      - Market proximity: candidate Top-3 Current OR <=10 points behind Market Leader
      - same candidate for 2 consecutive monthly observations

    Early SWITCH quick (one observation):
      - same convergence
      - Macro confidence HIGH + Agreement HIGH
      - MacroScore >= 75, Emerging >= 70, candidate Current >= 60
      - stricter Market proximity: candidate Top-2 Current OR <=7 points behind Market Leader

    PREPARE requires weaker but meaningful convergence.  WATCH is divergence.
    An early-switched DecisionRegime is sticky: it is not abandoned on one
    noisy month; reversal to the validated MarketActive requires 2 consecutive
    months in which both Macro and Market no longer support the early regime.
    """
    if market_history is None or market_history.empty or macro_history is None or macro_history.empty:
        return pd.DataFrame()

    mh = market_history.copy()
    mh["Date"] = pd.to_datetime(mh["Date"])
    mh["Period"] = mh["Date"].dt.to_period("M")
    # one market long-table snapshot per period; use the last real market date
    market_periods = {p: g[g["Date"] == g["Date"].max()].copy() for p, g in mh.groupby("Period")}

    mac = macro_history.copy()
    mac["Date"] = pd.to_datetime(mac["Date"])
    mac["Period"] = mac["Date"].dt.to_period("M")
    macro_periods = {p: g[g["Date"] == g["Date"].max()].copy() for p, g in mac.groupby("Period")}

    periods = sorted(set(market_periods).intersection(macro_periods))
    rows: list[dict] = []
    decision_regime: str | None = None
    previous_market_active: str | None = None
    pending_candidate: str | None = None
    pending_streak = 0
    abort_streak = 0

    for period in periods:
        m = _decision_market_snapshot(market_periods[period])
        q = _decision_macro_snapshot(macro_periods[period])
        if not m or not q:
            continue

        market_active = m["MarketActive"]
        if decision_regime is None:
            decision_regime = market_active

        # A fully validated Market Engine switch is always accepted immediately.
        market_switched = previous_market_active is not None and market_active != previous_market_active
        if market_switched:
            decision_regime = market_active
            pending_candidate, pending_streak, abort_streak = None, 0, 0

        macro_next = q["MacroNext"]
        macro_conf_rank = _rank_label(q["MacroConfidence"], _CONF_RANK)
        agreement_rank = _rank_label(q["MacroAgreement"], _AGR_RANK)
        emerging_match = macro_next == m["MarketEmerging"]
        leader_match = macro_next == m["MarketLeader"]
        e_score = m["MarketEmergingScore"]

        # Market Proximity Filter (v2.5): Macro may anticipate only a regime that
        # is already reasonably close to the cross-sectional Current leadership.
        current_ranked = market_periods[period].dropna(subset=["Current"]).sort_values("Current", ascending=False).copy()
        current_ranked["_CurrentRank"] = np.arange(1, len(current_ranked) + 1)
        macro_row = current_ranked.loc[current_ranked["Regime"].astype(str) == str(macro_next)]
        if len(macro_row):
            cand_current = float(macro_row.iloc[0]["Current"])
            cand_rank = int(macro_row.iloc[0]["_CurrentRank"])
        else:
            cand_current = np.nan
            cand_rank = np.nan
        leader_current = float(current_ranked.iloc[0]["Current"]) if len(current_ranked) else np.nan
        cand_gap = float(leader_current - cand_current) if np.isfinite(leader_current) and np.isfinite(cand_current) else np.nan

        normal_proximity = (
            np.isfinite(cand_current) and cand_current >= 60.0
            and ((np.isfinite(cand_rank) and cand_rank <= 3) or (np.isfinite(cand_gap) and cand_gap <= 10.0))
        )
        quick_proximity = (
            np.isfinite(cand_current) and cand_current >= 60.0
            and ((np.isfinite(cand_rank) and cand_rank <= 2) or (np.isfinite(cand_gap) and cand_gap <= 7.0))
        )

        normal_setup = (
            macro_next != decision_regime
            and emerging_match
            and np.isfinite(e_score) and e_score >= 60.0
            and macro_conf_rank >= _CONF_RANK["MEDIUM-HIGH"]
            and agreement_rank >= _AGR_RANK["MEDIUM"]
            and normal_proximity
        )
        quick_setup = (
            normal_setup
            and q["MacroConfidence"] == "HIGH"
            and q["MacroAgreement"] == "HIGH"
            and q["MacroScore"] >= 75.0
            and e_score >= 70.0
            and quick_proximity
        )
        prepare_setup = (
            macro_next != decision_regime
            and macro_conf_rank >= _CONF_RANK["MEDIUM"]
            and agreement_rank >= _AGR_RANK["MEDIUM"]
            and (
                (emerging_match and np.isfinite(e_score) and e_score >= 55.0)
                or (leader_match and np.isfinite(m["MarketLeaderScore"]) and m["MarketLeaderScore"] >= 60.0)
            )
        )

        action = "HOLD"
        reason = ""
        switch_type = ""

        # If an earlier Macro switch is still ahead of Market, keep it sticky.
        early_ahead = decision_regime != market_active
        if early_ahead:
            still_supported = (
                macro_next == decision_regime
                or m["MarketEmerging"] == decision_regime
                or m["MarketLeader"] == decision_regime
            )
            if still_supported:
                abort_streak = 0
                action = "HOLD"
                reason = f"Mantieni early regime {decision_regime}; Market non ha ancora completato la conferma"
            else:
                abort_streak += 1
                if abort_streak >= 2:
                    decision_regime = market_active
                    abort_streak = 0
                    action = "SWITCH"
                    switch_type = "ABORT EARLY"
                    reason = f"Rientro sul Market Regime validato {market_active}: early signal non più confermato per 2 mesi"
                else:
                    action = "WATCH"
                    reason = f"Early regime {decision_regime} perde conferme; 1/2 mesi prima del rientro al Market Regime"
        else:
            abort_streak = 0
            if macro_next == decision_regime:
                pending_candidate, pending_streak = None, 0
                action = "HOLD"
                reason = f"Macro 2–3M allineato al regime operativo ({q['MacroConfidence']}, agreement {q['MacroAgreement']})"
            elif quick_setup:
                decision_regime = macro_next
                pending_candidate, pending_streak = None, 0
                action = "SWITCH"
                switch_type = "QUICK EARLY"
                reason = f"Convergenza eccezionale Macro + Market Emerging verso {macro_next}"
            elif normal_setup:
                if pending_candidate == macro_next:
                    pending_streak += 1
                else:
                    pending_candidate, pending_streak = macro_next, 1
                if pending_streak >= 2:
                    decision_regime = macro_next
                    action = "SWITCH"
                    switch_type = "EARLY 2M"
                    reason = f"Convergenza Macro + Market Emerging confermata per {pending_streak} mesi verso {macro_next}"
                    pending_candidate, pending_streak = None, 0
                else:
                    action = "PREPARE"
                    reason = f"Convergenza Macro + Market Emerging verso {macro_next}: conferma {pending_streak}/2"
            elif prepare_setup:
                if pending_candidate == macro_next:
                    pending_streak += 1
                else:
                    pending_candidate, pending_streak = macro_next, 1
                action = "PREPARE"
                if emerging_match and not normal_proximity and np.isfinite(cand_gap):
                    reason = (
                        f"Convergenza verso {macro_next}, ma Market proximity non ancora sufficiente "
                        f"(Current rank {int(cand_rank) if np.isfinite(cand_rank) else 'n.d.'}, gap leader {cand_gap:.1f})"
                    )
                else:
                    reason = f"Segnale di transizione verso {macro_next}, ma non ancora abbastanza forte per early switch"
            else:
                pending_candidate, pending_streak = None, 0
                action = "WATCH" if macro_next != decision_regime else "HOLD"
                if q["MacroConfidence"] == "LOW":
                    reason = f"Macro diverge verso {macro_next}, ma confidence LOW"
                elif q["MacroAgreement"] == "LOW":
                    reason = f"Macro diverge verso {macro_next}, ma Economic ed Empirical non concordano"
                else:
                    reason = f"Macro punta a {macro_next}, senza sufficiente conferma del Market Emerging"

        snapshot = {**m, **q}
        tscore = _transition_score(snapshot)
        dt = pd.Timestamp(market_periods[period]["Date"].max())  # actual last available basket date; avoid future month-end labels
        rows.append({
            "Date": dt,
            "Period": str(period),
            "DecisionRegime": decision_regime,
            "MarketActive": market_active,
            "MarketActiveScore": m["MarketActiveScore"],
            "MarketState": m["MarketState"],
            "MarketConfidence": m["MarketConfidence"],
            "MarketLeader": m["MarketLeader"],
            "MarketLeaderScore": m["MarketLeaderScore"],
            "MarketEmerging": m["MarketEmerging"],
            "MarketEmergingScore": m["MarketEmergingScore"],
            "MacroCandidateCurrent": cand_current,
            "MacroCandidateCurrentRank": cand_rank,
            "MacroCandidateGapToLeader": cand_gap,
            "NormalProximityOK": bool(normal_proximity),
            "QuickProximityOK": bool(quick_proximity),
            "MacroNext": macro_next,
            "MacroScore": q["MacroScore"],
            "MacroRunnerUp": q["MacroRunnerUp"],
            "MacroSpread": q["MacroSpread"],
            "MacroConfidence": q["MacroConfidence"],
            "EconomicLeader": q["EconomicLeader"],
            "EmpiricalLeader": q["EmpiricalLeader"],
            "MacroAgreement": q["MacroAgreement"],
            "TransitionScore": tscore,
            "Action": action,
            "SwitchType": switch_type,
            "PendingCandidate": pending_candidate,
            "PendingStreak": pending_streak,
            "AbortStreak": abort_streak,
            "Reason": reason,
        })
        previous_market_active = market_active

    return pd.DataFrame(rows)


def decision_validation(decision_history: pd.DataFrame, basket_prices: pd.DataFrame | None) -> pd.DataFrame:
    """Simple one-month forward validation of DecisionRegime vs MarketActive.

    It is diagnostic only and never feeds the decision rules.
    """
    if decision_history is None or decision_history.empty or basket_prices is None or basket_prices.empty:
        return pd.DataFrame()
    p = basket_prices.copy().sort_index()
    p.index = pd.to_datetime(p.index)
    cols = [c for c in REGIMES if c in p.columns]
    p = p[cols].resample("ME").last()
    fwd = p.shift(-1) / p - 1.0
    fwd.index = fwd.index.to_period("M")

    rows = []
    for _, r in decision_history.iterrows():
        period = pd.Period(r["Period"], freq="M")
        if period not in fwd.index:
            continue
        dr = str(r["DecisionRegime"])
        mr = str(r["MarketActive"])
        if dr not in fwd.columns or mr not in fwd.columns:
            continue
        rd = fwd.at[period, dr]
        rm = fwd.at[period, mr]
        if not (np.isfinite(rd) and np.isfinite(rm)):
            continue
        rows.append({
            "Period": str(period), "DecisionRegime": dr, "MarketActive": mr,
            "DecisionReturn1M": float(rd), "MarketReturn1M": float(rm),
            "DeltaDecisionVsMarket": float(rd - rm),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame()
    summary = pd.DataFrame([{
        "Observations": len(d),
        "DecisionAvg1M": 100 * d["DecisionReturn1M"].mean(),
        "MarketAvg1M": 100 * d["MarketReturn1M"].mean(),
        "AvgDeltaBp": 10000 * d["DeltaDecisionVsMarket"].mean(),
        "DecisionBeatMarketPct": 100 * (d["DeltaDecisionVsMarket"] > 0).mean(),
        "EarlyMonths": int((d["DecisionRegime"] != d["MarketActive"]).sum()),
    }])
    return summary


def decision_preview(market_latest: pd.DataFrame, macro_latest: pd.DataFrame, decision_history: pd.DataFrame | None = None) -> dict:
    """Latest Decision Engine snapshot. Prefer the chronological history when supplied."""
    if decision_history is not None and not decision_history.empty:
        r = decision_history.sort_values("Date").iloc[-1]
        return {
            "OperationalRegime": str(r["DecisionRegime"]),
            "MarketActive": str(r["MarketActive"]),
            "MacroNext": str(r["MacroNext"]),
            "MacroScore": float(r["MacroScore"]),
            "MacroConfidence": str(r["MacroConfidence"]),
            "MacroAgreement": str(r["MacroAgreement"]),
            "EconomicLeader": str(r["EconomicLeader"]) if pd.notna(r["EconomicLeader"]) else None,
            "EmpiricalLeader": str(r["EmpiricalLeader"]) if pd.notna(r["EmpiricalLeader"]) else None,
            "MarketEmerging": str(r["MarketEmerging"]) if pd.notna(r["MarketEmerging"]) else None,
            "MarketEmergingScore": float(r["MarketEmergingScore"]) if pd.notna(r["MarketEmergingScore"]) else np.nan,
            "MacroCandidateCurrent": float(r["MacroCandidateCurrent"]) if "MacroCandidateCurrent" in r.index and pd.notna(r["MacroCandidateCurrent"]) else np.nan,
            "MacroCandidateCurrentRank": int(r["MacroCandidateCurrentRank"]) if "MacroCandidateCurrentRank" in r.index and pd.notna(r["MacroCandidateCurrentRank"]) else None,
            "MacroCandidateGapToLeader": float(r["MacroCandidateGapToLeader"]) if "MacroCandidateGapToLeader" in r.index and pd.notna(r["MacroCandidateGapToLeader"]) else np.nan,
            "NormalProximityOK": bool(r["NormalProximityOK"]) if "NormalProximityOK" in r.index and pd.notna(r["NormalProximityOK"]) else False,
            "QuickProximityOK": bool(r["QuickProximityOK"]) if "QuickProximityOK" in r.index and pd.notna(r["QuickProximityOK"]) else False,
            "TransitionScore": float(r["TransitionScore"]),
            "Action": str(r["Action"]),
            "SwitchType": str(r["SwitchType"]) if pd.notna(r["SwitchType"]) else "",
            "TransitionText": str(r["Reason"]),
        }

    # Fallback for latest-only use.
    if market_latest.empty or macro_latest.empty:
        return {}
    m = _decision_market_snapshot(market_latest)
    q = _decision_macro_snapshot(macro_latest)
    if not m or not q:
        return {}
    snap = {**m, **q}
    macro_next = q["MacroNext"]
    active = m["MarketActive"]
    if macro_next == active:
        action = "HOLD"
        reason = "Macro 2–3M allineato al regime operativo"
    elif q["MacroConfidence"] == "LOW" or q["MacroAgreement"] == "LOW":
        action = "WATCH"
        reason = f"Macro diverge verso {macro_next}, ma il segnale non è ancora robusto"
    elif m["MarketEmerging"] == macro_next and m["MarketEmergingScore"] >= 55:
        action = "PREPARE"
        reason = f"Convergenza Macro + Market Emerging verso {macro_next}"
    else:
        action = "WATCH"
        reason = f"Macro punta a {macro_next}, senza conferma sufficiente del Market Emerging"
    return {
        "OperationalRegime": active,
        "MarketActive": active,
        "MacroNext": macro_next,
        "MacroScore": q["MacroScore"],
        "MacroConfidence": q["MacroConfidence"],
        "MacroAgreement": q["MacroAgreement"],
        "EconomicLeader": q["EconomicLeader"],
        "EmpiricalLeader": q["EmpiricalLeader"],
        "MarketEmerging": m["MarketEmerging"],
        "MarketEmergingScore": m["MarketEmergingScore"],
        "TransitionScore": _transition_score(snap),
        "Action": action,
        "SwitchType": "",
        "TransitionText": reason,
    }
