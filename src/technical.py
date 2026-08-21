from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def rma(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(alpha=1 / length, adjust=False).mean()


def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    diff = s.diff()
    gain = diff.clip(lower=0)
    loss = (-diff.clip(upper=0))
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss.ne(0), 100.0).where(avg_gain.ne(0), 0.0)


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h: pd.Series, l: pd.Series, c: pd.Series, length: int = 14) -> pd.Series:
    return rma(true_range(h, l, c), length)


def supertrend(
    h: pd.Series, l: pd.Series, c: pd.Series, factor: float = 3.0, length: int = 14
) -> tuple[pd.Series, pd.Series]:
    """Supertrend. Direction convention: -1 uptrend, +1 downtrend."""
    hl2 = (h + l) / 2.0
    a = atr(h, l, c, length)
    upper_basic = hl2 + factor * a
    lower_basic = hl2 - factor * a

    n = len(c)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    st_line = np.full(n, np.nan)

    ub = upper_basic.to_numpy(dtype=float)
    lb = lower_basic.to_numpy(dtype=float)
    cv = c.to_numpy(dtype=float)

    for i in range(n):
        if np.isnan(cv[i]) or np.isnan(ub[i]) or np.isnan(lb[i]):
            continue
        if i == 0 or np.isnan(upper[i - 1]):
            upper[i], lower[i] = ub[i], lb[i]
            direction[i] = -1.0
            st_line[i] = lower[i]
            continue

        prev_c = cv[i - 1]
        upper[i] = ub[i] if (ub[i] < upper[i - 1] or prev_c > upper[i - 1]) else upper[i - 1]
        lower[i] = lb[i] if (lb[i] > lower[i - 1] or prev_c < lower[i - 1]) else lower[i - 1]

        prev_dir = direction[i - 1]
        if prev_dir < 0:
            direction[i] = 1.0 if cv[i] < lower[i] else -1.0
        else:
            direction[i] = -1.0 if cv[i] > upper[i] else 1.0
        st_line[i] = lower[i] if direction[i] < 0 else upper[i]

    return (
        pd.Series(st_line, index=c.index, name="supertrend"),
        pd.Series(direction, index=c.index, name="st_dir"),
    )


def dmi(
    h: pd.Series, l: pd.Series, c: pd.Series, di_len: int = 14, adx_len: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr_s = rma(true_range(h, l, c), di_len)
    plus_di = 100 * rma(plus_dm, di_len) / tr_s.replace(0, np.nan)
    minus_di = 100 * rma(minus_dm, di_len) / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_v = rma(dx.fillna(0), adx_len)
    return plus_di, minus_di, adx_v


def psar(
    h: pd.Series, l: pd.Series, start: float = 0.02, inc: float = 0.02, mx: float = 0.2
) -> pd.Series:
    hi = h.to_numpy(dtype=float)
    lo = l.to_numpy(dtype=float)
    n = len(h)
    sar = np.full(n, np.nan)
    if n < 2:
        return pd.Series(sar, index=h.index, name="psar")

    long = True
    af = start
    ep = hi[0]
    sar[0] = lo[0]

    for i in range(1, n):
        prev = sar[i - 1]
        if np.isnan(prev) or np.isnan(hi[i]) or np.isnan(lo[i]):
            continue
        s = prev + af * (ep - prev)
        if long:
            if i >= 2:
                s = min(s, lo[i - 1], lo[i - 2])
            else:
                s = min(s, lo[i - 1])
            if lo[i] < s:
                long = False
                s = ep
                ep = lo[i]
                af = start
            elif hi[i] > ep:
                ep = hi[i]
                af = min(mx, af + inc)
        else:
            if i >= 2:
                s = max(s, hi[i - 1], hi[i - 2])
            else:
                s = max(s, hi[i - 1])
            if hi[i] > s:
                long = True
                s = ep
                ep = hi[i]
                af = start
            elif lo[i] < ep:
                ep = lo[i]
                af = min(mx, af + inc)
        sar[i] = s

    return pd.Series(sar, index=h.index, name="psar")


def donchian_mid(h: pd.Series, l: pd.Series, length: int = 14) -> pd.Series:
    return (h.rolling(length).max() + l.rolling(length).min()) / 2.0


def trend_signal(df: pd.DataFrame) -> pd.DataFrame:
    """CDG-style weekly trend composite, reconstructed as a standalone implementation.

    Preset mirrors the original Market Map philosophy:
    Supertrend 14/3, EMA 13/40/100, ADX 14 threshold 18, Donchian 14.
    Score is normalized to [-1, +1].
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    out = pd.DataFrame(index=df.index)

    _, st_dir = supertrend(h, l, c, 3.0, 14)
    e13, e40, e100 = ema(c, 13), ema(c, 40), ema(c, 100)
    sar_v = psar(h, l, 0.02, 0.02, 0.2)
    dip, dim, adx_v = dmi(h, l, c, 14, 14)
    dc = donchian_mid(h, l, 14)

    out["sig_st"] = np.where(st_dir < 0, 1.0, -1.0)
    out["sig_ema"] = np.where(e13 > e40, 1.0, -1.0)
    out["sig_trd"] = np.where(c > e100, 1.0, -1.0)
    out["sig_sar"] = np.where(c > sar_v, 1.0, -1.0)
    directional = np.where(dip >= dim, 1.0, -1.0)
    out["sig_adx"] = np.where(adx_v >= 18.0, directional, 0.0)
    out["sig_dc"] = np.where(c > dc, 1.0, -1.0)
    out["adx"] = adx_v

    weights = {
        "sig_st": 1.0,
        "sig_ema": 1.0,
        "sig_sar": 0.7,
        "sig_adx": 0.7,
        "sig_dc": 0.5,
        "sig_trd": 0.5,
    }
    total_w = sum(weights.values())
    s = sum(out[k] * w for k, w in weights.items()) / total_w
    out["score"] = s.clip(-1, 1)
    out["state"] = np.select(
        [out["score"] >= 0.30, out["score"] <= -0.30, out["score"] >= 0],
        [2, -2, 1],
        default=-1,
    )
    return out


def oscillator(df: pd.DataFrame) -> pd.Series:
    rsi_v = rsi(df["Close"], 14)
    hh = rsi_v.rolling(30).max()
    ll = rsi_v.rolling(30).min()
    denom = (hh - ll).replace(0, np.nan)
    stoch_v = 100 * (rsi_v - ll) / denom
    return ema(stoch_v.fillna(50), 7).rename("osc")


def sign_series(state: pd.Series) -> pd.Series:
    s = np.sign(state.astype(float)).replace(0, np.nan).ffill().fillna(1.0)
    return s


def persistence(state: pd.Series, window: int = 26, flip_cap: int = 4) -> float:
    sgn = sign_series(state).iloc[-window:]
    if len(sgn) == 0:
        return np.nan
    cur = sgn.iloc[-1]
    frac_same = float((sgn == cur).mean())
    flips = int((sgn.diff().fillna(0) != 0).sum())
    flip_score = max(0.0, 1.0 - flips / flip_cap)
    return 100.0 * (0.5 * frac_same + 0.5 * flip_score)


def strength(score_last: float, osc_last: float) -> float:
    ts_scaled = 50.0 * (float(score_last) + 1.0)
    return float(np.clip(0.60 * ts_scaled + 0.40 * float(osc_last), 0, 100))


def weekly_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    )
    return out.dropna()


def technical_history(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end history of Strength and Persistence for one basket OHLC series."""
    w = weekly_ohlc(df)
    ts = trend_signal(w)
    osc = oscillator(w)
    rows = []
    # Use the actual last weekly observation available in each month as the signal date.
    # This avoids labelling a partial month (e.g. data through 14/08) as 31/08.
    month_last_dates = [g.index.max() for _, g in w.groupby(w.index.to_period("M"))]
    for dt in month_last_dates:
        hist = w.loc[:dt]
        if len(hist) < 35:
            continue
        t = trend_signal(hist)
        o = oscillator(hist)
        score_last = float(t["score"].iloc[-1])
        osc_last = float(o.iloc[-1])
        rows.append(
            {
                "Date": dt,
                "F": strength(score_last, osc_last),
                "Q": persistence(t["state"], 26, 4),
                "TS": score_last,
                "Osc": osc_last,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["F", "Q", "TS", "Osc"])
    return pd.DataFrame(rows).set_index("Date")
