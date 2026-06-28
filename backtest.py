"""
TradeOdds v2 — Walk-Forward Backtest  (v4 — + SEC EDGAR point-in-time fundamentals)
====================================================================================
Validates the price-based + macro-regime probability engine against actual outcomes.
No look-ahead bias: all model inputs use only data available on the test date.

Improvements in this version:
  1. Regime-conditional base rate  — win rate computed only from historical windows
     whose VIX/SPY regime matched the current regime, not the unconditional average.
  2. Cross-sectional relative momentum vs sector ETF  — compares stock momentum
     to its sector ETF (XLK, XLV etc.) to produce a relative-strength score.
     Stocks outperforming their sector are meaningfully different from underperformers.
  3. Multi-horizon base rate  — averages win rates at 30/60/90/180d horizons
     (weighted toward target), reducing noise from any single lookback.
  4. Momentum/mean-reversion regime switching  — at VIX>30, RSI and momentum
     signals flip sign (oversold = bullish, overbought = bearish). At VIX<18,
     momentum persists (trend-following). Blended between 18–30.

Output:
  backtest_results.csv   — every prediction row
  backtest_report.txt    — calibration, Brier score, hit rate by bucket, new modules

Usage:
  python backtest.py
  python backtest.py --tickers AAPL MSFT NVDA --horizon 90 --direction long
"""

import argparse
import math
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from edgar import (get_fundamentals_as_of, prefetch_tickers,
                   has_earnings_in_window, get_insider_score)

warnings.filterwarnings("ignore")

# ── Universe ───────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    # Tech / Growth
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "AMD", "CRM",
    # Consumer / Retail
    "AMZN", "TSLA", "HD", "COST", "NKE",
    # Financials
    "JPM", "BAC", "GS",
    # Healthcare
    "UNH", "LLY", "JNJ",
    # Energy / Industrial
    "XOM", "CAT", "GE",
    # Defensive
    "PG", "KO", "MRK",
    # Small / mid volatile
    "SMCI", "ROKU", "SNAP",
]

# Sector ETF mapping — used for cross-sectional relative momentum (improvement 2)
TICKER_SECTOR_ETF = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"XLK","GOOGL":"XLC","META":"XLC",
    "AVGO":"XLK","AMD":"XLK","CRM":"XLK","AMZN":"XLY","TSLA":"XLY",
    "HD":"XLY","COST":"XLP","NKE":"XLY","JPM":"XLF","BAC":"XLF","GS":"XLF",
    "UNH":"XLV","LLY":"XLV","JNJ":"XLV","XOM":"XLE","CAT":"XLI","GE":"XLI",
    "PG":"XLP","KO":"XLP","MRK":"XLV","SMCI":"XLK","ROKU":"XLC","SNAP":"XLC",
}
SECTOR_ETFS = list(set(TICKER_SECTOR_ETF.values()))  # XLK, XLC, XLY, XLP, XLF, XLV, XLE, XLI

# Test dates: every other Monday, Jan 2022 – Dec 2023
def test_dates(start="2022-01-03", end="2023-12-31", step_weeks=2):
    d = datetime.strptime(start, "%Y-%m-%d")
    end_d = datetime.strptime(end, "%Y-%m-%d")
    out = []
    while d <= end_d:
        out.append(d)
        d += timedelta(weeks=step_weeks)
    return out


# ── Price data cache ───────────────────────────────────────────────────────────
_price_cache: dict = {}

def get_full_history(ticker: str) -> pd.DataFrame:
    """Download full history once per ticker, cache in memory."""
    if ticker in _price_cache:
        return _price_cache[ticker]
    try:
        raw = yf.download(ticker, start="2019-01-01", end=datetime.today().strftime("%Y-%m-%d"),
                          progress=False, auto_adjust=True)
        if raw.empty:
            _price_cache[ticker] = pd.DataFrame()
            return pd.DataFrame()
        df = raw.reset_index()
        # flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if c[1] == '' or c[1] == ticker else c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                                 "low": "low", "close": "close", "volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        _price_cache[ticker] = df
        return df
    except Exception as e:
        print(f"  [!] Could not download {ticker}: {e}")
        _price_cache[ticker] = pd.DataFrame()
        return pd.DataFrame()


def slice_at_date(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Return only rows up to and including as_of date."""
    return df[df["date"] <= pd.Timestamp(as_of)].copy().reset_index(drop=True)


# ── Model modules (price-based, no fundamentals) ───────────────────────────────

def _returns(closes):
    return np.diff(np.log(np.array(closes, dtype=float) + 1e-9))


def mod_technical(df: pd.DataFrame, cur: float, vix: float = 20.0) -> float:
    """
    Returns score in (-1, +1).

    Improvement 4 — momentum/mean-reversion switching:
    - VIX < 18: pure momentum regime — trend signals trusted at full weight
    - VIX > 30: mean-reversion regime — RSI flips (oversold=bullish), momentum inverted
    - 18–30:    blend between the two
    """
    closes  = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    n = len(closes)
    if n < 20:
        return 0.0

    def mom(days):
        idx = max(0, n - days)
        base = float(closes[idx])
        return (cur - base) / base if base > 0 else 0.0

    d  = np.diff(closes[-30:] if n >= 30 else closes)
    g  = np.where(d > 0, d, 0.0)
    l  = np.where(d < 0, -d, 0.0)
    ag = np.mean(g[-14:]) if len(g) >= 14 else 0
    al = np.mean(l[-14:]) if len(l) >= 14 else 1e-9
    rsi = 100 - (100 / (1 + ag / (al + 1e-9)))

    ma50  = float(np.mean(closes[-50:]))  if n >= 50  else cur
    ma200 = float(np.mean(closes[-200:])) if n >= 200 else cur

    vol20     = float(np.mean(volumes[-20:])) if n >= 20 else 1
    vol60     = float(np.mean(volumes[-60:])) if n >= 60 else 1
    vol_ratio = vol20 / (vol60 + 1e-9)

    mom3m = mom(63)

    # Mean-reversion blend factor: 0 = pure momentum, 1 = pure mean-reversion
    mr_blend = float(np.clip((vix - 18) / 12, 0.0, 1.0))  # 0 at VIX=18, 1 at VIX=30+

    # Momentum signal (standard: RSI>50 and above MAs = bullish)
    rsi_mom = float(np.clip((rsi - 50) / 30, -1, 1))
    mom_sig = float(np.clip(mom3m * 4, -1, 1))

    # Mean-reversion signal (inverted: oversold RSI = bullish, strong momentum = bearish)
    rsi_mr  = -rsi_mom   # oversold (low RSI) = bullish in mean-reversion regime
    mom_mr  = -mom_sig   # fallen hard → bounce expected

    rsi_final = rsi_mom * (1 - mr_blend) + rsi_mr * mr_blend
    mom_final = mom_sig * (1 - mr_blend) + mom_mr * mr_blend

    score = (
        0.25 * rsi_final +
        0.20 * (1.0 if cur > ma50  else -1.0) +
        0.20 * (1.0 if cur > ma200 else -1.0) +
        0.20 * mom_final +
        0.15 * float(np.clip((vol_ratio - 1) * 2, -1, 1))
    )
    return float(np.clip(score, -1, 1))


def mod_market_factor(df: pd.DataFrame, spy_df: pd.DataFrame) -> tuple:
    """Returns (alpha_ann, score)."""
    merged = pd.merge(
        df[["date", "close"]].rename(columns={"close": "stock"}),
        spy_df[["date", "close"]].rename(columns={"close": "spy"}),
        on="date"
    ).dropna()
    if len(merged) < 60:
        return 0.0, 0.0
    sr = merged["stock"].pct_change().dropna().values
    mr = merged["spy"].pct_change().dropna().values
    n  = min(len(sr), len(mr))
    sr, mr = sr[-n:], mr[-n:]
    X = np.column_stack([np.ones(n), mr])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, sr, rcond=None)
        alpha_ann = float(coeffs[0]) * 252
    except Exception:
        alpha_ann = 0.0
    score = float(np.clip(alpha_ann / 0.20, -1, 1))
    return alpha_ann, score


def mod_vol_surface(df: pd.DataFrame) -> float:
    """Returns implied vol proxy."""
    closes = df["close"].values.astype(float)
    rets   = _returns(closes)
    v20    = float(np.std(rets[-20:])  * math.sqrt(252)) if len(rets) >= 20  else 0.25
    v60    = float(np.std(rets[-60:])  * math.sqrt(252)) if len(rets) >= 60  else 0.25
    v252   = float(np.std(rets[-252:]) * math.sqrt(252)) if len(rets) >= 252 else 0.25
    return float(np.clip(0.5 * v60 + 0.3 * v252 + 0.2 * v20, 0.08, 2.5))


def _build_regime_series(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.Series:
    """
    Precompute regime label for every date in history.
    Used by regime-conditional base rate to match past windows to current regime.
    """
    spy = spy_df[["date","close"]].set_index("date").sort_index()
    vix = vix_df[["date","close"]].rename(columns={"close":"vix"}).set_index("date").sort_index()
    merged = spy.join(vix, how="inner")
    merged["ma200"] = merged["close"].rolling(200, min_periods=50).mean()

    def _label(row):
        if pd.isna(row["vix"]) or pd.isna(row["ma200"]):
            return "neutral"
        bull = row["close"] > row["ma200"]
        v    = row["vix"]
        if bull and v < 18:  return "bull_low_vol"
        if bull and v < 28:  return "bull_normal"
        if bull:             return "bull_high_vol"
        if v < 28:           return "bear_normal"
        return "bear_high_vol"

    return merged.apply(_label, axis=1)


def mod_base_rate_regime(df: pd.DataFrame, direction: str, horizon: int,
                         regime_series: pd.Series, current_regime: str) -> tuple:
    """
    Improvement 1 — Regime-conditional base rate.

    Counts only historical windows whose starting-date regime matched
    current_regime. Returns (regime_base_rate, unconditional_base_rate, n_regime_windows).

    Falls back to unconditional rate when fewer than 30 regime-matched windows exist.
    """
    closes = df["close"].values.astype(float)
    dates  = df["date"].values

    if len(closes) < horizon + 20:
        return 0.50, 0.50, 0

    wins_cond, total_cond = 0, 0
    wins_all,  total_all  = 0, 0

    for i in range(len(closes) - horizon):
        ret  = (closes[i + horizon] - closes[i]) / closes[i]
        win  = int(ret > 0) if direction == "long" else int(ret < 0)
        wins_all  += win
        total_all += 1
        # check if regime at date i matches current regime
        dt = pd.Timestamp(dates[i])
        if dt in regime_series.index:
            hist_regime = regime_series.loc[dt]
            if hist_regime == current_regime:
                wins_cond  += win
                total_cond += 1

    unconditional = wins_all / total_all if total_all > 0 else 0.50

    if total_cond >= 30:
        conditional = wins_cond / total_cond
        # Blend: 70% regime-conditional, 30% unconditional — avoids sparse-data noise
        blended = 0.70 * conditional + 0.30 * unconditional
    else:
        blended = unconditional   # fall back when not enough regime history

    return blended, unconditional, total_cond


def mod_multi_horizon_base_rate(df: pd.DataFrame, direction: str,
                                horizon: int, regime_series: pd.Series,
                                current_regime: str) -> float:
    """
    Improvement 3 — Multi-horizon base rate.
    Average regime-conditional base rates at several horizons, weighted toward
    the target. Reduces noise from any single lookback window.
    """
    horizons = [21, 42, 63, 126, 252]   # 1m, 2m, 3m, 6m, 12m trading days
    # Weight: closest to target horizon gets most weight
    dists  = [abs(h - horizon) + 1 for h in horizons]
    inv    = [1.0 / d for d in dists]
    total  = sum(inv)
    weights = [v / total for v in inv]

    rates = []
    for h, w in zip(horizons, weights):
        r, _, _ = mod_base_rate_regime(df, direction, h, regime_series, current_regime)
        rates.append(r * w)

    return float(sum(rates))


def mod_relative_momentum(df: pd.DataFrame, sector_df: pd.DataFrame, cur: float) -> float:
    """
    Improvement 2 — Cross-sectional relative momentum vs sector ETF.

    Computes the stock's 1m/3m/12m momentum relative to its sector ETF.
    A stock outperforming XLK by 15% over 12m is a qualitatively different signal
    than one underperforming it, even if both have positive absolute momentum.

    Returns a score in (-1, +1).
    """
    if sector_df.empty or len(df) < 21:
        return 0.0

    merged = pd.merge(
        df[["date","close"]].rename(columns={"close":"stock"}),
        sector_df[["date","close"]].rename(columns={"close":"sector"}),
        on="date"
    ).dropna()

    if len(merged) < 21:
        return 0.0

    n = len(merged)

    def rel_mom(days):
        idx  = max(0, n - days)
        s0   = float(merged["stock"].iloc[idx])
        e0   = float(merged["sector"].iloc[idx])
        s1   = float(merged["stock"].iloc[-1])
        e1   = float(merged["sector"].iloc[-1])
        if s0 <= 0 or e0 <= 0:
            return 0.0
        stock_ret  = (s1 - s0) / s0
        sector_ret = (e1 - e0) / e0
        return stock_ret - sector_ret   # positive = outperforming sector

    rm1m  = rel_mom(21)  if n >= 21  else 0.0
    rm3m  = rel_mom(63)  if n >= 63  else rm1m
    rm12m = rel_mom(252) if n >= 252 else rm3m

    # Composite relative momentum score — 12m has most weight (trend), 1m least (noise)
    composite = 0.20 * rm1m + 0.30 * rm3m + 0.50 * rm12m
    return float(np.clip(composite * 5, -1, 1))   # scale: ±20% outperformance → ±1


def mod_analogues(df: pd.DataFrame, direction: str, horizon: int, cur_vol: float) -> float:
    closes = df["close"].values.astype(float)
    rets   = _returns(closes)
    if len(closes) < 90 + horizon:
        return 0.50
    results = []
    for i in range(30, len(closes) - horizon):
        hv = float(np.std(rets[max(0, i-30):i]) * math.sqrt(252))
        if abs(hv - cur_vol) / (cur_vol + 1e-9) < 0.30:
            fwd = (closes[i + horizon] - closes[i]) / closes[i]
            results.append(fwd if direction == "long" else -fwd)
    if not results:
        return 0.50
    return float(np.mean([r > 0 for r in results]))


def mod_fundamentals(fund: dict, direction: str) -> float:
    """
    Point-in-time fundamental quality score from SEC EDGAR XBRL data.
    Returns score in (-1, +1) — positive = quality company, bullish for longs.

    Four sub-scores:
      Profitability — gross margin, net margin, ROE
      Growth        — revenue growth YoY
      Safety        — debt-to-equity, asset turnover
      Innovation    — R&D intensity (future optionality)
    """
    if not fund.get("has_data"):
        return 0.0

    def safe(k, default=0.0):
        v = fund.get(k)
        return float(v) if v is not None else default

    gm  = safe("gross_margin")
    nm  = safe("net_margin")
    roe = safe("roe")
    rev_growth = safe("revenue_growth_yoy")
    de  = safe("debt_to_equity", 1.0)
    at  = safe("asset_turnover")
    rd  = safe("rd_intensity")

    # Cap extreme values that indicate data artifacts
    gm  = float(np.clip(gm,  -0.5, 1.0))
    nm  = float(np.clip(nm,  -0.5, 0.6))
    roe = float(np.clip(roe, -1.0, 3.0))
    de  = float(np.clip(de,   0.0, 10.0))

    # Profitability: high margins and ROE = quality (normalised to -1/+1)
    profitability = float(np.clip(
        0.40 * (gm  - 0.30) / 0.30 +    # gross margin: 30%=neutral, 60%=max
        0.35 * (nm  - 0.08) / 0.12 +    # net margin:   8%=neutral, 20%=max
        0.25 * (roe - 0.12) / 0.18,     # ROE:          12%=neutral, 30%=max
    -1, 1))

    # Growth: positive YoY revenue growth is a tailwind
    growth = float(np.clip(rev_growth / 0.20, -1, 1))   # 20% growth = max score

    # Safety: low debt, high asset turnover
    safety = float(np.clip(
        0.60 * (1 - de / 3.0) +         # de=0 → 1.0, de=3 → 0.0, de>3 → negative
        0.40 * (at  - 0.3) / 0.7,       # asset turnover: 0.3=neutral, 1.0=max
    -1, 1))

    # Innovation: R&D intensity 5-25% is the sweet spot for future optionality
    if rd > 0.40:      rd_s = 0.2    # excessive R&D (burning cash)
    elif rd > 0.05:    rd_s = min(1.0, rd / 0.15)
    else:              rd_s = rd / 0.05 * 0.4
    innovation = float(np.clip(rd_s * 2 - 1, -1, 1))

    composite = float(
        0.40 * profitability +
        0.25 * growth        +
        0.20 * safety        +
        0.15 * innovation
    )

    # Direction flip: high-quality fundamentals support longs, hurt shorts
    d = 1 if direction == "long" else -1
    return float(np.clip(composite * d, -1, 1))


def mod_volume(df: pd.DataFrame, cur: float) -> float:
    """
    Volume confirmation score from OHLCV data.  No new data — purely derived
    from the price/volume history we already download.

    Three sub-signals:
      OBV trend      — On-Balance Volume slope over last 20 days.
                       Rising OBV on rising price = institutional accumulation.
      Volume ratio   — Avg volume last 5d vs avg volume last 20d.
                       Expansion (>1.2x) with momentum = conviction signal.
      Up/down volume — On days price rose, was volume above average?
                       Measures whether buyers are more active than sellers.

    Returns score in (-1, +1): positive = volume confirming bullish move.
    """
    if len(df) < 25:
        return 0.0

    closes  = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    # Guard against zero volume (e.g. ETF data gaps)
    volumes = np.where(volumes < 1, 1.0, volumes)

    # OBV: cumulative sum of volume signed by daily price direction
    daily_ret = np.diff(closes)
    signs     = np.sign(daily_ret)
    obv       = np.cumsum(signs * volumes[1:])
    # OBV trend: linear slope over last 20 days, normalised by mean OBV
    obv_recent = obv[-20:]
    obv_mean   = np.mean(np.abs(obv_recent)) + 1
    x = np.arange(len(obv_recent))
    slope = float(np.polyfit(x, obv_recent, 1)[0]) / obv_mean
    obv_score = float(np.clip(slope * 50, -1, 1))

    # Volume ratio: recent vs baseline (expansion = conviction)
    vol_5  = float(np.mean(volumes[-5:]))
    vol_20 = float(np.mean(volumes[-20:]))
    vol_ratio = vol_5 / (vol_20 + 1e-9)
    # >1.5x expansion = strong, <0.5x = drying up
    vol_ratio_score = float(np.clip((vol_ratio - 1.0) / 0.5, -1, 1))

    # Up/down volume: ratio of avg volume on up-days vs down-days (last 20d)
    rets_20   = daily_ret[-20:]
    vols_20   = volumes[-20:][-len(rets_20):]
    up_mask   = rets_20 > 0
    down_mask = rets_20 < 0
    up_vol    = float(np.mean(vols_20[up_mask]))   if up_mask.any()   else vol_20
    down_vol  = float(np.mean(vols_20[down_mask])) if down_mask.any() else vol_20
    ud_ratio  = up_vol / (down_vol + 1e-9)
    ud_score  = float(np.clip((ud_ratio - 1.0) / 0.5, -1, 1))

    composite = 0.40 * obv_score + 0.35 * vol_ratio_score + 0.25 * ud_score
    return float(np.clip(composite, -1, 1))


def _sample_t(df_deg, rows, cols, rng):
    z = rng.standard_normal((rows, cols))
    v = rng.chisquare(df_deg, (rows, cols))
    return z / np.sqrt(v / df_deg)


def mod_monte_carlo(cur: float, direction: str, horizon: int,
                    sigma: float, alpha_ann: float) -> float:
    rf_d    = 0.045 / 252
    drift_d = rf_d + alpha_ann / 252
    sig_d   = sigma / math.sqrt(252)
    t_df    = 5.0
    t_scale = math.sqrt((t_df - 2) / t_df)
    rng     = np.random.default_rng(seed=42)
    shocks  = _sample_t(t_df, 10_000, horizon, rng) * sig_d / t_scale
    log_r   = (drift_d - 0.5 * sig_d ** 2) + shocks
    terminal = cur * np.exp(log_r.sum(axis=1))
    wins = terminal > cur if direction == "long" else terminal < cur
    return float(wins.mean())


# Regime-conditional factor weights — now includes rel_momentum slot
REGIME_WEIGHTS = {
    #                  tech   mkt    base   ana    relm   fund   vol    insider
    "bull_low_vol":  {"technical":0.10,"market":0.08,"base_rate":0.22,"analogues":0.18,"rel_mom":0.15,"fundamentals":0.12,"volume":0.10,"insider":0.05},
    "bull_normal":   {"technical":0.09,"market":0.08,"base_rate":0.23,"analogues":0.19,"rel_mom":0.14,"fundamentals":0.12,"volume":0.10,"insider":0.05},
    "bull_high_vol": {"technical":0.06,"market":0.08,"base_rate":0.25,"analogues":0.20,"rel_mom":0.14,"fundamentals":0.12,"volume":0.10,"insider":0.05},
    "bear_normal":   {"technical":0.07,"market":0.11,"base_rate":0.23,"analogues":0.21,"rel_mom":0.12,"fundamentals":0.12,"volume":0.09,"insider":0.05},
    "bear_high_vol": {"technical":0.04,"market":0.11,"base_rate":0.25,"analogues":0.23,"rel_mom":0.11,"fundamentals":0.12,"volume":0.09,"insider":0.05},
    "neutral":       {"technical":0.08,"market":0.09,"base_rate":0.23,"analogues":0.20,"rel_mom":0.13,"fundamentals":0.12,"volume":0.10,"insider":0.05},
}

REGIME_PRIOR_SHIFT = {
    "bull_low_vol":   0.03,
    "bull_normal":    0.02,
    "bull_high_vol":  0.00,
    "bear_normal":   -0.04,
    "bear_high_vol": -0.07,
    "neutral":        0.00,
}

SHRINKAGE = 0.68


def classify_regime(spy_slice: pd.DataFrame, vix_slice: pd.DataFrame) -> tuple:
    """Returns (regime_str, vix_float)."""
    if spy_slice.empty or vix_slice.empty:
        return "neutral", 20.0
    spy_closes = spy_slice["close"].values.astype(float)
    vix_closes = vix_slice["close"].values.astype(float)
    vix     = float(vix_closes[-1])
    spy_cur = float(spy_closes[-1])
    ma200   = float(np.mean(spy_closes[-200:])) if len(spy_closes) >= 200 else spy_cur
    bull    = spy_cur > ma200
    if bull and vix < 18:  return "bull_low_vol",  vix
    if bull and vix < 28:  return "bull_normal",    vix
    if bull:               return "bull_high_vol",  vix
    if vix < 28:           return "bear_normal",    vix
    return "bear_high_vol", vix


def ensemble(tech_score, mkt_score, p_mc, p_base, p_ana, rel_mom_score,
             fund_score, vol_score, insider_score,
             direction: str, regime: str = "neutral") -> float:
    """
    Regime-weighted ensemble — v5: +volume confirmation +insider transactions.

    s2p multiplier: 0.18 — less aggressive probability conversion.
    Shrinkage: 0.68 — pulls final probability toward 0.50.
    """
    d = 1 if direction == "long" else -1

    def s2p(s):
        return float(np.clip(0.50 + float(s) * d * 0.18, 0.22, 0.78))

    p_tech     = s2p(tech_score)
    p_mkt      = s2p(mkt_score)
    p_rel_mom  = s2p(rel_mom_score)
    p_fund     = s2p(fund_score)
    p_vol      = s2p(vol_score)
    p_insider  = s2p(insider_score)
    p_base_dir = p_base if direction == "long" else (1 - p_base)
    p_ana_dir  = p_ana

    w = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["neutral"])
    factor_blend = (
        w["technical"]    * p_tech     +
        w["market"]       * p_mkt      +
        w["base_rate"]    * p_base_dir +
        w["analogues"]    * p_ana_dir  +
        w["rel_mom"]      * p_rel_mom  +
        w["fundamentals"] * p_fund     +
        w["volume"]       * p_vol      +
        w["insider"]      * p_insider
    )

    raw = float(np.clip(0.40 * p_mc + 0.60 * factor_blend, 0.05, 0.95))

    prior_shift = REGIME_PRIOR_SHIFT.get(regime, 0.0) * d
    raw = float(np.clip(raw + prior_shift, 0.05, 0.95))

    final = 0.50 + (raw - 0.50) * SHRINKAGE
    return float(np.clip(final, 0.10, 0.90))


# ── Single prediction ──────────────────────────────────────────────────────────

def predict(ticker: str, as_of: datetime, direction: str, horizon: int,
            full_df: pd.DataFrame, spy_df: pd.DataFrame, vix_df: pd.DataFrame,
            sector_df: pd.DataFrame, regime_series: pd.Series,
            edgar_cache: dict | None = None) -> dict | None:
    df = slice_at_date(full_df, as_of)
    if len(df) < 100:
        return None

    cur          = float(df["close"].iloc[-1])
    spy_slice    = slice_at_date(spy_df, as_of)
    vix_slice    = slice_at_date(vix_df, as_of)
    sector_slice = slice_at_date(sector_df, as_of) if not sector_df.empty else pd.DataFrame()

    regime, vix          = classify_regime(spy_slice, vix_slice)
    tech_score           = mod_technical(df, cur, vix)
    alpha_ann, mkt_score = mod_market_factor(df, spy_slice)
    implied_vol          = mod_vol_surface(df)
    p_base               = mod_multi_horizon_base_rate(
                               df, direction, horizon, regime_series, regime)
    p_ana                = mod_analogues(df, direction, horizon, implied_vol)
    rel_mom_score        = mod_relative_momentum(df, sector_slice, cur)
    p_mc                 = mod_monte_carlo(cur, direction, horizon, implied_vol, alpha_ann)

    # Point-in-time fundamentals from SEC EDGAR
    fund = get_fundamentals_as_of(ticker, as_of)
    fund_score = mod_fundamentals(fund if fund else {}, direction)

    # Volume confirmation (OBV + expansion + up/down volume ratio)
    vol_score = mod_volume(df, cur)

    # Insider transaction score (net buying/selling last 90 days via Form 4)
    insider_score = get_insider_score(ticker, as_of, lookback_days=90)

    p_win = ensemble(tech_score, mkt_score, p_mc, p_base, p_ana,
                     rel_mom_score, fund_score, vol_score, insider_score,
                     direction, regime)

    BULL_REGIMES = {"bull_low_vol", "bull_normal", "bull_high_vol"}
    BEAR_REGIMES = {"bear_normal", "bear_high_vol"}
    regime_match = (
        (direction == "long"  and regime in BULL_REGIMES) or
        (direction == "short" and regime in BEAR_REGIMES)
    )

    # Earnings flag: does this horizon contain a known earnings event?
    exit_date = as_of + timedelta(days=horizon)
    earnings_in_window = has_earnings_in_window(ticker, as_of, exit_date)

    return {
        "ticker":              ticker,
        "as_of":               as_of.strftime("%Y-%m-%d"),
        "direction":           direction,
        "horizon":             horizon,
        "regime":              regime,
        "regime_match":        regime_match,
        "earnings_in_window":  earnings_in_window,
        "vix":                 round(vix, 1),
        "p_win":               round(p_win, 4),
        "p_base":              round(p_base, 4),
        "rel_mom_score":       round(rel_mom_score, 4),
        "tech_score":          round(tech_score, 4),
        "fund_score":          round(fund_score, 4),
        "vol_score":           round(vol_score, 4),
        "insider_score":       round(insider_score, 4),
        "entry_px":            round(cur, 2),
        "exit_date":           exit_date.strftime("%Y-%m-%d"),
    }


def resolve(row: dict, full_df: pd.DataFrame) -> dict:
    """Look up actual outcome using post-as_of price data."""
    exit_dt = datetime.strptime(row["exit_date"], "%Y-%m-%d")
    after   = full_df[full_df["date"] > pd.Timestamp(row["as_of"])].reset_index(drop=True)
    if after.empty:
        return {**row, "exit_px": None, "actual_ret": None, "outcome": None}

    # find closest trading day on or after exit_date
    after_exit = after[after["date"] >= pd.Timestamp(exit_dt)]
    if after_exit.empty:
        after_exit = after.tail(1)
    exit_px  = float(after_exit["close"].iloc[0])
    entry_px = row["entry_px"]
    ret      = (exit_px - entry_px) / entry_px
    outcome  = int(ret > 0) if row["direction"] == "long" else int(ret < 0)

    return {**row, "exit_px": round(exit_px, 2),
            "actual_ret": round(ret, 4), "outcome": outcome}


# ── Calibration & stats ────────────────────────────────────────────────────────

def calibration_report(df: pd.DataFrame) -> str:
    resolved = df.dropna(subset=["outcome"]).copy()
    n_total  = len(resolved)
    if n_total == 0:
        return "No resolved predictions."

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"TradeOdds v2 — Backtest Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"{'='*60}")
    lines.append(f"Total predictions:  {n_total}")
    lines.append(f"Tickers:            {resolved['ticker'].nunique()}")
    lines.append(f"Date range:         {resolved['as_of'].min()} → {resolved['as_of'].max()}")
    if "regime" in resolved.columns:
        regime_counts = resolved["regime"].value_counts().to_dict()
        lines.append(f"Regimes seen:       {dict(regime_counts)}")
    lines.append(f"Shrinkage factor:   {SHRINKAGE}  (pulls p toward 0.50)")
    lines.append(f"s2p multiplier:     0.18  (reduced from 0.35)")
    lines.append(f"Improvements:       1) regime-conditional base rate  2) cross-sectional rel-mom")
    lines.append(f"                    3) multi-horizon base rate  4) momentum/MR regime switching")
    lines.append(f"")

    # Overall hit rate
    hit = resolved["outcome"].mean()
    lines.append(f"Overall hit rate:   {hit:.1%}  (across all p_win buckets)")

    # Brier score: mean((p_win - outcome)^2), lower is better, 0.25 = random
    brier = float(((resolved["p_win"] - resolved["outcome"]) ** 2).mean())
    brier_skill = 1 - brier / 0.25  # skill score vs random baseline
    lines.append(f"Brier score:        {brier:.4f}  (0.25 = random, lower is better)")
    lines.append(f"Brier skill score:  {brier_skill:+.1%}  vs. random baseline")
    lines.append(f"")

    # ── Regime-gated filter ────────────────────────────────────────────────────
    # Only trade longs in bull regimes, shorts in bear regimes.
    if "regime_match" in resolved.columns:
        gated    = resolved[resolved["regime_match"] == True]
        ungated  = resolved[resolved["regime_match"] == False]
        lines.append(f"REGIME-GATED FILTER  (longs in bull only · shorts in bear only)")
        lines.append(f"{'─'*60}")
        if not gated.empty:
            g_hit   = gated["outcome"].mean()
            g_brier = float(((gated["p_win"] - gated["outcome"]) ** 2).mean())
            g_skill = 1 - g_brier / 0.25
            lines.append(f"  Favorable-regime predictions:  n={len(gated):4}  "
                         f"hit={g_hit:.1%}  brier={g_brier:.4f}  skill={g_skill:+.1%}")
            for d in ["long", "short"]:
                sub = gated[gated["direction"] == d]
                if sub.empty: continue
                lines.append(f"    {d.upper():6}  n={len(sub):4}  hit={sub['outcome'].mean():.1%}  "
                             f"mean_p={sub['p_win'].mean():.1%}")
        if not ungated.empty:
            u_hit = ungated["outcome"].mean()
            lines.append(f"  Unfavorable-regime predictions: n={len(ungated):4}  "
                         f"hit={u_hit:.1%}  (skipped in gated strategy)")
        lines.append(f"")

        # Regime-gated calibration
        if not gated.empty:
            lines.append(f"  CALIBRATION — regime-gated predictions only")
            lines.append(f"  {'Predicted p_win':20} {'N':>5} {'Actual win %':>14} {'Δ (bias)':>10}")
            lines.append(f"  {'─'*56}")
            buckets_g = [(0.40,0.50,"40–50%"),(0.50,0.55,"50–55%"),
                         (0.55,0.60,"55–60%"),(0.60,0.65,"60–65%"),(0.65,1.01,"65%+")]
            for lo, hi, label in buckets_g:
                sub = gated[(gated["p_win"] >= lo) & (gated["p_win"] < hi)]
                if sub.empty: continue
                actual = sub["outcome"].mean()
                mid    = (lo + hi) / 2 if hi < 1 else 0.68
                bias   = actual - mid
                flag   = " ✓" if abs(bias) < 0.05 else (" ↑ over" if bias < 0 else " ↑ under")
                lines.append(f"  {label:17} {len(sub):>5}    {actual:>10.1%}    {bias:>+7.1%}{flag}")
            lines.append(f"  {'─'*56}")

        # Regime-gated Kelly P&L
        lines.append(f"")
        lines.append(f"  KELLY P&L — regime-gated trades only")
        capital_g = 10_000.0
        for _, row in gated.sort_values("as_of").iterrows():
            kelly = max(0.0, (2 * row["p_win"] - 1) * 0.5)
            size  = capital_g * kelly * 0.1
            ret   = row["actual_ret"] if row["direction"] == "long" else -row["actual_ret"]
            pnl   = size * ret - size * 0.0005
            capital_g += pnl
        lines.append(f"  Starting capital:  $10,000")
        lines.append(f"  Ending capital:    ${capital_g:,.0f}")
        lines.append(f"  Total return:      {(capital_g/10000-1):+.1%}  "
                     f"(n={len(gated)} trades, {len(gated)/n_total:.0%} of all predictions)")
        # Compound filter: regime-gated AND no earnings in window
        if "earnings_in_window" in resolved.columns:
            clean = gated[gated["earnings_in_window"] == False]
            risky = gated[gated["earnings_in_window"] == True]
            lines.append(f"")
            lines.append(f"  EARNINGS FILTER (within regime-gated set)")
            if not clean.empty:
                lines.append(f"  No earnings in window:  n={len(clean):4}  "
                             f"hit={clean['outcome'].mean():.1%}  "
                             f"(vs {gated['outcome'].mean():.1%} pre-filter)")
            if not risky.empty:
                lines.append(f"  Earnings in window:     n={len(risky):4}  "
                             f"hit={risky['outcome'].mean():.1%}  ← noise floor")

            # Best-case: regime-gated + no earnings
            if not clean.empty:
                c_hit   = clean["outcome"].mean()
                c_brier = float(((clean["p_win"] - clean["outcome"]) ** 2).mean())
                c_skill = 1 - c_brier / 0.25
                lines.append(f"")
                lines.append(f"  BEST-CASE FILTER (regime + no earnings)")
                lines.append(f"  n={len(clean):4}  hit={c_hit:.1%}  "
                             f"brier={c_brier:.4f}  skill={c_skill:+.1%}  "
                             f"({len(clean)/n_total:.0%} of all predictions)")
                capital_c = 10_000.0
                for _, row in clean.sort_values("as_of").iterrows():
                    kelly = max(0.0, (2 * row["p_win"] - 1) * 0.5)
                    size  = capital_c * kelly * 0.1
                    ret   = row["actual_ret"] if row["direction"] == "long" else -row["actual_ret"]
                    capital_c += size * ret - size * 0.0005
                lines.append(f"  Kelly P&L: ${capital_c:,.0f}  ({(capital_c/10000-1):+.1%})")

        lines.append(f"")
        lines.append(f"{'─'*60}")
        lines.append(f"")

    # Calibration by bucket
    lines.append(f"{'CALIBRATION — does predicted probability match actual win rate?':}")
    lines.append(f"{'─'*60}")
    lines.append(f"{'Predicted p_win':20} {'N':>5} {'Actual win %':>14} {'Δ (bias)':>10}")
    lines.append(f"{'─'*60}")
    buckets = [(0.0,0.40,"<40%"),(0.40,0.50,"40–50%"),(0.50,0.55,"50–55%"),
               (0.55,0.60,"55–60%"),(0.60,0.65,"60–65%"),(0.65,0.70,"65–70%"),(0.70,1.01,">70%")]
    calibration_rows = []
    for lo, hi, label in buckets:
        sub = resolved[(resolved["p_win"] >= lo) & (resolved["p_win"] < hi)]
        if sub.empty:
            continue
        actual = sub["outcome"].mean()
        mid    = (lo + hi) / 2 if hi < 1 else 0.75
        bias   = actual - mid
        calibration_rows.append((label, len(sub), actual, bias))
        flag   = " ✓" if abs(bias) < 0.05 else (" ↑ overconfident" if bias < 0 else " ↑ underconfident")
        lines.append(f"  {label:17} {len(sub):>5}    {actual:>10.1%}    {bias:>+7.1%}{flag}")
    lines.append(f"{'─'*60}")
    lines.append(f"")

    # By direction
    lines.append(f"BY DIRECTION")
    for d in ["long","short"]:
        sub = resolved[resolved["direction"]==d]
        if sub.empty: continue
        lines.append(f"  {d.upper():6}  n={len(sub):4}  hit={sub['outcome'].mean():.1%}  "
                     f"mean_p={sub['p_win'].mean():.1%}  "
                     f"brier={((sub['p_win']-sub['outcome'])**2).mean():.4f}")
    lines.append(f"")

    # By horizon
    lines.append(f"BY HORIZON")
    for h in sorted(resolved["horizon"].unique()):
        sub = resolved[resolved["horizon"]==h]
        lines.append(f"  {h:3}d  n={len(sub):4}  hit={sub['outcome'].mean():.1%}  "
                     f"mean_p={sub['p_win'].mean():.1%}")
    lines.append(f"")

    # By regime
    if "regime" in resolved.columns:
        lines.append(f"BY MACRO REGIME")
        regime_order = ["bull_low_vol","bull_normal","bull_high_vol","bear_normal","bear_high_vol","neutral"]
        for r in regime_order:
            sub = resolved[resolved["regime"]==r]
            if sub.empty: continue
            long_sub  = sub[sub["direction"]=="long"]
            short_sub = sub[sub["direction"]=="short"]
            long_hit  = f"{long_sub['outcome'].mean():.1%}" if not long_sub.empty else "—"
            short_hit = f"{short_sub['outcome'].mean():.1%}" if not short_sub.empty else "—"
            lines.append(f"  {r:16}  n={len(sub):4}  hit={sub['outcome'].mean():.1%}  "
                         f"mean_p={sub['p_win'].mean():.1%}  "
                         f"long_hit={long_hit}  short_hit={short_hit}")
        lines.append(f"")

    # Relative momentum quartile analysis (improvement 2 validation)
    if "rel_mom_score" in resolved.columns:
        lines.append(f"RELATIVE MOMENTUM QUARTILE ANALYSIS (improvement 2)")
        lines.append(f"  Do stocks outperforming their sector actually win more often?")
        q25 = resolved["rel_mom_score"].quantile(0.25)
        q75 = resolved["rel_mom_score"].quantile(0.75)
        for label, mask in [
            ("Top quartile (rel outperformers)", resolved["rel_mom_score"] >= q75),
            ("Middle 50%",                       (resolved["rel_mom_score"] >= q25) & (resolved["rel_mom_score"] < q75)),
            ("Bottom quartile (rel underperformers)", resolved["rel_mom_score"] < q25),
        ]:
            sub = resolved[mask]
            if sub.empty: continue
            long_sub  = sub[sub["direction"]=="long"]
            short_sub = sub[sub["direction"]=="short"]
            lh = f"{long_sub['outcome'].mean():.1%}" if not long_sub.empty else "—"
            sh = f"{short_sub['outcome'].mean():.1%}" if not short_sub.empty else "—"
            lines.append(f"  {label:42}  n={len(sub):4}  long_hit={lh}  short_hit={sh}")
        lines.append(f"  (Top quartile longs should outperform bottom quartile longs if signal is real)")
        lines.append(f"")

    # Top / bottom tickers
    by_ticker = resolved.groupby("ticker").agg(
        n=("outcome","count"), hit=("outcome","mean"),
        mean_p=("p_win","mean"), brier=(("p_win"),"var")
    ).reset_index().sort_values("hit", ascending=False)
    lines.append(f"TOP 5 TICKERS (by actual hit rate)")
    for _, row in by_ticker.head(5).iterrows():
        lines.append(f"  {row['ticker']:6}  n={int(row['n']):3}  hit={row['hit']:.1%}  mean_p={row['mean_p']:.1%}")
    lines.append(f"")
    lines.append(f"BOTTOM 5 TICKERS (by actual hit rate)")
    for _, row in by_ticker.tail(5).iterrows():
        lines.append(f"  {row['ticker']:6}  n={int(row['n']):3}  hit={row['hit']:.1%}  mean_p={row['mean_p']:.1%}")
    lines.append(f"")

    # Kelly-weighted simulated P&L (illustrative, not real trading)
    lines.append(f"KELLY-WEIGHTED SIMULATED P&L (illustrative)")
    lines.append(f"  Assumes $10,000 portfolio, size = (2p-1) * 0.5 of remaining capital per trade")
    lines.append(f"  Costs: 0.05% per trade (slippage + commission)")
    capital = 10_000.0
    for _, row in resolved.sort_values("as_of").iterrows():
        kelly = max(0.0, (2 * row["p_win"] - 1) * 0.5)
        size  = capital * kelly * 0.1   # max 10% per trade
        ret   = row["actual_ret"] if row["direction"] == "long" else -row["actual_ret"]
        pnl   = size * ret - size * 0.0005
        capital += pnl
    lines.append(f"  Starting capital:  $10,000")
    lines.append(f"  Ending capital:    ${capital:,.0f}")
    lines.append(f"  Total return:      {(capital/10000-1):+.1%}")
    lines.append(f"")
    lines.append(f"{'='*60}")
    lines.append(f"NOTE: Statistical analysis only. Not financial advice.")
    lines.append(f"      Past model performance does not guarantee future results.")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradeOdds v2 Walk-Forward Backtest")
    parser.add_argument("--tickers",   nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--horizon",   type=int,  default=90, help="Days forward (30, 60, 90, 180)")
    parser.add_argument("--direction", default="both", choices=["long","short","both"])
    parser.add_argument("--start",     default="2022-01-03")
    parser.add_argument("--end",       default="2023-06-30",
                        help="End of test-date window (leave room for horizon resolution)")
    parser.add_argument("--step",      type=int,  default=2, help="Weeks between test dates")
    parser.add_argument("--out",       default="backtest_results.csv")
    args = parser.parse_args()

    directions = ["long","short"] if args.direction == "both" else [args.direction]
    dates      = test_dates(args.start, args.end, args.step)

    print(f"TradeOdds v2 Walk-Forward Backtest")
    print(f"Tickers:    {len(args.tickers)}  |  Horizon: {args.horizon}d  |  "
          f"Dates: {len(dates)}  |  Directions: {directions}")
    print(f"Test window: {args.start} to {args.end}")
    print(f"Downloading price history...")

    # Pre-download SPY, VIX, and all sector ETFs
    spy_full = get_full_history("SPY")
    vix_full = get_full_history("^VIX")
    sector_data = {}
    for etf in SECTOR_ETFS:
        sector_data[etf] = get_full_history(etf)
        sys.stdout.write(f"\r  Downloaded {etf:6}   ")
        sys.stdout.flush()

    # Precompute regime label for every historical date (used by regime-conditional base rate)
    print(f"\n  Building regime series...")
    regime_series = _build_regime_series(spy_full, vix_full)

    all_data = {}
    for tk in args.tickers:
        all_data[tk] = get_full_history(tk)
        sys.stdout.write(f"\r  Downloaded {tk:6}   ")
        sys.stdout.flush()

    # Pre-fetch SEC EDGAR fundamentals for all tickers (downloads once per ticker, cached)
    print(f"\n  Fetching SEC EDGAR fundamentals (point-in-time)...")
    prefetch_tickers(args.tickers)
    print(f"  EDGAR prefetch complete.")
    print(f"\nRunning predictions...")

    rows = []
    total = len(args.tickers) * len(dates) * len(directions)
    done  = 0
    for ticker in args.tickers:
        full_df    = all_data[ticker]
        etf_ticker = TICKER_SECTOR_ETF.get(ticker, "XLK")
        sector_df  = sector_data.get(etf_ticker, pd.DataFrame())
        if full_df.empty:
            done += len(dates) * len(directions)
            continue
        for as_of in dates:
            for direction in directions:
                result = predict(ticker, as_of, direction, args.horizon,
                                 full_df, spy_full, vix_full,
                                 sector_df, regime_series,
                                 edgar_cache=None)   # per-call lookup uses disk cache
                if result:
                    resolved = resolve(result, full_df)
                    rows.append(resolved)
                done += 1
                if done % 50 == 0:
                    sys.stdout.write(f"\r  {done}/{total} predictions computed...  ")
                    sys.stdout.flush()

    print(f"\n  Done. {len(rows)} predictions.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"  Saved → {args.out}")

    report = calibration_report(df)
    report_file = args.out.replace(".csv", "_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"  Report → {report_file}")
    print()
    print(report)


if __name__ == "__main__":
    main()
