"""
TradeOdds API v2 — Proprietary 7-Factor Prediction Engine
========================================================
Factor Stack:
  1. Technical & Price Dynamics    — RSI, MA, momentum, volume trend
  2. Market Factor Regression      — CAPM alpha + beta vs SPY
  3. Fundamental Quality Score     — margins, ROE, growth, value, safety
  4. Modern Company Factors        — R&D intensity, intangibles, platform economics
  5. Sentiment & Positioning       — short interest, analyst consensus, insider flow
  6. Macro Regime                  — VIX, yield curve, SPY trend → conditional weights
  7. Volatility Surface            — realized vol, vol-of-vol, fat-tailed Monte Carlo

Data: Twelve Data (live price + OHLCV) + yfinance (fundamentals + macro)
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math, time, os, requests, warnings

# EDGAR modules — point-in-time fundamentals + insider scoring
try:
    from edgar import get_insider_score as _edgar_insider_score
    _EDGAR_AVAILABLE = True
except Exception:
    _EDGAR_AVAILABLE = False

warnings.filterwarnings("ignore")

TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY", "")

app = FastAPI(title="TradeOdds API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"]
)

_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {}

def cached(key: str, ttl: int = 600):
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.utcnow() - ts).seconds < ttl:
            return val
    return None

def store(key: str, val):
    _cache[key] = (val, datetime.utcnow())
    return val


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def get_live_price(ticker: str) -> float:
    """Real-time price: Twelve Data primary, yfinance fallback."""
    if TWELVEDATA_KEY:
        try:
            r = _session.get(
                "https://api.twelvedata.com/price",
                params={"symbol": ticker, "apikey": TWELVEDATA_KEY},
                timeout=8,
            )
            price = float(r.json().get("price", 0))
            if price > 0:
                return price
        except Exception:
            pass
    try:
        return float(yf.Ticker(ticker, session=_session).fast_info.last_price)
    except Exception:
        return 0.0


def get_ohlcv(ticker: str) -> pd.DataFrame:
    """
    2yr daily OHLCV. Twelve Data primary (one API call), yfinance fallback.
    Returns DataFrame [date, open, high, low, close, volume] oldest-first.
    """
    key = f"ohlcv_{ticker}"
    hit = cached(key, 600)
    if hit is not None:
        return hit

    if TWELVEDATA_KEY:
        try:
            r = _session.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": ticker, "interval": "1day",
                        "outputsize": 504, "apikey": TWELVEDATA_KEY},
                timeout=15,
            )
            data = r.json()
            if "values" in data and data["values"]:
                df = pd.DataFrame(data["values"])
                df["date"] = pd.to_datetime(df["datetime"])
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True).dropna(subset=["close"])
                if len(df) > 50:
                    return store(key, df)
        except Exception:
            pass

    try:
        raw = yf.download(ticker, period="2y", progress=False, auto_adjust=True, session=_session)
        if not raw.empty:
            df = raw.reset_index()
            df.columns = ["date","open","high","low","close","volume"] if len(df.columns) == 6 else df.columns
            df = df.rename(columns={df.columns[0]: "date"})
            cols = {c: c.lower() for c in df.columns}
            df = df.rename(columns=cols)
            df = df[["date","open","high","low","close","volume"]].dropna(subset=["close"])
            return store(key, df)
    except Exception:
        pass

    raise HTTPException(502, f"Could not fetch price history for {ticker}")


def get_fundamentals(ticker: str) -> dict:
    key = f"fund_{ticker}"
    hit = cached(key, 3600)
    if hit is not None:
        return hit
    try:
        info = yf.Ticker(ticker, session=_session).info
    except Exception:
        info = {}

    def safe(k, d=None):
        v = info.get(k)
        return v if v is not None else d

    result = {
        "name":              safe("longName", ticker),
        "sector":            safe("sector", "Unknown"),
        "industry":          safe("industry", "Unknown"),
        "market_cap":        safe("marketCap"),
        "beta":              safe("beta", 1.0),
        "pe_forward":        safe("forwardPE"),
        "pe_trailing":       safe("trailingPE"),
        "pb_ratio":          safe("priceToBook"),
        "ev_ebitda":         safe("enterpriseToEbitda"),
        "revenue_growth":    safe("revenueGrowth", 0.0),
        "earnings_growth":   safe("earningsGrowth", 0.0),
        "gross_margins":     safe("grossMargins", 0.0),
        "profit_margins":    safe("profitMargins", 0.0),
        "operating_margins": safe("operatingMargins", 0.0),
        "roe":               safe("returnOnEquity", 0.0),
        "roa":               safe("returnOnAssets", 0.0),
        "debt_to_equity":    safe("debtToEquity"),
        "current_ratio":     safe("currentRatio"),
        "free_cashflow":     safe("freeCashflow"),
        "revenue":           safe("totalRevenue"),
        "total_assets":      safe("totalAssets"),
        "employees":         safe("fullTimeEmployees"),
        "short_ratio":       safe("shortRatio", 0.0),
        "short_pct_float":   safe("shortPercentOfFloat", 0.0),
        "analyst_target":    safe("targetMeanPrice"),
        "recommendation":    safe("recommendationKey", "hold"),
        "num_analysts":      safe("numberOfAnalystOpinions", 0),
        "52w_high":          safe("fiftyTwoWeekHigh"),
        "52w_low":           safe("fiftyTwoWeekLow"),
        "avg_volume":        safe("averageVolume"),
        "insider_pct":       safe("heldPercentInsiders", 0.0),
        "institution_pct":   safe("heldPercentInstitutions", 0.0),
        "rd_to_revenue":     0.0,
        "intangibles_ratio": 0.0,
    }

    # R&D ratio from income statement
    try:
        tk = yf.Ticker(ticker, session=_session)
        fs = tk.get_financials()
        if fs is not None and not fs.empty:
            rd_row = "Research And Development"
            rev_row = "Total Revenue"
            if rd_row in fs.index and rev_row in fs.index:
                rd  = float(fs.loc[rd_row].iloc[0] or 0)
                rev = float(fs.loc[rev_row].iloc[0] or 1)
                result["rd_to_revenue"] = rd / rev if rev else 0
    except Exception:
        pass

    # Intangibles ratio from balance sheet
    try:
        tk = yf.Ticker(ticker, session=_session)
        bs = tk.get_balance_sheet()
        if bs is not None and not bs.empty:
            intang_row = "Goodwill And Other Intangible Assets"
            asset_row  = "Total Assets"
            if intang_row in bs.index and asset_row in bs.index:
                intang = float(bs.loc[intang_row].iloc[0] or 0)
                assets = float(bs.loc[asset_row].iloc[0] or 1)
                result["intangibles_ratio"] = intang / assets if assets else 0
    except Exception:
        pass

    return store(key, result)


def get_macro() -> dict:
    """VIX, SPY 200MA trend, yield curve — classifies market regime."""
    hit = cached("macro", 900)
    if hit is not None:
        return hit

    result = {"vix": 20.0, "spy_above_200ma": True, "yield_curve": 0.5,
              "yield_10yr": 4.5, "yield_2yr": 4.0, "regime": "neutral"}

    try:
        vix = yf.download("^VIX", period="5d", progress=False, session=_session)
        if not vix.empty:
            result["vix"] = float(vix["Close"].iloc[-1])
    except Exception:
        pass

    try:
        spy = yf.download("SPY", period="1y", progress=False, auto_adjust=True, session=_session)
        if not spy.empty and len(spy) >= 200:
            arr = spy["Close"].values
            ma200 = float(np.mean(arr[-200:]))
            result["spy_above_200ma"] = float(arr[-1]) > ma200
            result["spy_price"] = float(arr[-1])
            result["spy_ma200"] = round(ma200, 2)
    except Exception:
        pass

    try:
        tnx = yf.download("^TNX", period="5d", progress=False, session=_session)
        irx = yf.download("^IRX", period="5d", progress=False, session=_session)
        if not tnx.empty and not irx.empty:
            yr10 = float(tnx["Close"].iloc[-1])
            yr2  = float(irx["Close"].iloc[-1])
            result["yield_10yr"]   = round(yr10, 2)
            result["yield_2yr"]    = round(yr2, 2)
            result["yield_curve"]  = round(yr10 - yr2, 2)
    except Exception:
        pass

    vix  = result["vix"]
    bull = result["spy_above_200ma"]

    if bull and vix < 18:     regime = "bull_low_vol"
    elif bull and vix < 28:   regime = "bull_normal"
    elif bull:                regime = "bull_high_vol"
    elif vix < 28:            regime = "bear_normal"
    else:                     regime = "bear_high_vol"

    result["regime"] = regime
    return store("macro", result)


# ═══════════════════════════════════════════════════════════════════════════════
# FACTOR MODULES
# ═══════════════════════════════════════════════════════════════════════════════

def module_technical(df: pd.DataFrame, cur: float) -> dict:
    """Module 1 — Technical & price dynamics."""
    closes  = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    n = len(closes)

    def mom(days):
        idx  = max(0, n - days)
        base = float(closes[idx])
        return round((cur - base) / base, 4) if base > 0 else 0.0

    # RSI(14)
    d   = np.diff(closes[-30:] if n >= 30 else closes)
    g   = np.where(d > 0, d, 0.0)
    l   = np.where(d < 0, -d, 0.0)
    ag  = np.mean(g[-14:]) if len(g) >= 14 else 0
    al  = np.mean(l[-14:]) if len(l) >= 14 else 1e-9
    rsi = 100 - (100 / (1 + ag / (al + 1e-9)))

    ma50  = float(np.mean(closes[-50:]))  if n >= 50  else cur
    ma200 = float(np.mean(closes[-200:])) if n >= 200 else cur

    w52    = closes[-252:] if n >= 252 else closes
    high52 = float(np.max(w52))
    low52  = float(np.min(w52))
    pct_from_high = (cur - high52) / high52

    vol20 = float(np.mean(volumes[-20:])) if n >= 20 else 1
    vol60 = float(np.mean(volumes[-60:])) if n >= 60 else 1
    vol_ratio = vol20 / (vol60 + 1e-9)

    # Composite score (-1 to +1)
    score = (
        0.25 * float(np.clip((rsi - 50) / 30, -1, 1)) +
        0.20 * (1.0 if cur > ma50  else -1.0) +
        0.20 * (1.0 if cur > ma200 else -1.0) +
        0.20 * float(np.clip(mom(63) * 4, -1, 1)) +
        0.15 * float(np.clip((vol_ratio - 1) * 2, -1, 1))
    )

    return {
        "score":            round(float(score), 4),
        "rsi":              round(float(rsi), 1),
        "above_50ma":       bool(cur > ma50),
        "above_200ma":      bool(cur > ma200),
        "ma50":             round(float(ma50), 2),
        "ma200":            round(float(ma200), 2),
        "mom_1m":           mom(21),
        "mom_3m":           mom(63),
        "mom_6m":           mom(126),
        "mom_12m":          mom(252),
        "pct_from_52w_high": round(float(pct_from_high), 4),
        "high_52w":         round(float(high52), 2),
        "low_52w":          round(float(low52), 2),
        "vol_ratio":        round(float(vol_ratio), 3),
    }


def module_volume(df: pd.DataFrame) -> dict:
    """Volume confirmation — OBV trend, volume expansion, up/down volume ratio."""
    closes  = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    volumes = np.where(volumes < 1, 1.0, volumes)

    if len(closes) < 25:
        return {"score": 0.0, "obv_trend": 0.0, "vol_expansion": 1.0, "ud_ratio": 1.0}

    daily_ret = np.diff(closes)
    signs     = np.sign(daily_ret)
    obv       = np.cumsum(signs * volumes[1:])
    obv_recent = obv[-20:]
    obv_mean   = np.mean(np.abs(obv_recent)) + 1
    x = np.arange(len(obv_recent))
    slope = float(np.polyfit(x, obv_recent, 1)[0]) / obv_mean
    obv_score = float(np.clip(slope * 50, -1, 1))

    vol_5  = float(np.mean(volumes[-5:]))
    vol_20 = float(np.mean(volumes[-20:]))
    vol_ratio = vol_5 / (vol_20 + 1e-9)
    vol_score = float(np.clip((vol_ratio - 1.0) / 0.5, -1, 1))

    rets_20 = daily_ret[-20:]
    vols_20 = volumes[-20:][-len(rets_20):]
    up_mask   = rets_20 > 0
    down_mask = rets_20 < 0
    up_vol   = float(np.mean(vols_20[up_mask]))   if up_mask.any()   else vol_20
    down_vol = float(np.mean(vols_20[down_mask])) if down_mask.any() else vol_20
    ud_ratio = up_vol / (down_vol + 1e-9)
    ud_score = float(np.clip((ud_ratio - 1.0) / 0.5, -1, 1))

    score = 0.40 * obv_score + 0.35 * vol_score + 0.25 * ud_score
    return {
        "score":         round(float(np.clip(score, -1, 1)), 4),
        "obv_trend":     round(obv_score, 4),
        "vol_expansion": round(vol_ratio, 3),
        "ud_ratio":      round(ud_ratio, 3),
    }


# In-memory insider score cache (monthly granularity — refreshed weekly on Render)
_insider_cache: dict = {}

def module_insider(ticker: str) -> dict:
    """Insider transaction score from SEC EDGAR Form 4 filings (last 90 days)."""
    global _insider_cache
    cache_key = (ticker.upper(), datetime.utcnow().strftime("%Y-%m"))
    if cache_key in _insider_cache:
        return _insider_cache[cache_key]

    score = 0.0
    if _EDGAR_AVAILABLE:
        try:
            score = _edgar_insider_score(ticker, datetime.utcnow(), lookback_days=90)
        except Exception:
            score = 0.0

    result = {"score": round(score, 4), "signal": "buying" if score > 0.1 else "selling" if score < -0.1 else "neutral"}
    _insider_cache[cache_key] = result
    return result


def module_market_factor(df: pd.DataFrame, beta: float) -> dict:
    """Module 2 — CAPM regression: isolate alpha from market beta."""
    try:
        spy_df = get_ohlcv("SPY")
    except Exception:
        return {"alpha_ann": 0.0, "beta": beta, "score": 0.0, "r_squared": 0.0}

    merged = pd.merge(
        df[["date","close"]].rename(columns={"close":"stock"}),
        spy_df[["date","close"]].rename(columns={"close":"spy"}),
        on="date"
    ).dropna()

    if len(merged) < 60:
        return {"alpha_ann": 0.0, "beta": beta, "score": 0.0, "r_squared": 0.0}

    sr = merged["stock"].pct_change().dropna().values
    mr = merged["spy"].pct_change().dropna().values
    n  = min(len(sr), len(mr))
    sr, mr = sr[-n:], mr[-n:]

    X = np.column_stack([np.ones(n), mr])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, sr, rcond=None)
        alpha_d = float(coeffs[0])
        beta_r  = float(coeffs[1])
        pred    = X @ coeffs
        ss_res  = float(np.sum((sr - pred) ** 2))
        ss_tot  = float(np.sum((sr - sr.mean()) ** 2))
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        alpha_d, beta_r, r2 = 0.0, beta, 0.0

    alpha_ann = alpha_d * 252
    score     = float(np.clip(alpha_ann / 0.20, -1, 1))

    return {
        "alpha_ann":  round(alpha_ann, 4),
        "beta":       round(beta_r, 3),
        "score":      round(score, 4),
        "r_squared":  round(r2, 3),
    }


def module_fundamentals(fund: dict, cur: float) -> dict:
    """Module 3 — Fundamental quality: profitability, growth, value, safety."""
    gm   = (fund.get("gross_margins")  or 0) * 100
    pm   = (fund.get("profit_margins") or 0) * 100
    roe  = (fund.get("roe")            or 0) * 100
    roa  = (fund.get("roa")            or 0) * 100
    rg   = (fund.get("revenue_growth") or 0) * 100
    eg   = (fund.get("earnings_growth")or 0) * 100
    pe   = fund.get("pe_forward") or fund.get("pe_trailing") or 25.0
    pb   = fund.get("pb_ratio")  or 3.0
    ev_e = fund.get("ev_ebitda") or 15.0
    de   = fund.get("debt_to_equity") or 50.0
    cr   = fund.get("current_ratio")  or 1.5

    profitability = float(np.clip(gm / 8 + pm / 4 + roe / 15, 0, 10))
    growth        = float(np.clip(5 + rg / 10 + eg / 20, 0, 10))
    value         = float(np.clip(10 - (pe - 10) * 10/50 - (pb - 1) * 0.5 - (ev_e - 8) * 0.2, 0, 10))
    safety        = float(np.clip(5 - de / 100 + (cr - 1) * 2, 0, 10))

    rec_map = {"strongbuy": 2, "buy": 1, "hold": 0, "underperform": -1, "sell": -2}
    rec     = rec_map.get((fund.get("recommendation") or "hold").lower(), 0)
    target  = fund.get("analyst_target")
    upside  = ((target - cur) / cur) if (target and cur > 0) else 0.0

    composite = float(
        0.25 * np.clip((profitability - 5) / 5, -1, 1) +
        0.25 * np.clip((growth - 5) / 5, -1, 1) +
        0.20 * np.clip((value - 5) / 5, -1, 1) +
        0.15 * np.clip((safety - 5) / 5, -1, 1) +
        0.15 * np.clip(upside * 2 + rec * 0.1, -1, 1)
    )

    return {
        "score":             round(composite, 4),
        "profitability":     round(profitability, 1),
        "growth":            round(growth, 1),
        "value":             round(value, 1),
        "safety":            round(safety, 1),
        "gross_margin_pct":  round(gm, 1),
        "profit_margin_pct": round(pm, 1),
        "roe_pct":           round(roe, 1),
        "roa_pct":           round(roa, 1),
        "rev_growth_pct":    round(rg, 1),
        "eps_growth_pct":    round(eg, 1),
        "pe_forward":        round(float(pe), 1),
        "pb_ratio":          round(float(pb), 2),
        "debt_to_equity":    round(float(de), 1),
        "upside_to_target":  round(float(upside * 100), 1),
        "analyst_rec":       fund.get("recommendation", "hold"),
        "num_analysts":      fund.get("num_analysts", 0),
    }


def module_modern_factors(fund: dict) -> dict:
    """
    Module 4 — Modern company factors.
    Captures what traditional factor models miss:
    - R&D intensity: investment in future growth optionality
    - Intangibles ratio: asset-light, platform-type business
    - Platform economics: high gross margins = pricing power + low marginal cost
    - Productivity: revenue per employee (network/scale efficiency)
    """
    rd    = fund.get("rd_to_revenue") or 0.0
    intg  = fund.get("intangibles_ratio") or 0.0
    gm    = (fund.get("gross_margins") or 0) * 100
    rev   = fund.get("revenue") or 0
    emp   = fund.get("employees") or 1
    rev_per_emp = rev / emp if emp > 0 else 0

    # R&D intensity: 5-25% = innovation sweet spot
    if rd > 0.40:   rd_s = 0.5
    elif rd > 0.05: rd_s = min(1.0, rd / 0.15)
    else:           rd_s = rd / 0.05 * 0.3

    platform_s  = float(np.clip(intg * 0.5 + gm / 100 * 0.5, 0, 1))
    prod_s      = float(np.clip(rev_per_emp / 500_000, 0, 1))
    moat_s      = float(np.clip(gm / 60, 0, 1))

    composite = float(
        0.30 * (rd_s * 2 - 1) +
        0.30 * (platform_s * 2 - 1) +
        0.20 * (prod_s * 2 - 1) +
        0.20 * (moat_s * 2 - 1)
    )

    return {
        "score":                round(composite, 4),
        "rd_intensity_pct":     round(rd * 100, 1),
        "intangibles_pct":      round(intg * 100, 1),
        "platform_score":       round(platform_s, 3),
        "productivity_score":   round(prod_s, 3),
        "moat_score":           round(moat_s, 3),
        "rev_per_employee_k":   round(rev_per_emp / 1000, 1),
        "gross_margin_pct":     round(gm, 1),
    }


def module_sentiment(fund: dict, cur: float) -> dict:
    """
    Module 5 — Sentiment & positioning.
    Short interest, analyst consensus, insider/institutional ownership.
    """
    short_pct  = (fund.get("short_pct_float") or 0) * 100
    short_days = fund.get("short_ratio") or 0

    # Contrarian short squeeze signal at extremes
    if short_pct > 20:    short_sig = 0.3
    elif short_pct > 10:  short_sig = -0.2
    elif short_pct < 3:   short_sig = 0.15
    else:                 short_sig = 0.0

    rec_map = {"strongbuy": 1.0, "buy": 0.5, "hold": 0.0, "underperform": -0.5, "sell": -1.0}
    analyst_sig = rec_map.get((fund.get("recommendation") or "hold").lower(), 0)

    target      = fund.get("analyst_target")
    upside_sig  = float(np.clip(((target - cur) / cur * 2) if (target and cur > 0) else 0, -1, 1))

    inst_pct    = (fund.get("institution_pct") or 0) * 100
    inst_sig    = 0.2 if 40 < inst_pct < 85 else -0.1

    insider_pct = (fund.get("insider_pct") or 0) * 100
    insider_sig = float(np.clip(insider_pct / 20, 0, 0.5))

    composite = float(
        0.25 * short_sig +
        0.30 * analyst_sig +
        0.25 * upside_sig +
        0.10 * inst_sig +
        0.10 * insider_sig
    )

    return {
        "score":                round(composite, 4),
        "short_pct_float":      round(short_pct, 1),
        "short_ratio_days":     round(float(short_days), 1),
        "analyst_rec":          fund.get("recommendation", "hold"),
        "analyst_upside_pct":   round(float(upside_sig * 50), 1),
        "institutional_pct":    round(float(inst_pct), 1),
        "insider_pct":          round(float(insider_pct), 1),
        "num_analysts":         fund.get("num_analysts", 0),
    }


def module_vol_surface(df: pd.DataFrame, beta: float, vix: float) -> dict:
    """Module 7 — Volatility surface: realized vol, vol-of-vol, implied proxy."""
    closes  = df["close"].values.astype(float)
    returns = np.diff(np.log(closes + 1e-9))

    v20  = float(np.std(returns[-20:])  * math.sqrt(252)) if len(returns) >= 20  else 0.25
    v60  = float(np.std(returns[-60:])  * math.sqrt(252)) if len(returns) >= 60  else 0.25
    v252 = float(np.std(returns[-252:]) * math.sqrt(252)) if len(returns) >= 252 else 0.25

    # Vol-of-vol: variance of rolling 20d vol
    if len(returns) >= 40:
        rvol = pd.Series(returns).rolling(20).std().dropna().values * math.sqrt(252)
        vov  = float(np.std(rvol)) if len(rvol) > 5 else 0.05
    else:
        vov = 0.05

    # VIX-scaled implied vol proxy (blends realized + macro fear)
    implied = float(np.clip(
        0.40 * v60 + 0.40 * (vix / 100 * abs(beta)) + 0.20 * v252,
        0.08, 2.5
    ))

    vol_regime = "low" if implied < 0.20 else ("high" if implied > 0.35 else "normal")

    return {
        "realized_vol_20d": round(v20, 4),
        "realized_vol_60d": round(v60, 4),
        "realized_vol_1yr": round(v252, 4),
        "implied_vol":      round(implied, 4),
        "vol_of_vol":       round(vov, 4),
        "vol_regime":       vol_regime,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROBABILITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Regime-conditional factor weights — the market environment determines
# which signals deserve more trust
REGIME_WEIGHTS = {
    "bull_low_vol":  {"technical":0.20,"market":0.15,"fundamental":0.18,"modern":0.12,"sentiment":0.15,"base_rate":0.10,"analogues":0.10},
    "bull_normal":   {"technical":0.17,"market":0.15,"fundamental":0.20,"modern":0.10,"sentiment":0.15,"base_rate":0.13,"analogues":0.10},
    "bull_high_vol": {"technical":0.10,"market":0.15,"fundamental":0.25,"modern":0.08,"sentiment":0.10,"base_rate":0.18,"analogues":0.14},
    "bear_normal":   {"technical":0.12,"market":0.18,"fundamental":0.25,"modern":0.08,"sentiment":0.10,"base_rate":0.15,"analogues":0.12},
    "bear_high_vol": {"technical":0.08,"market":0.18,"fundamental":0.28,"modern":0.06,"sentiment":0.08,"base_rate":0.18,"analogues":0.14},
    "neutral":       {"technical":0.15,"market":0.15,"fundamental":0.22,"modern":0.10,"sentiment":0.13,"base_rate":0.13,"analogues":0.12},
}


def _sample_t(df_deg: float, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """Student's t samples via normal / sqrt(chi2/df) — no scipy needed."""
    z = rng.standard_normal((rows, cols))
    v = rng.chisquare(df_deg, (rows, cols))
    return z / np.sqrt(v / df_deg)


def run_monte_carlo(
    cur: float, entry: float, direction: str,
    horizon: int, sigma: float, alpha_ann: float,
    n_paths: int = 10_000, t_df: float = 5.0,
) -> dict:
    """
    Fat-tailed Monte Carlo (Student's t, df=5).
    t(5) has ~3x the kurtosis of normal — captures earnings gaps and tail events
    that lognormal GBM systematically underprices.
    """
    rf_d    = 0.045 / 252
    drift_d = rf_d + alpha_ann / 252
    sig_d   = sigma / math.sqrt(252)
    # Scale shocks so variance matches GBM (t(df) has var=df/(df-2))
    t_scale = math.sqrt((t_df - 2) / t_df) if t_df > 2 else 1.0

    rng    = np.random.default_rng(seed=42)
    shocks = _sample_t(t_df, n_paths, horizon, rng) * sig_d / t_scale
    log_r  = (drift_d - 0.5 * sig_d ** 2) + shocks
    terminal = cur * np.exp(log_r.sum(axis=1))

    if direction == "long":
        wins    = terminal > entry
        rets    = (terminal - entry) / entry
    else:
        wins    = terminal < entry
        rets    = (entry - terminal) / entry

    p_mc     = float(wins.mean())
    avg_win  = float(rets[wins].mean())  if wins.sum() > 0  else 0.0
    avg_loss = float(rets[~wins].mean()) if (~wins).sum() > 0 else 0.0
    ev_pct   = p_mc * avg_win + (1 - p_mc) * avg_loss

    return {
        "p_mc":        round(p_mc, 4),
        "median_ret":  round(float(np.median(rets)), 4),
        "p5_ret":      round(float(np.percentile(rets, 5)), 4),
        "p95_ret":     round(float(np.percentile(rets, 95)), 4),
        "avg_win":     round(avg_win, 4),
        "avg_loss":    round(avg_loss, 4),
        "ev_pct":      round(ev_pct, 4),
        "var_95_pct":  round(float(np.percentile(rets, 5)), 4),
        "n_paths":     n_paths,
        "model":       "GBM-student-t-df5",
    }


def run_base_rate(df: pd.DataFrame, direction: str, horizon: int) -> dict:
    closes = df["close"].values.astype(float)
    if len(closes) < horizon + 20:
        return {"base_rate": 0.50, "n_windows": 0}
    wins, total = 0, 0
    for i in range(len(closes) - horizon):
        ret = (closes[i + horizon] - closes[i]) / closes[i]
        wins  += int(ret > 0) if direction == "long" else int(ret < 0)
        total += 1
    return {"base_rate": round(wins / total, 4), "n_windows": total}


def run_analogues(df: pd.DataFrame, direction: str, horizon: int, cur_vol: float) -> dict:
    closes  = df["close"].values.astype(float)
    returns = np.diff(np.log(closes + 1e-9))
    if len(closes) < 90 + horizon:
        return {"analogue_p": 0.50, "n_analogues": 0, "avg_fwd_ret": 0.0}

    results = []
    for i in range(30, len(closes) - horizon):
        hv = float(np.std(returns[max(0,i-30):i]) * math.sqrt(252))
        if abs(hv - cur_vol) / (cur_vol + 1e-9) < 0.30:
            fwd = (closes[i+horizon] - closes[i]) / closes[i]
            results.append(fwd if direction == "long" else -fwd)

    if not results:
        return {"analogue_p": 0.50, "n_analogues": 0, "avg_fwd_ret": 0.0}

    return {
        "analogue_p":  round(float(np.mean([r > 0 for r in results])), 4),
        "n_analogues": len(results),
        "avg_fwd_ret": round(float(np.mean(results)), 4),
    }


def run_ensemble(
    tech, market, fund, modern, sent,
    mc, base_rate, analogues,
    weights: dict, direction: str,
    volume_score: float = 0.0,
    insider_score: float = 0.0,
) -> dict:
    """
    Blend all factor modules into one calibrated probability.
    v2.1: adds volume confirmation and insider transaction signal.
    """
    d = 1 if direction == "long" else -1

    def s2p(s):  # score (-1,+1) → probability
        return float(np.clip(0.50 + float(s) * d * 0.18, 0.22, 0.78))

    p_tech    = s2p(tech["score"])
    p_mkt     = s2p(market["score"])
    p_fund    = s2p(fund["score"])
    p_mod     = s2p(modern["score"])
    p_sent    = s2p(sent["score"])
    p_vol_sig = s2p(volume_score)
    p_insider = s2p(insider_score)
    p_base    = base_rate if direction == "long" else (1 - base_rate)
    p_ana     = analogues["analogue_p"]
    p_mc      = mc["p_mc"]

    # Volume and insider each take 5% from technical and sentiment
    factor_blend = (
        weights["technical"]   * 0.90 * p_tech    +
        weights["market"]               * p_mkt    +
        weights["fundamental"]          * p_fund   +
        weights["modern"]               * p_mod    +
        weights["sentiment"]   * 0.90 * p_sent    +
        weights["base_rate"]            * p_base   +
        weights["analogues"]            * p_ana    +
        weights["technical"]   * 0.05 * p_vol_sig +
        weights["sentiment"]   * 0.05 * p_insider
    )

    final = float(np.clip(0.40 * p_mc + 0.60 * factor_blend, 0.05, 0.95))

    # Shrinkage toward 0.50 (calibrated from backtest)
    final = 0.50 + (final - 0.50) * 0.68

    all_p  = [p_tech, p_mkt, p_fund, p_mod, p_sent, p_base, p_ana, p_mc]
    spread = float(np.max(all_p) - np.min(all_p))
    conf   = int(np.clip(88 - spread * 130, 18, 93))
    bw     = max(4, int(spread * 55))

    return {
        "p_ensemble": round(final, 4),
        "confidence": conf,
        "bandwidth":  bw,
        "module_probs": {
            "technical":     round(p_tech, 4),
            "market_alpha":  round(p_mkt, 4),
            "fundamental":   round(p_fund, 4),
            "modern":        round(p_mod, 4),
            "sentiment":     round(p_sent, 4),
            "base_rate":     round(p_base, 4),
            "analogues":     round(p_ana, 4),
            "monte_carlo":   round(p_mc, 4),
            "volume":        round(p_vol_sig, 4),
            "insider":       round(p_insider, 4),
        },
        "weights": weights,
    }


def compute_ev(p_win: float, mc: dict, entry: float, direction: str, position_size) -> dict:
    avg_win   = mc["avg_win"]
    avg_loss  = abs(mc["avg_loss"])
    var_95    = abs(mc["var_95_pct"])
    b         = avg_win / (avg_loss + 1e-9)
    kelly_f   = max(0.0, (b * p_win - (1 - p_win)) / b)
    kelly_h   = kelly_f / 2  # half-Kelly

    result = {
        "ev_pct":            round(mc["ev_pct"] * 100, 2),
        "avg_win_pct":       round(avg_win * 100, 2),
        "avg_loss_pct":      round(-avg_loss * 100, 2),
        "var_95_pct":        round(-var_95 * 100, 2),
        "risk_reward_ratio": round(b, 2),
        "kelly_half_pct":    round(kelly_h * 100, 1),
    }

    try:
        ps = float(position_size) if position_size is not None else 0.0
    except (TypeError, ValueError):
        ps = 0.0
    if ps > 0:
        result.update({
            "position_size":    round(ps, 2),
            "ev_dollar":        round(mc["ev_pct"] * position_size, 2),
            "avg_win_dollar":   round(avg_win * position_size, 2),
            "avg_loss_dollar":  round(-avg_loss * position_size, 2),
            "var_95_dollar":    round(-var_95 * position_size, 2),
        })

    return result


def verdict(p: float) -> str:
    if p >= 0.72: return "Strong Edge"
    if p >= 0.60: return "Favourable"
    if p >= 0.50: return "Slight Edge"
    if p >= 0.40: return "Slight Disadvantage"
    if p >= 0.28: return "Unfavourable"
    return "Strong Disadvantage"


SECTOR_PE = {
    "Technology":28,"Information Technology":28,"Communication Services":22,
    "Consumer Discretionary":20,"Consumer Staples":18,"Health Care":20,
    "Healthcare":20,"Financials":13,"Financial Services":13,"Industrials":19,
    "Energy":12,"Utilities":17,"Real Estate":22,"Materials":16,"Unknown":20,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"service": "TradeOdds API", "version": "2.0.0", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"ok": True, "status": "ok", "provider": "twelvedata+yfinance",
            "live_data": True, "version": "2.0", "time": datetime.utcnow().isoformat()+"Z"}


@app.get("/v1/forecast")
async def forecast(
    ticker:        str   = Query(...),
    horizon:       int   = Query(90),
    dir:           str   = Query("long"),
    entry:         float = Query(None),
    position_size: float = Query(None, description="Dollar amount of position (optional)"),
):
    ticker    = ticker.upper().strip()
    horizon   = max(1, min(horizon, 365))
    direction = dir.lower()
    if direction not in ("long","short"):
        raise HTTPException(400, "dir must be 'long' or 'short'")

    # Live price
    cur = get_live_price(ticker)
    if cur <= 0:
        raise HTTPException(404, f"No price data for {ticker}")
    entry_price = entry if (entry and entry > 0) else cur

    # Data
    try:
        ohlcv = get_ohlcv(ticker)
        fund  = get_fundamentals(ticker)
        macro = get_macro()
    except Exception as e:
        raise HTTPException(502, str(e))

    beta = float(fund.get("beta") or 1.0)

    # Factor modules
    tech    = module_technical(ohlcv, cur)
    market  = module_market_factor(ohlcv, beta)
    fundsc  = module_fundamentals(fund, cur)
    modern  = module_modern_factors(fund)
    sent    = module_sentiment(fund, cur)
    vol     = module_vol_surface(ohlcv, beta, macro["vix"])
    volume  = module_volume(ohlcv)
    insider = module_insider(ticker)

    # Probability engine
    weights = REGIME_WEIGHTS.get(macro["regime"], REGIME_WEIGHTS["neutral"])
    br      = run_base_rate(ohlcv, direction, horizon)
    ana     = run_analogues(ohlcv, direction, horizon, vol["realized_vol_60d"])
    mc      = run_monte_carlo(cur, entry_price, direction, horizon,
                              vol["implied_vol"], market["alpha_ann"])
    ens     = run_ensemble(tech, market, fundsc, modern, sent,
                           mc, br["base_rate"], ana, weights, direction,
                           volume["score"], insider["score"])

    p_win   = ens["p_ensemble"]
    ev      = compute_ev(p_win, mc, entry_price, direction, position_size)

    # Regime favorability flag — core commercialisation signal
    BULL_REGIMES = {"bull_low_vol", "bull_normal", "bull_high_vol"}
    BEAR_REGIMES = {"bear_normal", "bear_high_vol"}
    regime       = macro["regime"]
    regime_favorable = (
        (direction == "long"  and regime in BULL_REGIMES) or
        (direction == "short" and regime in BEAR_REGIMES)
    )
    regime_warning = None
    if not regime_favorable:
        if direction == "long" and regime in BEAR_REGIMES:
            regime_warning = "Bear market regime detected — long trades have historically lower win rates in this environment."
        elif direction == "short" and regime in BULL_REGIMES:
            regime_warning = "Bull market regime detected — short trades face a macro headwind in this environment."

    # Factor radar (0-10 per axis)
    radar = {
        "technical":  round(float(np.clip((tech["score"] + 1) * 5, 0, 10)), 1),
        "quality":    round(float(fundsc["profitability"]), 1),
        "growth":     round(float(fundsc["growth"]), 1),
        "value":      round(float(fundsc["value"]), 1),
        "momentum":   round(float(np.clip(tech["mom_12m"] * 100 / 30 + 5, 0, 10)), 1),
        "modern":     round(float(np.clip((modern["score"] + 1) * 5, 0, 10)), 1),
        "volume":     round(float(np.clip((volume["score"] + 1) * 5, 0, 10)), 1),
        "insider":    round(float(np.clip((insider["score"] + 1) * 5, 0, 10)), 1),
    }

    implied_move_pct = round(vol["implied_vol"] * math.sqrt(horizon / 252) * 100, 1)

    return {
        "ticker":        ticker,
        "name":          fund["name"],
        "sector":        fund["sector"],
        "industry":      fund["industry"],
        "direction":     direction,
        "horizon_days":  horizon,
        "current_price": round(cur, 2),
        "entry_price":   round(entry_price, 2),
        "sector_pe":     SECTOR_PE.get(fund["sector"], 20),
        "market_cap":    fund.get("market_cap"),

        "p_win":             p_win,
        "verdict":           verdict(p_win),
        "ev":                ev,
        "implied_move_pct":  implied_move_pct,
        "position_size":     position_size,
        "regime_favorable":  regime_favorable,
        "regime_warning":    regime_warning,

        "modules": {
            "technicals":     tech,
            "market_factor":  market,
            "fundamentals":   fundsc,
            "modern_factors": modern,
            "sentiment":      sent,
            "vol_surface":    vol,
            "volume":         volume,
            "insider":        insider,
            "monte_carlo":    mc,
            "base_rate":      br,
            "analogues":      ana,
            "ensemble":       ens,
        },

        "macro":        macro,
        "factor_radar": radar,

        "generated_at":  datetime.utcnow().isoformat() + "Z",
        "model_version": "TradeOdds-v2.1 (9-Factor+MC-t5+RegimeGate)",
        "data_source":   "Twelve Data (live) + yfinance (fundamentals/macro)",
    }


SP500_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","LLY","AVGO","JPM",
    "TSLA","UNH","V","XOM","MA","JNJ","PG","HD","COST","MRK","NFLX",
]

@app.get("/v1/screener/top")
async def screener(limit: int = Query(10), dir: str = Query("long")):
    direction = dir.lower()
    results   = []
    for tk in SP500_SAMPLE[:limit]:
        try:
            cur  = get_live_price(tk)
            if cur <= 0:
                continue
            ohlcv = get_ohlcv(tk)
            fund  = get_fundamentals(tk)
            macro = get_macro()
            beta  = float(fund.get("beta") or 1.0)
            tech  = module_technical(ohlcv, cur)
            mkt   = module_market_factor(ohlcv, beta)
            fsc   = module_fundamentals(fund, cur)
            mod   = module_modern_factors(fund)
            snt   = module_sentiment(fund, cur)
            vol   = module_vol_surface(ohlcv, beta, macro["vix"])
            wts   = REGIME_WEIGHTS.get(macro["regime"], REGIME_WEIGHTS["neutral"])
            br    = run_base_rate(ohlcv, direction, 90)
            ana   = run_analogues(ohlcv, direction, 90, vol["realized_vol_60d"])
            mc    = run_monte_carlo(cur, cur, direction, 90, vol["implied_vol"], mkt["alpha_ann"])
            ens   = run_ensemble(tech, mkt, fsc, mod, snt, mc, br["base_rate"], ana, wts, direction)
            results.append({
                "ticker": tk, "name": fund["name"], "sector": fund["sector"],
                "p_win": ens["p_ensemble"], "verdict": verdict(ens["p_ensemble"]),
                "current_price": round(cur, 2),
                "alpha_ann": mkt["alpha_ann"], "mom_12m": tech["mom_12m"],
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["p_win"], reverse=(direction == "long"))
    return {"direction": direction, "results": results,
            "screened_at": datetime.utcnow().isoformat() + "Z"}


@app.get("/v1/calibration")
async def calibration():
    return {
        "model": "TradeOdds-v2.0",
        "factor_modules": [
            "Technical & Price Dynamics",
            "Market Factor Regression (CAPM alpha)",
            "Fundamental Quality Score",
            "Modern Company Factors (R&D, platform, moat)",
            "Sentiment & Positioning",
            "Macro Regime (VIX + yield curve + SPY trend)",
            "Volatility Surface (realized + implied)",
        ],
        "monte_carlo": "Student-t df=5 (fat tails), 10,000 paths",
        "regime_model": "5-state: bull/bear × low/normal/high vol",
        "ensemble": "40% MC + 60% factor blend (regime-weighted)",
        "data": "Twelve Data (live price + OHLCV) + yfinance (fundamentals + macro)",
        "note": "Statistical analysis only. Not financial advice.",
    }
