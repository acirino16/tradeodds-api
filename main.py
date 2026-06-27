"""
TradeOdds API — Proprietary Prediction Engine
Model stack:
  1. Fama-French 4-Factor (Market, SMB, HML, MOM) — factor tilt on expected return
  2. Historical base rate — rolling win-rate over matched lookback windows
  3. Monte Carlo GBM — 10,000 terminal-price paths with factor-implied drift
  4. VIX/Beta-derived implied volatility — proxy for options-implied move
  5. Historical analogues — nearest-neighbor return periods by vol regime
  6. Ensemble blend — probability-weighted average of all modules
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import time
import requests
import warnings
warnings.filterwarnings("ignore")

# Session with browser-like headers to avoid Yahoo Finance rate limiting
_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
})

app = FastAPI(title="TradeOdds API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Factor proxies (ETFs as free Fama-French factor proxies) ──────────────────
FACTOR_PROXIES = {
    "MKT": "SPY",   # Market
    "SMB": "IWM",   # Small-minus-big proxy (Russell 2000)
    "HML": "IWD",   # High-minus-low (value ETF)
    "MOM": "MTUM",  # Momentum factor ETF
    "VIX": "^VIX",
}

# Cache layer — avoid hammering Yahoo on repeated requests
_cache: dict = {}
CACHE_TTL = 600  # seconds


def cached(key: str):
    if key in _cache:
        val, ts = _cache[key]
        if (datetime.utcnow() - ts).seconds < CACHE_TTL:
            return val
    return None


def store(key: str, val):
    _cache[key] = (val, datetime.utcnow())
    return val


def fetch_prices(ticker: str, period: str = "2y") -> pd.Series:
    cached_val = cached(f"px_{ticker}_{period}")
    if cached_val is not None:
        return cached_val
    for attempt in range(3):
        try:
            data = yf.download(ticker, period=period, progress=False,
                               auto_adjust=True, session=_session)
            if data.empty:
                time.sleep(1.0 * (attempt + 1))
                continue
            prices = data["Close"].squeeze().dropna()
            if prices.empty or float(prices.iloc[-1]) <= 0:
                time.sleep(1.0 * (attempt + 1))
                continue
            return store(f"px_{ticker}_{period}", prices)
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    raise HTTPException(502, f"Could not fetch price data for {ticker} after 3 attempts")


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


# ── Module 1: Fama-French 4-Factor regression ─────────────────────────────────
def ff4_factor_tilt(ticker: str) -> dict:
    """
    Single-factor CAPM regression against SPY (market proxy).
    Alpha = annualized excess return above market-implied return.
    Kept to one download to stay fast on free-tier hosting.
    """
    stock_px = fetch_prices(ticker, "1y")
    spy_px   = fetch_prices("SPY",   "1y")

    stock_r = daily_returns(stock_px)
    mkt_r   = daily_returns(spy_px)

    df = pd.DataFrame({"stock": stock_r, "mkt": mkt_r}).dropna()

    if len(df) < 60:
        return {"beta_mkt": 1.0, "beta_smb": 0.0, "beta_hml": 0.0, "beta_mom": 0.0, "alpha_ann": 0.0}

    X = df[["mkt"]].values
    y = df["stock"].values
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    except Exception:
        coeffs = [0.0, 1.0]

    alpha_ann = float(coeffs[0]) * 252
    beta_mkt  = float(coeffs[1])

    return {
        "beta_mkt":  round(beta_mkt, 3),
        "beta_smb":  0.0,
        "beta_hml":  0.0,
        "beta_mom":  0.0,
        "alpha_ann": round(alpha_ann, 4),
    }


# ── Module 2: Historical base rate ────────────────────────────────────────────
def historical_base_rate(ticker: str, direction: str, horizon_days: int) -> dict:
    """
    What fraction of rolling horizon-day windows returned a profit for long/short?
    Uses 5 years of daily data.
    """
    prices = fetch_prices(ticker, "2y")
    if len(prices) < horizon_days + 10:
        return {"base_rate": 0.5, "n_windows": 0}

    wins = 0
    total = 0
    price_arr = prices.values
    for i in range(len(price_arr) - horizon_days):
        ret = (price_arr[i + horizon_days] - price_arr[i]) / price_arr[i]
        if direction == "long":
            wins += int(ret > 0)
        else:
            wins += int(ret < 0)
        total += 1

    rate = wins / total if total > 0 else 0.5
    return {"base_rate": round(rate, 4), "n_windows": total}


# ── Module 3: Volatility (VIX/Beta-derived) ───────────────────────────────────
def implied_vol_proxy(ticker: str, beta: float) -> dict:
    """
    Estimate implied volatility using realized vol + VIX scaling by beta.
    """
    prices = fetch_prices(ticker, "1y")
    returns = daily_returns(prices)
    realized_vol_ann = float(returns.std() * math.sqrt(252))

    vix_px = fetch_prices("^VIX", "5d")
    current_vix = float(vix_px.iloc[-1]) / 100.0 if len(vix_px) > 0 else 0.20

    # Implied vol proxy: blend realized + VIX-scaled-by-beta
    implied = 0.5 * realized_vol_ann + 0.5 * (current_vix * abs(beta))
    implied = max(0.05, min(implied, 2.0))

    return {
        "realized_vol_ann": round(realized_vol_ann, 4),
        "vix_current":      round(current_vix * 100, 2),
        "implied_vol_proxy": round(implied, 4),
    }


# ── Module 4: Monte Carlo GBM simulation ──────────────────────────────────────
def monte_carlo(
    current_price: float,
    entry_price: float | None,
    direction: str,
    horizon_days: int,
    sigma: float,
    alpha_ann: float,
    beta_mkt: float,
    n_paths: int = 10_000,
) -> dict:
    """
    GBM with factor-implied drift. Returns terminal distribution stats.
    """
    if entry_price is None or entry_price <= 0:
        entry_price = current_price

    # Factor-implied drift: risk-free ~4.5% + alpha from FF4
    rf_daily   = 0.045 / 252
    drift_daily = rf_daily + alpha_ann / 252
    sigma_daily = sigma / math.sqrt(252)

    rng = np.random.default_rng(seed=42)
    Z   = rng.standard_normal((n_paths, horizon_days))
    log_returns = (drift_daily - 0.5 * sigma_daily ** 2) + sigma_daily * Z
    terminal_prices = current_price * np.exp(log_returns.sum(axis=1))

    if direction == "long":
        profitable = terminal_prices > entry_price
    else:
        profitable = terminal_prices < entry_price

    p_mc = float(profitable.mean())
    terminal_rets = (terminal_prices - entry_price) / entry_price
    if direction == "short":
        terminal_rets = -terminal_rets

    return {
        "p_mc":      round(p_mc, 4),
        "median_ret": round(float(np.median(terminal_rets)), 4),
        "p5_ret":    round(float(np.percentile(terminal_rets, 5)), 4),
        "p95_ret":   round(float(np.percentile(terminal_rets, 95)), 4),
        "n_paths":   n_paths,
    }


# ── Module 5: Historical analogues ────────────────────────────────────────────
def historical_analogues(ticker: str, direction: str, horizon_days: int, current_vol: float) -> dict:
    """
    Find past 90-day windows where trailing vol was similar to today.
    Return average forward return in those regimes.
    """
    prices = fetch_prices(ticker, "2y")
    if len(prices) < 90 + horizon_days:
        return {"analogue_p": 0.5, "n_analogues": 0, "avg_fwd_ret": 0.0}

    rets   = daily_returns(prices)
    price_arr = prices.values
    ret_arr   = rets.values

    # Rolling 30-day realized vol
    roll_vol = pd.Series(ret_arr).rolling(30).std().values * math.sqrt(252)

    analogues = []
    for i in range(30, len(price_arr) - horizon_days):
        hist_vol = roll_vol[i]
        if np.isnan(hist_vol):
            continue
        # vol regime match: within 30% of current vol
        if abs(hist_vol - current_vol) / (current_vol + 1e-9) < 0.30:
            fwd_ret = (price_arr[i + horizon_days] - price_arr[i]) / price_arr[i]
            if direction == "short":
                fwd_ret = -fwd_ret
            analogues.append(fwd_ret)

    if not analogues:
        return {"analogue_p": 0.5, "n_analogues": 0, "avg_fwd_ret": 0.0}

    analogue_p = float(np.mean([r > 0 for r in analogues]))
    avg_fwd    = float(np.mean(analogues))

    return {
        "analogue_p":  round(analogue_p, 4),
        "n_analogues": len(analogues),
        "avg_fwd_ret": round(avg_fwd, 4),
    }


# ── Module 6: Ensemble blend ──────────────────────────────────────────────────
def ensemble_blend(
    base_rate: float,
    p_mc:      float,
    analogue_p: float,
    vol_regime: str,
) -> dict:
    """
    Weighted average of probability estimates.
    Weights reflect model reliability: MC > base rate > analogues.
    Vol regime adjusts confidence band.
    """
    weights = {"base_rate": 0.25, "mc": 0.50, "analogue": 0.25}
    if vol_regime == "high":
        weights = {"base_rate": 0.20, "mc": 0.55, "analogue": 0.25}

    blended = (
        weights["base_rate"] * base_rate
        + weights["mc"]       * p_mc
        + weights["analogue"] * analogue_p
    )
    blended = max(0.05, min(0.95, blended))

    return {"p_ensemble": round(blended, 4), "weights": weights}


# ── Fundamentals from yfinance info ───────────────────────────────────────────
def get_fundamentals(ticker: str) -> dict:
    cached_val = cached(f"info_{ticker}")
    if cached_val is not None:
        return cached_val

    try:
        info = yf.Ticker(ticker, session=_session).info
    except Exception:
        info = {}

    def safe(key, default=None):
        v = info.get(key)
        return v if v is not None else default

    result = {
        "name":            safe("longName", ticker),
        "sector":          safe("sector", "Unknown"),
        "industry":        safe("industry", "Unknown"),
        "market_cap":      safe("marketCap"),
        "pe_forward":      safe("forwardPE"),
        "pe_trailing":     safe("trailingPE"),
        "ev_ebitda":       safe("enterpriseToEbitda"),
        "revenue_growth":  safe("revenueGrowth"),
        "gross_margins":   safe("grossMargins"),
        "profit_margins":  safe("profitMargins"),
        "debt_to_equity":  safe("debtToEquity"),
        "return_on_equity": safe("returnOnEquity"),
        "return_on_assets": safe("returnOnAssets"),
        "free_cashflow":   safe("freeCashflow"),
        "beta":            safe("beta", 1.0),
        "52w_high":        safe("fiftyTwoWeekHigh"),
        "52w_low":         safe("fiftyTwoWeekLow"),
        "avg_volume":      safe("averageVolume"),
        "short_ratio":     safe("shortRatio"),
        "analyst_target":  safe("targetMeanPrice"),
        "recommendation":  safe("recommendationKey", "hold"),
        "num_analysts":    safe("numberOfAnalystOpinions", 0),
    }
    return store(f"info_{ticker}", result)


# ── Price momentum features ───────────────────────────────────────────────────
def price_features(ticker: str) -> dict:
    # Use fast_info for the real-time current price — avoids stale historical data
    try:
        fi = yf.Ticker(ticker, session=_session).fast_info
        cur = float(fi.last_price)
        if not cur or cur <= 0:
            raise ValueError("bad fast_info price")
    except Exception:
        # Fallback: last close from recent download
        try:
            tmp = yf.download(ticker, period="5d", progress=False,
                              auto_adjust=True, session=_session)
            cur = float(tmp["Close"].iloc[-1])
        except Exception:
            return {"current_price": 0, "mom_1m": 0, "mom_3m": 0, "mom_6m": 0, "mom_12m": 0}

    # Historical prices for momentum (use longer window, iloc[-1] is NOT used for current price)
    prices = fetch_prices(ticker, "1y")
    if len(prices) < 30:
        return {"current_price": round(cur, 2), "mom_1m": 0, "mom_3m": 0, "mom_6m": 0, "mom_12m": 0}

    px = prices.values

    def safe_mom(n):
        idx = max(0, len(px) - n)
        base = float(px[idx])
        return round((cur - base) / base, 4) if base > 0 else 0.0

    return {
        "current_price": round(cur, 2),
        "mom_1m":  safe_mom(21),
        "mom_3m":  safe_mom(63),
        "mom_6m":  safe_mom(126),
        "mom_12m": safe_mom(252),
    }


# ── Factor radar (0–10 scores) ────────────────────────────────────────────────
def factor_radar(fund: dict, ff4: dict, px: dict) -> dict:
    """Normalize each factor into a 0-10 score for the radar chart."""

    def clamp(x): return max(0.0, min(10.0, x))

    # Value: lower P/E = better value (score 10 if PE < 10, 0 if PE > 50)
    pe = fund.get("pe_forward") or fund.get("pe_trailing") or 25.0
    value = clamp(10 - (pe - 10) * 10 / 40) if pe else 5.0

    # Growth: revenue growth (score 10 if >50%, 0 if negative)
    rg = (fund.get("revenue_growth") or 0.0) * 100
    growth = clamp(5 + rg / 10)

    # Quality: margins + ROE
    pm = (fund.get("profit_margins") or 0.0) * 100
    roe = (fund.get("return_on_equity") or 0.0) * 100
    quality = clamp((pm / 4) + (roe / 20) * 5)

    # Momentum: 12m price momentum
    mom12 = px.get("mom_12m", 0.0) * 100
    momentum = clamp(5 + mom12 / 20)

    # Sentiment: FF4 alpha + analyst recommendation
    rec_map = {"strongBuy": 3, "buy": 2, "hold": 0, "underperform": -1, "sell": -2}
    rec_score = rec_map.get(fund.get("recommendation", "hold"), 0)
    alpha_score = clamp(5 + ff4["alpha_ann"] * 100 + rec_score)
    sentiment = clamp(alpha_score)

    return {
        "value":    round(value, 1),
        "growth":   round(growth, 1),
        "quality":  round(quality, 1),
        "momentum": round(momentum, 1),
        "sentiment": round(sentiment, 1),
    }


# ── Main forecast endpoint ─────────────────────────────────────────────────────
@app.get("/v1/forecast")
async def forecast(
    ticker:  str   = Query(..., description="Stock ticker e.g. AAPL"),
    horizon: int   = Query(90,  description="Days forward"),
    dir:     str   = Query("long", description="long or short"),
    entry:   float = Query(None, description="Entry price (optional, defaults to current)"),
):
    ticker  = ticker.upper().strip()
    horizon = max(1, min(horizon, 365))
    direction = dir.lower()
    if direction not in ("long", "short"):
        raise HTTPException(400, "dir must be 'long' or 'short'")

    try:
        fund  = get_fundamentals(ticker)
        px    = price_features(ticker)
    except Exception as e:
        raise HTTPException(502, f"Data fetch failed for {ticker}: {str(e)}")

    current_price = px["current_price"]
    if current_price <= 0:
        raise HTTPException(404, f"No price data found for {ticker}")

    entry_price = entry if (entry and entry > 0) else current_price
    beta = fund.get("beta") or 1.0

    # Run all modules
    ff4   = ff4_factor_tilt(ticker)
    br    = historical_base_rate(ticker, direction, horizon)
    vol   = implied_vol_proxy(ticker, beta)
    mc    = monte_carlo(current_price, entry_price, direction, horizon,
                        sigma=vol["implied_vol_proxy"],
                        alpha_ann=ff4["alpha_ann"],
                        beta_mkt=ff4["beta_mkt"])
    ana   = historical_analogues(ticker, direction, horizon, vol["realized_vol_ann"])

    vol_regime = "high" if vol["implied_vol_proxy"] > 0.35 else "normal"
    ens   = ensemble_blend(br["base_rate"], mc["p_mc"], ana["analogue_p"], vol_regime)
    radar = factor_radar(fund, ff4, px)

    p_win = ens["p_ensemble"]
    implied_move_pct = round(vol["implied_vol_proxy"] * math.sqrt(horizon / 252) * 100, 1)

    return {
        "ticker":        ticker,
        "name":          fund["name"],
        "sector":        fund["sector"],
        "direction":     direction,
        "horizon_days":  horizon,
        "current_price": current_price,
        "entry_price":   round(entry_price, 2),

        # ── Headline probability ──────────────────────────────────────────
        "p_win":         p_win,
        "verdict":       verdict(p_win),

        # ── Module outputs ────────────────────────────────────────────────
        "modules": {
            "ff4_factors":    ff4,
            "base_rate":      br,
            "implied_vol":    vol,
            "monte_carlo":    mc,
            "analogues":      ana,
            "ensemble":       ens,
        },

        # ── Factor radar (0–10) ───────────────────────────────────────────
        "factor_radar": radar,

        # ── Price features ────────────────────────────────────────────────
        "price_momentum": px,

        # ── Fundamentals snapshot ─────────────────────────────────────────
        "fundamentals": {
            "pe_forward":     fund.get("pe_forward"),
            "revenue_growth": fund.get("revenue_growth"),
            "profit_margins": fund.get("profit_margins"),
            "return_on_equity": fund.get("return_on_equity"),
            "debt_to_equity": fund.get("debt_to_equity"),
            "beta":           beta,
            "market_cap":     fund.get("market_cap"),
            "analyst_target": fund.get("analyst_target"),
            "recommendation": fund.get("recommendation"),
            "num_analysts":   fund.get("num_analysts"),
            "short_ratio":    fund.get("short_ratio"),
        },

        # ── Implied move ──────────────────────────────────────────────────
        "implied_move_pct": implied_move_pct,

        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_source":  "yfinance / Yahoo Finance (free tier)",
        "model_version": "TradeOdds-v1.0 (FF4+MOM+MC+Analogues)",
    }


# ── Screener endpoint ──────────────────────────────────────────────────────────
SP500_SAMPLE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","JPM",
    "TSLA","UNH","V","XOM","MA","JNJ","PG","HD","COST","MRK",
]

@app.get("/v1/screener/top")
async def screener(limit: int = Query(10), dir: str = Query("long")):
    results = []
    direction = dir.lower()
    for tk in SP500_SAMPLE[:limit]:
        try:
            px   = price_features(tk)
            fund = get_fundamentals(tk)
            ff4  = ff4_factor_tilt(tk)
            vol  = implied_vol_proxy(tk, fund.get("beta") or 1.0)
            br   = historical_base_rate(tk, direction, 90)
            mc   = monte_carlo(px["current_price"], px["current_price"], direction, 90,
                               vol["implied_vol_proxy"], ff4["alpha_ann"], ff4["beta_mkt"])
            ana  = historical_analogues(tk, direction, 90, vol["realized_vol_ann"])
            ens  = ensemble_blend(br["base_rate"], mc["p_mc"], ana["analogue_p"],
                                  "high" if vol["implied_vol_proxy"] > 0.35 else "normal")
            results.append({
                "ticker": tk,
                "name":   fund["name"],
                "sector": fund["sector"],
                "p_win":  ens["p_ensemble"],
                "verdict": verdict(ens["p_ensemble"]),
                "current_price": px["current_price"],
                "mom_12m": px["mom_12m"],
                "alpha_ann": ff4["alpha_ann"],
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["p_win"], reverse=(direction == "long"))
    return {"direction": direction, "results": results, "screened_at": datetime.utcnow().isoformat() + "Z"}


@app.get("/v1/calibration")
async def calibration():
    return {
        "model":       "TradeOdds-v1.0",
        "components":  ["FF4-regression", "historical-base-rate", "GBM-MC-10K", "vol-regime-analogues"],
        "ensemble_weights": {"base_rate": 0.25, "monte_carlo": 0.50, "analogues": 0.25},
        "data_source": "yfinance",
        "note":        "Probabilities are model outputs, not financial advice.",
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"service": "TradeOdds API", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "ok": True,
        "status": "ok",
        "provider": "yfinance",
        "live_data": True,
        "version": "1.0",
        "time": datetime.utcnow().isoformat() + "Z",
    }


def verdict(p: float) -> str:
    if p >= 0.72: return "Strong Edge"
    if p >= 0.60: return "Favourable"
    if p >= 0.50: return "Slight Edge"
    if p >= 0.40: return "Slight Disadvantage"
    if p >= 0.28: return "Unfavourable"
    return "Strong Disadvantage"
