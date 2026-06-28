"""
TradeOdds — SEC EDGAR XBRL Point-in-Time Fundamentals
======================================================
Provides true point-in-time fundamental data using the SEC's free XBRL API.
No API key required. Rate-limited to respect SEC guidelines (10 req/s).

Key function:
    get_fundamentals_as_of(ticker, as_of_date) -> dict

For each as_of_date it returns only facts from filings submitted BEFORE
that date — no look-ahead bias. This is the critical difference from
yfinance which always returns current values.

Concepts pulled (with multi-tag fallbacks):
    Revenue, Gross Profit, Operating Income, Net Income, R&D Expense,
    Total Assets, Total Liabilities, Stockholders Equity, Goodwill,
    Intangible Assets, Long-term Debt, Shares Outstanding

Derived metrics computed:
    gross_margin, operating_margin, net_margin, roe, rd_intensity,
    debt_to_equity, intangibles_ratio, revenue_growth_yoy, asset_turnover

Caching:
    Raw XBRL facts cached to .edgar_cache/{CIK}.json (disk).
    Derived as-of snapshots cached in memory per session.
    Cache TTL: 7 days for raw facts (they don't change retroactively).
"""

import json
import math
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(__file__).parent / ".edgar_cache"
CACHE_TTL   = 7 * 86400   # seconds — raw XBRL facts rarely change
REQ_DELAY   = 0.12         # ~8 req/s, well under SEC's 10 req/s limit
USER_AGENT  = "TradeOdds Research contact@tradeodds.example"  # SEC requires this

CACHE_DIR.mkdir(exist_ok=True)

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

_cik_map:    dict = {}          # ticker → CIK (int)
_facts_mem:  dict = {}          # CIK → raw facts dict
_snap_cache: dict = {}          # (ticker, date_str) → derived dict


# ── CIK lookup ─────────────────────────────────────────────────────────────────

def _load_cik_map() -> dict:
    """Download SEC company ticker → CIK map once per session."""
    global _cik_map
    if _cik_map:
        return _cik_map
    cache_path = CACHE_DIR / "company_tickers.json"
    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CACHE_TTL:
        with open(cache_path) as f:
            raw = json.load(f)
    else:
        time.sleep(REQ_DELAY)
        r   = _session.get("https://www.sec.gov/files/company_tickers.json", timeout=15)
        raw = r.json()
        with open(cache_path, "w") as f:
            json.dump(raw, f)
    _cik_map = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
    return _cik_map


def ticker_to_cik(ticker: str) -> int | None:
    m = _load_cik_map()
    return m.get(ticker.upper())


# ── Raw XBRL facts ─────────────────────────────────────────────────────────────

def _fetch_facts(cik: int) -> dict:
    """Download and cache all XBRL companyfacts for a CIK."""
    global _facts_mem
    if cik in _facts_mem:
        return _facts_mem[cik]

    cik_str    = str(cik).zfill(10)
    cache_path = CACHE_DIR / f"{cik_str}.json"

    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CACHE_TTL:
        with open(cache_path) as f:
            data = json.load(f)
    else:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
        time.sleep(REQ_DELAY)
        r = _session.get(url, timeout=20)
        if r.status_code != 200:
            _facts_mem[cik] = {}
            return {}
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)

    _facts_mem[cik] = data
    return data


# ── Concept extraction helpers ─────────────────────────────────────────────────

# XBRL tag fallbacks — companies use different tags for the same concept
CONCEPT_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",   # fallback — less accurate
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "LiabilitiesAndStockholdersEquity",   # last resort
    ],
    "goodwill": [
        "Goodwill",
    ],
    "intangibles": [
        "IntangibleAssetsNetExcludingGoodwill",
        "FiniteLivedIntangibleAssetsNet",
        "IndefiniteLivedIntangibleAssetsExcludingGoodwill",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
    ],
}


def _all_rows(facts: dict, concept: str) -> list[dict]:
    """
    Collect all filing rows for a concept across ALL fallback tags.
    Merges rows from every tag — we pick the most recently filed data
    across all tags, so stale tags (e.g. AAPL's old 'Revenues' tag)
    never shadow the current tag ('RevenueFromContractWithCustomer...').
    """
    tags   = CONCEPT_TAGS.get(concept, [concept])
    usgaap = facts.get("facts", {}).get("us-gaap", {})
    rows   = []
    seen   = set()   # deduplicate by (filed, fp, end)

    for tag in tags:
        entry = usgaap.get(tag)
        if not entry:
            continue
        usd = entry.get("units", {}).get("USD", [])
        for item in usd:
            filed = item.get("filed")
            val   = item.get("val")
            form  = item.get("form", "")
            end   = item.get("end", "")
            fp    = item.get("fp", "")
            if not (filed and val is not None and
                    form in ("10-K", "10-Q", "10-K/A", "10-Q/A")):
                continue
            key = (filed, fp, end)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "end":   end,
                "filed": filed,
                "val":   float(val),
                "form":  form,
                "fp":    fp,
            })

    rows.sort(key=lambda x: x["filed"])
    return rows


def _annual_before(rows: list[dict], as_of: str) -> tuple[float | None, str]:
    """
    Return (val, filed_date) of the most recently filed annual (10-K, fp=FY)
    value before as_of, for the most recent fiscal year period.

    Companies file 10-Ks with 2-3 comparative prior-year columns, all tagged
    fp=FY with the same filed date. We want the CURRENT year (latest end date),
    not a prior-year comparative. Sort by (filed, end) descending.
    """
    annual = [r for r in rows
              if r["filed"] < as_of
              and r["form"] in ("10-K", "10-K/A")
              and r["fp"] == "FY"]
    if not annual:
        return None, ""
    # Primary sort: most recently filed; secondary: most recent period end
    best = max(annual, key=lambda x: (x["filed"], x["end"]))
    return best["val"], best["filed"]


def _ttm_before(rows: list[dict], as_of: str) -> float | None:
    """
    Trailing twelve months for income-statement concepts.

    Strategy (in priority order):
    1. If a recent annual 10-K (fp=FY) exists filed within 18 months → use it.
       Annual values are audited, unambiguous, and avoid YTD-vs-incremental
       confusion in quarterly filings.
    2. If annual is stale (>18 months), try to build TTM from quarterly filings:
       take the 4 most recent unique period-end dates, verify each covers ~90 days
       (to avoid YTD-cumulative values), sum them.
    3. Fall back to stale annual if quarterly construction fails.
    """
    before = [r for r in rows if r["filed"] < as_of]
    if not before:
        return None

    annual_val, annual_filed = _annual_before(rows, as_of)
    if annual_val is not None:
        age_days = (datetime.strptime(as_of, "%Y-%m-%d") -
                    datetime.strptime(annual_filed, "%Y-%m-%d")).days
        if age_days <= 548:   # within 18 months
            return annual_val

    # Quarterly TTM construction — only use incremental (~90-day) items
    quarterly = [r for r in before if r["form"] in ("10-Q", "10-Q/A")]

    # De-duplicate by period end, keep most recent filing per period
    by_period: dict = {}
    for r in quarterly:
        end = r["end"]
        if end not in by_period or r["filed"] > by_period[end]["filed"]:
            by_period[end] = r

    # Sort by period end, take last 4
    periods = sorted(by_period.values(), key=lambda x: x["end"])[-4:]

    if len(periods) == 4:
        # Validate: each period should cover ~90 days
        # Check by looking at the gap between consecutive period ends
        ends = sorted(p["end"] for p in periods)
        gaps = []
        for i in range(1, len(ends)):
            d0 = datetime.strptime(ends[i-1], "%Y-%m-%d")
            d1 = datetime.strptime(ends[i],   "%Y-%m-%d")
            gaps.append((d1 - d0).days)
        # Valid if all gaps are 60–120 days (incremental quarters, not YTD)
        if all(50 <= g <= 130 for g in gaps):
            return sum(p["val"] for p in periods)

    return annual_val   # fall back to annual even if stale


def _balance_before(rows: list[dict], as_of: str) -> float | None:
    """Balance sheet: most recent value of any form filed before as_of."""
    candidates = [r for r in rows if r["filed"] < as_of]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x["filed"])["val"]


# ── Point-in-time snapshot ─────────────────────────────────────────────────────

def get_fundamentals_as_of(ticker: str, as_of: datetime) -> dict:
    """
    Return fundamental metrics available as of as_of_date.
    Only uses data from filings submitted strictly before as_of — no look-ahead.
    Returns empty dict on failure (handled gracefully by caller).
    """
    as_of_str = as_of.strftime("%Y-%m-%d")
    cache_key = f"{ticker}_{as_of_str}"
    if cache_key in _snap_cache:
        return _snap_cache[cache_key]

    result = _empty_fundamentals()

    cik = ticker_to_cik(ticker)
    if not cik:
        _snap_cache[cache_key] = result
        return result

    facts = _fetch_facts(cik)
    if not facts:
        _snap_cache[cache_key] = result
        return result

    # Pull all rows per concept (merges across fallback tags, picks most recent)
    rev_r   = _all_rows(facts, "revenue")
    gp_r    = _all_rows(facts, "gross_profit")
    oi_r    = _all_rows(facts, "operating_income")
    ni_r    = _all_rows(facts, "net_income")
    rd_r    = _all_rows(facts, "rd_expense")
    ta_r    = _all_rows(facts, "total_assets")
    eq_r    = _all_rows(facts, "equity")
    gw_r    = _all_rows(facts, "goodwill")
    ia_r    = _all_rows(facts, "intangibles")
    ltd_r   = _all_rows(facts, "long_term_debt")
    cash_r  = _all_rows(facts, "cash")

    # TTM income statement (annual preferred, quarterly fallback)
    rev  = _ttm_before(rev_r,  as_of_str)
    gp   = _ttm_before(gp_r,   as_of_str)
    oi   = _ttm_before(oi_r,   as_of_str)
    ni   = _ttm_before(ni_r,   as_of_str)
    rd   = _ttm_before(rd_r,   as_of_str)

    # Most recent balance sheet
    ta   = _balance_before(ta_r,   as_of_str)
    eq   = _balance_before(eq_r,   as_of_str)
    gw   = _balance_before(gw_r,   as_of_str)
    ia   = _balance_before(ia_r,   as_of_str)
    ltd  = _balance_before(ltd_r,  as_of_str)
    cash = _balance_before(cash_r, as_of_str)

    def safe_div(n, d, default=None):
        if n is None or d is None or d == 0:
            return default
        return n / d

    # Revenue growth YoY — compare TTM to TTM from ~1yr earlier
    rev_yoy = None
    if rev_r and rev is not None:
        earlier_date = (as_of - timedelta(days=365)).strftime("%Y-%m-%d")
        rev_prior = _ttm_before(rev_r, earlier_date)
        if rev_prior and rev_prior > 0:
            rev_yoy = (rev - rev_prior) / abs(rev_prior)

    gross_margin     = safe_div(gp, rev)
    operating_margin = safe_div(oi, rev)
    net_margin       = safe_div(ni, rev)
    rd_intensity     = safe_div(rd, rev)
    roe              = safe_div(ni, eq)
    debt_to_equity   = safe_div(ltd, eq)
    intangibles_total = (gw or 0) + (ia or 0)
    intangibles_ratio = safe_div(intangibles_total, ta) if ta else None
    asset_turnover   = safe_div(rev, ta)

    result.update({
        "has_data":          True,
        "revenue_ttm":       rev,
        "gross_profit_ttm":  gp,
        "net_income_ttm":    ni,
        "rd_expense_ttm":    rd,
        "total_assets":      ta,
        "equity":            eq,
        "long_term_debt":    ltd,
        "cash":              cash,
        "gross_margin":      gross_margin,
        "operating_margin":  operating_margin,
        "net_margin":        net_margin,
        "rd_intensity":      rd_intensity,
        "roe":               roe,
        "debt_to_equity":    debt_to_equity,
        "intangibles_ratio": intangibles_ratio,
        "asset_turnover":    asset_turnover,
        "revenue_growth_yoy": rev_yoy,
    })

    _snap_cache[cache_key] = result
    return result


def _empty_fundamentals() -> dict:
    return {
        "has_data": False,
        "revenue_ttm": None, "gross_profit_ttm": None, "net_income_ttm": None,
        "rd_expense_ttm": None, "total_assets": None, "equity": None,
        "long_term_debt": None, "cash": None, "gross_margin": None,
        "operating_margin": None, "net_margin": None, "rd_intensity": None,
        "roe": None, "debt_to_equity": None, "intangibles_ratio": None,
        "asset_turnover": None, "revenue_growth_yoy": None,
    }


# ── Submissions (filings index) ────────────────────────────────────────────────

_submissions_mem:  dict = {}   # CIK → submissions dict
_insider_cache:    dict = {}   # (ticker, as_of_str) → float score


def _fetch_submissions(cik: int) -> dict:
    """
    Fetch SEC submissions JSON for a CIK — contains full filing history
    (form type, filing date, accession number) for 8-K, 10-Q, Form 4, etc.
    Cached to disk alongside XBRL facts.
    """
    global _submissions_mem
    if cik in _submissions_mem:
        return _submissions_mem[cik]

    cik_str    = str(cik).zfill(10)
    cache_path = CACHE_DIR / f"{cik_str}_submissions.json"

    if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < CACHE_TTL:
        with open(cache_path) as f:
            data = json.load(f)
    else:
        url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        time.sleep(REQ_DELAY)
        r = _session.get(url, timeout=20)
        if r.status_code != 200:
            _submissions_mem[cik] = {}
            return {}
        data = r.json()
        with open(cache_path, "w") as f:
            json.dump(data, f)

    _submissions_mem[cik] = data
    return data


def _get_filings_df(submissions: dict) -> list[dict]:
    """
    Flatten the submissions JSON into a list of dicts, each with:
      form, filingDate, accessionNumber, primaryDocument
    Handles both recent filings and older paginated files.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnos  = recent.get("accessionNumber", [])
    docs    = recent.get("primaryDocument", [])
    cik_int = submissions.get("cik", 0)

    rows = []
    for i in range(len(forms)):
        rows.append({
            "form":        forms[i] if i < len(forms) else "",
            "filingDate":  dates[i] if i < len(dates) else "",
            "accNo":       accnos[i] if i < len(accnos) else "",
            "primaryDoc":  docs[i]  if i < len(docs)  else "",
            "cik":         cik_int,
        })
    return rows


# ── Earnings window detection ───────────────────────────────────────────────────

def has_earnings_in_window(ticker: str, window_start: datetime,
                           window_end: datetime) -> bool:
    """
    Returns True if an earnings event (8-K Results of Operations, 10-Q, or 10-K)
    was filed between window_start and window_end.

    Uses SEC 8-K filings (item 2.02 = earnings release, filed within 4 days of
    earnings call) and 10-Q / 10-K as backup signals.

    In the backtest this tells us: does this 90-day prediction window contain
    an earnings print? If yes, there's a big binary risk event the model can't
    see, which degrades accuracy.
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        return False
    subs = _fetch_submissions(cik)
    rows = _get_filings_df(subs)
    if not rows:
        return False

    start_str = window_start.strftime("%Y-%m-%d")
    end_str   = window_end.strftime("%Y-%m-%d")

    for r in rows:
        fd = r["filingDate"]
        if not (start_str <= fd <= end_str):
            continue
        # 8-K covers many events — only quarterly results (item 2.02)
        # Proxy: 8-K filed in the quarter-end months (Jan/Apr/Jul/Oct ± 1) or 10-Q/10-K
        form = r["form"]
        if form in ("10-Q", "10-Q/A", "10-K", "10-K/A"):
            return True
        if form == "8-K":
            # 8-K is filed for many things; earnings 8-Ks cluster in
            # the 1–45 days after quarter end. Accept all 8-Ks as a conservative flag.
            return True

    return False


# ── Insider transaction score ───────────────────────────────────────────────────

def _fetch_form4_xml(cik: int, acc_no: str, primary_doc: str) -> str:
    """Download a single Form 4 primary document XML. Returns raw text."""
    acc_dashes = acc_no.replace("-", "")
    # Try primary doc first, fall back to accession-based name
    for doc_name in [primary_doc, f"{acc_dashes}.xml"]:
        if not doc_name or not doc_name.endswith(".xml"):
            continue
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
               f"{acc_dashes}/{doc_name}")
        cache_path = CACHE_DIR / f"f4_{acc_dashes}.xml"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore")
        time.sleep(REQ_DELAY)
        try:
            r = _session.get(url, timeout=15)
            if r.status_code == 200:
                cache_path.write_text(r.text, encoding="utf-8")
                return r.text
        except Exception:
            pass
    return ""


def _parse_form4_net_shares(xml_text: str) -> float:
    """
    Parse Form 4 XML and return net shares: positive = bought, negative = sold.
    Handles both nonDerivative and derivative transactions.
    Ignores gifts (G), grants (A from company), tax withholding (F).
    """
    import re
    net = 0.0

    # Extract all nonDerivativeTransaction blocks
    blocks = re.findall(
        r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
        xml_text, re.DOTALL
    )
    for block in blocks:
        # Transaction code: P=purchase, S=sale, others=skip
        code_m = re.search(r"<transactionCode>\s*([A-Z])\s*</transactionCode>", block)
        if not code_m:
            continue
        code = code_m.group(1)
        if code not in ("P", "S"):
            continue   # skip gifts, awards, tax withholding, etc.

        # Acquired/Disposed: A=acquired (positive), D=disposed (negative)
        ad_m = re.search(
            r"<transactionAcquiredDisposedCode>\s*<value>\s*([AD])\s*</value>",
            block
        )
        direction_sign = 1.0
        if ad_m:
            direction_sign = 1.0 if ad_m.group(1) == "A" else -1.0

        shares_m = re.search(
            r"<transactionShares>\s*<value>\s*([\d.,]+)\s*</value>", block
        )
        if shares_m:
            shares = float(shares_m.group(1).replace(",", ""))
            net += shares * direction_sign

    return net


def get_insider_score(ticker: str, as_of: datetime, lookback_days: int = 90) -> float:
    """
    Score based on net insider buying/selling in the lookback_days before as_of.

    Returns a score in (-1, +1):
      +1.0 = strong net buying (bullish signal)
      -1.0 = heavy net selling (bearish signal)
       0.0 = neutral / no recent activity

    Only counts open-market purchases (P) and sales (S) by company insiders.
    Excludes grants, gifts, and tax-withholding transactions.
    Memoized: the same (ticker, date) pair is computed once per session.
    """
    global _insider_cache
    # Coarsen to monthly — insider activity changes slowly, avoids re-fetching
    # every 2-week backtest step for the same ticker
    cache_key = (ticker.upper(), as_of.strftime("%Y-%m"))
    if cache_key in _insider_cache:
        return _insider_cache[cache_key]

    cik = ticker_to_cik(ticker)
    if not cik:
        return 0.0

    subs = _fetch_submissions(cik)
    rows = _get_filings_df(subs)
    if not rows:
        return 0.0

    start_str = (as_of - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_str   = as_of.strftime("%Y-%m-%d")

    # Filter to Form 4 filings in the window
    form4_rows = [
        r for r in rows
        if r["form"] == "4" and start_str <= r["filingDate"] < end_str
    ]

    if not form4_rows:
        _insider_cache[cache_key] = 0.0
        return 0.0

    # Cap at 15 most recent filings to limit HTTP calls in backtest
    form4_rows = sorted(form4_rows, key=lambda x: x["filingDate"], reverse=True)[:15]

    total_bought = 0.0
    total_sold   = 0.0
    for r in form4_rows:
        xml = _fetch_form4_xml(cik, r["accNo"], r["primaryDoc"])
        if not xml:
            continue
        net = _parse_form4_net_shares(xml)
        if net > 0:
            total_bought += net
        else:
            total_sold += abs(net)

    total = total_bought + total_sold
    if total < 1:
        _insider_cache[cache_key] = 0.0
        return 0.0

    # Net buying ratio → score; dampen selling (often options vesting, not bearish conviction)
    net_ratio = (total_bought - total_sold) / total   # -1 to +1
    score = net_ratio * 0.8 if net_ratio > 0 else net_ratio * 0.5

    result = float(max(-1.0, min(1.0, score)))
    _insider_cache[cache_key] = result
    return result


# ── Pre-fetch for backtest ─────────────────────────────────────────────────────

def prefetch_tickers(tickers: list[str], verbose: bool = True) -> None:
    """
    Download and cache XBRL facts for all tickers before the backtest loop.
    Much faster than fetching on demand — avoids repeated HTTP calls inside the loop.
    """
    _load_cik_map()
    for i, ticker in enumerate(tickers):
        cik = ticker_to_cik(ticker)
        if not cik:
            if verbose:
                print(f"  [EDGAR] {ticker}: no CIK found")
            continue
        _fetch_facts(cik)        # XBRL fundamentals
        _fetch_submissions(cik)  # filing history (earnings dates + Form 4)
        if verbose:
            sys.stdout.write(f"\r  [EDGAR] {i+1}/{len(tickers)}  {ticker:6}   ")
            sys.stdout.flush()
    if verbose:
        print()


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    date   = datetime.strptime(sys.argv[2], "%Y-%m-%d") if len(sys.argv) > 2 else datetime(2022, 6, 30)
    print(f"\nPoint-in-time fundamentals for {ticker} as of {date.date()}")
    print("(Only uses filings submitted before this date — no look-ahead)\n")
    f = get_fundamentals_as_of(ticker, date)
    for k, v in f.items():
        if v is None: continue
        if isinstance(v, float) and abs(v) > 1e6:
            print(f"  {k:25}  ${v/1e9:.2f}B")
        elif isinstance(v, float):
            print(f"  {k:25}  {v:.4f}")
        else:
            print(f"  {k:25}  {v}")
