import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import yfinance as yf
import requests
# pyrefly: ignore [missing-import]
import streamlit as st
import time
import random
from concurrent.futures import ThreadPoolExecutor
from src.logger_config import get_logger

logger = get_logger(__name__)

# Fallback dictionary of exchange rates to USD (1 unit of currency = X USD)
# Keeps the system functional even if live FX API fails or hits rate limits.
FX_FALLBACK_RATES = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.28,
    "JPY": 0.0065,
    "CAD": 0.73,
    "AUD": 0.66,
    "CHF": 1.12,
    "SEK": 0.095,
    "DKK": 0.14,
    "NOK": 0.093,
    "CNY": 0.14,
    "HKD": 0.13,
    "INR": 0.012,
    "KRW": 0.00073,
    "TWD": 0.031,
    "PLN": 0.25,
    "CZK": 0.043
}

# Mapping of reporting currencies to regional risk-free rate proxy yields
REGIONAL_RF_RATES = {
    "USD": 0.045,  # US 10-Year Treasury Yield proxy
    "EUR": 0.025,  # 10-Year German Bund Yield proxy
    "GBP": 0.040,  # UK 10-Year Gilt Yield proxy
    "JPY": 0.010,  # Japan 10-Year JGB Yield proxy
    "CHF": 0.010,  # Swiss 10-Year Bond Yield proxy
    "CAD": 0.035,  # Canada 10-Year Bond Yield proxy
    "AUD": 0.040,  # Australia 10-Year Bond Yield proxy
    "SEK": 0.023,  # Sweden 10-Year Bond Yield proxy
    "DKK": 0.024,  # Denmark 10-Year Bond Yield proxy
    "NOK": 0.035,  # Norway 10-Year Bond Yield proxy
    "CNY": 0.023,  # China 10-Year Bond Yield proxy
    "HKD": 0.035,  # Hong Kong 10-Year Bond Yield proxy
    "INR": 0.068,  # India 10-Year Bond Yield proxy
    "KRW": 0.032,  # South Korea 10-Year Bond Yield proxy
    "TWD": 0.016,  # Taiwan 10-Year Bond Yield proxy
    "PLN": 0.055,  # Poland 10-Year Bond Yield proxy
    "CZK": 0.041   # Czech Republic 10-Year Bond Yield proxy
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "CHF": "CHF ",
    "CAD": "CA$",
    "AUD": "A$",
    "SEK": "kr ",
    "DKK": "kr ",
    "NOK": "kr ",
    "HKD": "HK$",
    "INR": "₹",
    "KRW": "₩",
    "TWD": "NT$",
    "PLN": "zł ",
    "CZK": "Kč "
}

@st.cache_data(ttl=3600)
def get_fx_rate(currency: str) -> float:
    """
    Retrieves the live exchange rate to convert the specified currency into USD (1 currency unit = X USD).
    Uses Yahoo Finance live FX tickers (e.g., EURUSD=X). Falls back to FX_FALLBACK_RATES if API fails.
    """
    if not currency or currency.upper() == "USD":
        return 1.0
        
    curr = currency.upper().strip()
    ticker_pair = f"{curr}USD=X"
    logger.info(f"Starting FX rate fetch for currency pair: {ticker_pair}")
    start_time = time.perf_counter()
    try:
        fx_stock = yf.Ticker(ticker_pair)
        # Try fetching recent price history first
        hist = fx_stock.history(period="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            rate = float(hist["Close"].iloc[-1])
            if rate > 0:
                elapsed = time.perf_counter() - start_time
                logger.info(f"Successfully fetched FX rate for {ticker_pair}: {rate} (took {elapsed:.4f}s)")
                return rate
        else:
            logger.warning(f"FX price history is empty or missing 'Close' for {ticker_pair}. Attempting info fallback.")
        # Fall back to info dictionary if history is empty
        info = fx_stock.info
        if info and "regularMarketPrice" in info and info["regularMarketPrice"]:
            rate = float(info["regularMarketPrice"])
            if rate > 0:
                elapsed = time.perf_counter() - start_time
                logger.info(f"Successfully fetched FX rate from info for {ticker_pair}: {rate} (took {elapsed:.4f}s)")
                return rate
        logger.warning(f"Missing critical financial key 'regularMarketPrice' in info for {ticker_pair}.")
    except Exception as e:
        logger.error(f"FX live translation failed for pair {ticker_pair}: {e}. Using fallback rate.", exc_info=True)
        print(f"FX live translation failed for pair {ticker_pair}: {e}. Using fallback rate.")
        
    fallback_rate = FX_FALLBACK_RATES.get(curr, 1.0)
    logger.warning(f"Falling back to default FX exchange rate for {curr}: {fallback_rate}")
    return fallback_rate

def get_currency_symbol(currency: str) -> str:
    """Returns the visual symbol ($, €, £, ¥, etc.) for a given currency code."""
    if not currency:
        return "$"
    return CURRENCY_SYMBOLS.get(currency.upper().strip(), f"{currency} ")

def get_regional_rf_rate(currency: str) -> float:
    """Returns the macroeconomic Risk-Free Rate proxy for the given currency denomination."""
    if not currency:
        logger.warning("Currency not provided for RF rate; falling back to default USD risk-free rate (0.045).")
        return 0.045
    curr = currency.upper().strip()
    if curr not in REGIONAL_RF_RATES:
        logger.warning(f"No regional RF rate mapped for currency '{curr}'; falling back to default 0.045.")
    return REGIONAL_RF_RATES.get(curr, 0.045)


def _get_streamlit_ctx():
    """Safely retrieves Streamlit script run context across different Streamlit versions."""
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.ctx import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.scriptrunner import get_script_run_ctx
        return get_script_run_ctx()
    except Exception:
        return None


def _add_streamlit_ctx(ctx):
    """Safely attaches Streamlit script run context across different Streamlit versions."""
    if ctx is None:
        return
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.ctx import add_script_run_ctx
        add_script_run_ctx(ctx=ctx)
        return
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.scriptrunner_utils.script_run_context import add_script_run_ctx
        add_script_run_ctx(ctx=ctx)
        return
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(ctx=ctx)
        return
    except Exception:
        pass
    try:
        # pyrefly: ignore [missing-import]
        from streamlit.scriptrunner import add_script_run_ctx
        add_script_run_ctx(ctx=ctx)
        return
    except Exception:
        pass


def _get_ticker_info_with_retry(t: str, max_retries: int = 3) -> dict:
    """
    Fetches ticker info from Yahoo Finance with automatic retries and exponential backoff
    to handle Cloudflare/Yahoo rate limits and curl timeouts (e.g. curl: (28) Operation timed out).
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                sleep_time = (2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
                logger.info(f"Retrying fetch for {t} (Attempt {attempt + 1}/{max_retries}) after {sleep_time:.2f}s backoff...")
                time.sleep(sleep_time)
            elif max_retries > 1:
                # Add tiny jitter to spread out simultaneous thread pool start times
                time.sleep(random.uniform(0.05, 0.4))
            
            session = None
            if attempt > 0:
                try:
                    # pyrefly: ignore [missing-import]
                    from curl_cffi import requests as curl_requests
                    session = curl_requests.Session(impersonate="chrome")
                except ImportError:
                    try:
                        session = requests.Session()
                        session.headers.update({
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        })
                    except Exception:
                        pass

            stock = yf.Ticker(t, session=session) if session else yf.Ticker(t)
            info = stock.info
            if info and ('symbol' in info or len(info) > 5):
                return info
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for ticker {t}: {e}")
            if attempt == max_retries - 1:
                raise last_exception
            
    return {}


def _fetch_single_ticker_info(t: str, ctx=None) -> tuple[dict | None, list[str], list[str]]:
    """
    Helper function to fetch metrics for a single ticker.
    Returns: (row_dict or None, list of warnings, list of errors)
    """
    _add_streamlit_ctx(ctx)

    # Introduce a random delay between 0.2 and 0.8 seconds to stagger API hits and prevent HTTP 429 errors
    time.sleep(random.uniform(0.2, 0.8))

    warnings = []
    errors = []
    logger.info(f"Starting data fetch operation for ticker: {t}")
    start_time = time.perf_counter()
    try:
        info = _get_ticker_info_with_retry(t, max_retries=3)
        
        if not info or ('symbol' not in info and len(info) <= 5):
            msg = f"Yahoo Finance nevrátilo kompletní data pro ticker: {t}"
            logger.warning(f"Incomplete info returned from Yahoo Finance for ticker {t} (missing critical keys).")
            warnings.append(msg)
            return None, warnings, errors
        
        currency = info.get("currency", "USD")
        if not currency:
            logger.warning(f"Missing currency key for ticker {t}. Falling back to default USD.")
            currency = "USD"
        currency = currency.upper().strip()
        fx_rate = get_fx_rate(currency)
        
        market_cap = info.get("marketCap", None)
        fcf = info.get("freeCashflow", None)
        if fcf is None or pd.isna(fcf):
            ocf = info.get("operatingCashflow", None)
            capex = info.get("capitalExpenditures", None)
            if ocf is not None and not pd.isna(ocf) and capex is not None and not pd.isna(capex):
                fcf = ocf - abs(capex)
            else:
                logger.warning(f"Missing critical Capital Expenditures (CapEx) or operating cash flow for {t}. Setting FCF to np.nan.")
                fcf = np.nan
        if market_cap is None:
            logger.warning(f"Missing critical financial key 'marketCap' for {t}.")
        total_debt = info.get("totalDebt", None)
        total_cash = info.get("totalCash", None)
        if total_debt is None or total_cash is None:
            logger.warning(f"Missing balance sheet keys (totalDebt or totalCash) for {t}.")
        
        market_cap_usd = market_cap * fx_rate if market_cap is not None else None
        fcf_usd = fcf * fx_rate if (fcf is not None and not pd.isna(fcf)) else np.nan
        total_debt_usd = total_debt * fx_rate if total_debt is not None else None
        total_cash_usd = total_cash * fx_rate if total_cash is not None else None
        
        # Safely check for dividend existence
        div_yield = info.get("dividendYield", None)
        div_rate = info.get("dividendRate", None)
        has_div_yield = isinstance(div_yield, (int, float)) and div_yield > 0
        has_div_rate = isinstance(div_rate, (int, float)) and div_rate > 0
        dividend_status = "Ano" if (has_div_yield or has_div_rate) else "Ne"

        price_val = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", None)
        row_data = {
            "Ticker": t,
            "Jméno": info.get("shortName", "N/A"),
            "Měna": currency,
            "FX Kurz": fx_rate,
            "Cena": price_val,
            "Current Price": price_val,
            "Price": price_val,
            "Forward P/E": info.get("forwardPE", None),
            "EV/EBITDA": info.get("enterpriseToEbitda", None),
            "PEG Ratio": info.get("pegRatio", None),
            "ROA (%)": info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None,
            "ROIC (%)": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
            "Hrubá marže (%)": info.get("grossMargins", 0) * 100 if info.get("grossMargins") else None,
            "Provozní marže (%)": info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None,
            "Čistá marže (%)": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
            "Debt/Equity": info.get("debtToEquity", None),
            "Current Ratio": info.get("currentRatio", None),
            "Tržby YoY Růst (%)": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
            "Tržní kap.": market_cap,
            "Volné CF": fcf,
            "Celkový dluh": total_debt,
            "Hotovost": total_cash,
            "Tržní kap. (USD)": market_cap_usd,
            "Volné CF (USD)": fcf_usd,
            "Celkový dluh (USD)": total_debt_usd,
            "Hotovost (USD)": total_cash_usd,
            "Market Cap (USD)": market_cap_usd,
            "Free Cash Flow (USD)": fcf_usd,
            "Total Debt (USD)": total_debt_usd,
            "Cash & Equivalents (USD)": total_cash_usd,
            "Dividendy": dividend_status,
        }
        elapsed = time.perf_counter() - start_time
        logger.info(f"Successfully completed data fetch for {t} in {elapsed:.4f}s.")
        return row_data, warnings, errors
    except Exception as e:
        logger.error(f"Error fetching data for ticker {t}: {e}", exc_info=True)
        errors.append(f"Chyba při stahování dat pro {t}: {e}")
        return None, warnings, errors


@st.cache_data(ttl=300, show_spinner=False)
def fetch_company_info(tickers_string):
    """
    Retrieves basic metrics for the specified tickers concurrently using a thread pool.
    Returns: (DataFrame with data, list of warnings, list of errors)
    """
    data = []
    warnings = []
    errors = []
    ticker_list = [t.strip().upper() for t in tickers_string.split(',') if t.strip()]
    
    all_columns = [
        "Ticker", "Jméno", "Měna", "FX Kurz", "Cena", "Current Price", "Price",
        "Forward P/E", "EV/EBITDA", "PEG Ratio", "ROA (%)", 
        "ROIC (%)", "Hrubá marže (%)", "Provozní marže (%)", "Čistá marže (%)", 
        "Debt/Equity", "Current Ratio", "Tržby YoY Růst (%)",
        "Tržní kap.", "Volné CF", "Celkový dluh", "Hotovost",
        "Tržní kap. (USD)", "Volné CF (USD)", "Celkový dluh (USD)", "Hotovost (USD)",
        "Market Cap (USD)", "Free Cash Flow (USD)", "Total Debt (USD)", "Cash & Equivalents (USD)",
        "Dividendy"
    ]
    
    if not ticker_list:
        return pd.DataFrame(columns=all_columns), warnings, errors

    ctx = _get_streamlit_ctx()

    # Hardcode max_workers=2 to create a narrow pipeline and prevent HTTP 429 / Cloudflare rate limit blocks
    max_workers = min(len(ticker_list), 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_list = [(t, executor.submit(_fetch_single_ticker_info, t, ctx)) for t in ticker_list]
        for t, future in future_list:
            row_data, warn_list, err_list = future.result()
            if row_data:
                data.append(row_data)
            warnings.extend(warn_list)
            errors.extend(err_list)
            
    if not data:
        logger.warning(f"All ticker fetches failed or returned empty data for: {tickers_string}. Returning empty dataframe.")
        df = pd.DataFrame(columns=all_columns)
    else:
        df = pd.DataFrame(data)
        
    return df, warnings, errors


@st.cache_data(ttl=300, show_spinner=False)
def fetch_financial_history(ticker):
    """
    It downloads historical reports (both annual and quarterly) and converts them.
    """
    logger.info(f"Starting historical financial statements fetch for ticker: {ticker}")
    start_time = time.perf_counter()
    try:
        stock = yf.Ticker(ticker)
        inc_y = stock.income_stmt.T
        cf_y = stock.cashflow.T
        inc_q = stock.quarterly_income_stmt.T
        cf_q = stock.quarterly_cashflow.T
        
        if inc_y.empty or cf_y.empty:
            logger.warning(f"Empty annual financial statements dataframe returned for {ticker}.")
        if inc_q.empty or cf_q.empty:
            logger.warning(f"Empty quarterly financial statements dataframe returned for {ticker}.")
            
        elapsed = time.perf_counter() - start_time
        logger.info(f"Successfully fetched financial statement history for {ticker} in {elapsed:.4f}s.")
        return inc_y, cf_y, inc_q, cf_q
    except Exception as e:
        logger.error(f"Error fetching financial history for {ticker}: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def fetch_eps_history(ticker):
    """
    It downloads EPS history using the official Finnhub API.
    This allows it to bypass Yahoo Finance's cloud-based blocking.
    """
    logger.info(f"Starting Finnhub EPS history fetch for ticker: {ticker}")
    start_time = time.perf_counter()
    try:
        # Bezpečné načtení klíče
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            logger.warning(f"Finnhub API key not found in Streamlit secrets while fetching EPS for {ticker}.")
            print("Finnhub API klíč nenalezen.")
            return None
            
        # Finnhub API
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if not data:
                logger.warning(f"Finnhub API returned empty earnings data for ticker {ticker}.")
                return None
                
            # JSON TO PANDAS
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"Finnhub EPS dataframe is empty for ticker {ticker}.")
                return None
                
            # Finnhub returns data in reverse chronological order, but we want it in chronological order
            df['period'] = pd.to_datetime(df['period'])
            df = df.set_index('period')
            df = df.rename(columns={'actual': 'Reported EPS', 'estimate': 'EPS Estimate'})
            # Sorting
            df = df.sort_index().tail(8)
            
            elapsed = time.perf_counter() - start_time
            logger.info(f"Successfully fetched EPS history from Finnhub for {ticker} in {elapsed:.4f}s.")
            return df
        else:
            logger.warning(f"Finnhub API returned status code {response.status_code} for ticker {ticker}.")
            return None
            
    except Exception as e:
        logger.error(f"Chyba při stahování EPS z Finnhubu pro {ticker}: {e}", exc_info=True)
        print(f"Chyba při stahování EPS z Finnhubu: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(ticker, period="1y"):
    """
    Retrieves historical stock price data for the specified period.
    Supported periods: 1d, 5d, 1mo, 6mo, ytd, 1y, 5y, max
    """
    logger.info(f"Starting price history fetch for ticker {ticker} (period={period})")
    start_time = time.perf_counter()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist is None or hist.empty:
            logger.warning(f"Empty stock price history dataframe returned for ticker {ticker} (period={period}).")
        else:
            elapsed = time.perf_counter() - start_time
            logger.info(f"Successfully fetched price history for {ticker} (period={period}) in {elapsed:.4f}s.")
        return hist
    except Exception as e:
        logger.error(f"Error fetching price history for {ticker} (period={period}): {e}", exc_info=True)
        return None


def get_competitors(ticker: str, limit: int = 5) -> list[str]:
    """
    Dynamically fetches and populates peer companies based on a single seed ticker.
    Primary Source: Finnhub API company_peers endpoint (/stock/peers).
    Fallback Source: yfinance sector/industry top companies or recommendations.
    Returns a cleaned list of valid ticker strings starting with the seed ticker.
    """
    if not ticker or not isinstance(ticker, str):
        return []
        
    seed = ticker.upper().strip()
    logger.info(f"Starting competitor auto-discovery for ticker: {seed} (limit={limit})")
    start_time = time.perf_counter()
    peers: list[str] = []
    
    # Primary Source: Finnhub API
    try:
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if api_key:
            url = f"https://finnhub.io/api/v1/stock/peers?symbol={seed}&token={api_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    peers = [str(p).upper().strip() for p in data if p and isinstance(p, str)]
                    logger.info(f"Finnhub successfully returned {len(peers)} peers for {seed}.")
            else:
                logger.warning(f"Finnhub peers endpoint returned HTTP {response.status_code} for ticker {seed}. Attempting fallback.")
        else:
            logger.warning(f"Finnhub API key not found in Streamlit secrets during competitor search for {seed}. Attempting fallback.")
    except Exception as e:
        logger.warning(f"Finnhub API error or timeout during competitor search for {seed}: {e}. Attempting fallback.")
        
    # Fallback Source: yfinance sector/industry or recommendations
    if not peers or len(peers) <= 1:
        logger.info(f"Executing yfinance fallback competitor discovery for ticker {seed}.")
        try:
            stock = yf.Ticker(seed)
            # Try sector/industry top companies first
            info = getattr(stock, "info", {})
            sector_key = info.get("sectorKey") or info.get("sector", "").lower().replace(" ", "-")
            industry_key = info.get("industryKey") or info.get("industry", "").lower().replace(" ", "-")
            
            fb_peers: list[str] = []
            if hasattr(yf, "Industry") and industry_key:
                try:
                    ind = yf.Industry(industry_key)
                    top_df = getattr(ind, "top_companies", None)
                    if top_df is not None and not top_df.empty and "symbol" in top_df.columns:
                        fb_peers = [str(sym).upper().strip() for sym in top_df["symbol"].dropna().tolist()]
                except Exception as ex:
                    logger.warning(f"yf.Industry lookup failed for industry '{industry_key}': {ex}")
                    
            if not fb_peers and hasattr(yf, "Sector") and sector_key:
                try:
                    sec = yf.Sector(sector_key)
                    top_df = getattr(sec, "top_companies", None)
                    if top_df is not None and not top_df.empty and "symbol" in top_df.columns:
                        fb_peers = [str(sym).upper().strip() for sym in top_df["symbol"].dropna().tolist()]
                except Exception as ex:
                    logger.warning(f"yf.Sector lookup failed for sector '{sector_key}': {ex}")
            
            # If still empty, check stock recommendations or related items
            if not fb_peers:
                recs = getattr(stock, "recommendations", None)
                if recs is not None and not recs.empty and "symbol" in recs.columns:
                    fb_peers = [str(sym).upper().strip() for sym in recs["symbol"].dropna().tolist()]
                    
            if fb_peers:
                peers = fb_peers
                logger.info(f"yfinance fallback successfully discovered {len(peers)} peers for {seed}.")
            else:
                logger.warning(f"yfinance fallback could not discover peers for {seed}.")
        except Exception as e:
            logger.error(f"Error during yfinance fallback competitor search for {seed}: {e}", exc_info=True)
            
    # Clean, deduplicate, and format the final list starting with seed ticker
    cleaned_peers = [seed]
    seen = {seed}
    for p in peers:
        p_clean = p.upper().strip()
        # Exclude weird symbols or symbols with lots of dots/numbers unless valid
        if p_clean and p_clean not in seen and len(p_clean) <= 10:
            seen.add(p_clean)
            cleaned_peers.append(p_clean)
            if len(cleaned_peers) >= limit + 1:
                break
                
    elapsed = time.perf_counter() - start_time
    if len(cleaned_peers) > 1:
        logger.info(f"Competitor auto-discovery completed for {seed} in {elapsed:.4f}s. Discovered peers: {cleaned_peers}")
    else:
        logger.warning(f"No valid competitor peers discovered for {seed} after {elapsed:.4f}s. Returning seed ticker only.")
        
    return cleaned_peers