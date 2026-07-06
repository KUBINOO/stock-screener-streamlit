import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf
import requests
# pyrefly: ignore [missing-import]
import streamlit as st
import time
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


def fetch_company_info(tickers_string):
    """
    Retrieves basic metrics for the specified tickers.
    Returns: (DataFrame with data, list of warnings, list of errors)
    """
    data = []
    warnings = []
    errors = []
    ticker_list = [t.strip().upper() for t in tickers_string.split(',') if t.strip()]
    
    all_columns = [
        "Ticker", "Jméno", "Měna", "FX Kurz",
        "Forward P/E", "EV/EBITDA", "PEG Ratio", "ROA (%)", 
        "ROIC (%)", "Hrubá marže (%)", "Provozní marže (%)", "Čistá marže (%)", 
        "Debt/Equity", "Current Ratio", "Tržby YoY Růst (%)",
        "Tržní kap.", "Volné CF", "Celkový dluh", "Hotovost",
        "Tržní kap. (USD)", "Volné CF (USD)", "Celkový dluh (USD)", "Hotovost (USD)",
        "Market Cap (USD)", "Free Cash Flow (USD)", "Total Debt (USD)", "Cash & Equivalents (USD)"
    ]
    
    for t in ticker_list:
        logger.info(f"Starting data fetch operation for ticker: {t}")
        start_time = time.perf_counter()
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            if not info or 'symbol' not in info:
                msg = f"Yahoo Finance nevrátilo kompletní data pro ticker: {t}"
                logger.warning(f"Incomplete info returned from Yahoo Finance for ticker {t} (missing critical keys).")
                warnings.append(msg)
                continue
            
            currency = info.get("currency", "USD")
            if not currency:
                logger.warning(f"Missing currency key for ticker {t}. Falling back to default USD.")
                currency = "USD"
            currency = currency.upper().strip()
            fx_rate = get_fx_rate(currency)
            
            market_cap = info.get("marketCap", None)
            fcf = info.get("freeCashflow", None)
            if not fcf:
                fcf = info.get("operatingCashflow", None)
                if not fcf:
                    logger.warning(f"Missing critical financial keys 'freeCashflow' and 'operatingCashflow' for {t}.")
            if market_cap is None:
                logger.warning(f"Missing critical financial key 'marketCap' for {t}.")
            total_debt = info.get("totalDebt", None)
            total_cash = info.get("totalCash", None)
            if total_debt is None or total_cash is None:
                logger.warning(f"Missing balance sheet keys (totalDebt or totalCash) for {t}.")
            
            market_cap_usd = market_cap * fx_rate if market_cap is not None else None
            fcf_usd = fcf * fx_rate if fcf is not None else None
            total_debt_usd = total_debt * fx_rate if total_debt is not None else None
            total_cash_usd = total_cash * fx_rate if total_cash is not None else None
            
            data.append({
                "Ticker": t,
                "Jméno": info.get("shortName", "N/A"),
                "Měna": currency,
                "FX Kurz": fx_rate,
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
            })
            elapsed = time.perf_counter() - start_time
            logger.info(f"Successfully completed data fetch for {t} in {elapsed:.4f}s.")
        except Exception as e:
            logger.error(f"Error fetching data for ticker {t}: {e}", exc_info=True)
            errors.append(f"Chyba při stahování dat pro {t}: {e}")
            
    if not data:
        logger.warning(f"All ticker fetches failed or returned empty data for: {tickers_string}. Returning empty dataframe.")
        df = pd.DataFrame(columns=all_columns)
    else:
        df = pd.DataFrame(data)
        
    return df, warnings, errors


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