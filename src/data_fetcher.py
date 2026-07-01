import pandas as pd
import yfinance as yf
import requests
import streamlit as st

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
    try:
        fx_stock = yf.Ticker(ticker_pair)
        # Try fetching recent price history first
        hist = fx_stock.history(period="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            rate = float(hist["Close"].iloc[-1])
            if rate > 0:
                return rate
        # Fall back to info dictionary if history is empty
        info = fx_stock.info
        if info and "regularMarketPrice" in info and info["regularMarketPrice"]:
            rate = float(info["regularMarketPrice"])
            if rate > 0:
                return rate
    except Exception as e:
        print(f"FX live translation failed for pair {ticker_pair}: {e}. Using fallback rate.")
        
    return FX_FALLBACK_RATES.get(curr, 1.0)

def get_currency_symbol(currency: str) -> str:
    """Returns the visual symbol ($, €, £, ¥, etc.) for a given currency code."""
    if not currency:
        return "$"
    return CURRENCY_SYMBOLS.get(currency.upper().strip(), f"{currency} ")

def get_regional_rf_rate(currency: str) -> float:
    """Returns the macroeconomic Risk-Free Rate proxy for the given currency denomination."""
    if not currency:
        return 0.045
    return REGIONAL_RF_RATES.get(currency.upper().strip(), 0.045)


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
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            if not info or 'symbol' not in info:
                warnings.append(f"Yahoo Finance nevrátilo kompletní data pro ticker: {t}")
                continue
            
            currency = info.get("currency", "USD")
            if not currency:
                currency = "USD"
            currency = currency.upper().strip()
            fx_rate = get_fx_rate(currency)
            
            market_cap = info.get("marketCap", None)
            fcf = info.get("freeCashflow", None)
            if not fcf:
                fcf = info.get("operatingCashflow", None)
            total_debt = info.get("totalDebt", None)
            total_cash = info.get("totalCash", None)
            
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
        except Exception as e:
            errors.append(f"Chyba při stahování dat pro {t}: {e}")
            
    if not data:
        df = pd.DataFrame(columns=all_columns)
    else:
        df = pd.DataFrame(data)
        
    return df, warnings, errors


def fetch_financial_history(ticker):
    """
    It downloads historical reports (both annual and quarterly) and converts them.
    """
    stock = yf.Ticker(ticker)
    inc_y = stock.income_stmt.T
    cf_y = stock.cashflow.T
    inc_q = stock.quarterly_income_stmt.T
    cf_q = stock.quarterly_cashflow.T
    
    return inc_y, cf_y, inc_q, cf_q


def fetch_eps_history(ticker):
    """
    It downloads EPS history using the official Finnhub API.
    This allows it to bypass Yahoo Finance's cloud-based blocking.
    """
    try:
        # Bezpečné načtení klíče
        api_key = st.secrets.get("FINNHUB_API_KEY")
        if not api_key:
            print("Finnhub API klíč nenalezen.")
            return None
            
        # Finnhub API
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={api_key}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if not data:
                return None
                
            # JSON TO PANDAS
            df = pd.DataFrame(data)
            
            # Finnhub returns data in reverse chronological order, but we want it in chronological order
            df['period'] = pd.to_datetime(df['period'])
            df = df.set_index('period')
            df = df.rename(columns={'actual': 'Reported EPS', 'estimate': 'EPS Estimate'})
            # Sorting
            df = df.sort_index().tail(8)
            
            return df
        else:
            return None
            
    except Exception as e:
        print(f"Chyba při stahování EPS z Finnhubu: {e}")
        return None


def fetch_price_history(ticker, period="1y"):
    """
    Retrieves historical stock price data for the specified period.
    Supported periods: 1d, 5d, 1mo, 6mo, ytd, 1y, 5y, max
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except Exception:
        return None