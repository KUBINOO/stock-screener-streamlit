import pandas as pd
import yfinance as yf
import requests
import streamlit as st

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
        "Ticker", "Jméno", "Forward P/E", "EV/EBITDA", "PEG Ratio", "ROA (%)", 
        "ROIC (%)", "Hrubá marže (%)", "Provozní marže (%)", "Čistá marže (%)", 
        "Debt/Equity", "Current Ratio", "Tržby YoY Růst (%)"
    ]
    
    for t in ticker_list:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            if not info or 'symbol' not in info:
                warnings.append(f"Yahoo Finance nevrátilo kompletní data pro ticker: {t}")
                continue
            
            data.append({
                "Ticker": t,
                "Jméno": info.get("shortName", "N/A"),
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
            
        # Volání Finnhub API
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