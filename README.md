# 📈 AI Stock Screener & DCF Valuation Tool

A comprehensive web application built with Python and Streamlit for fundamental stock analysis. The tool integrates real-time market data fetching, interactive financial modeling, and an AI-powered analyst to provide actionable investment insights.

> **Note:** The underlying codebase and prompt engineering are structured in English, while the User Interface (UI) and AI outputs are localized in Czech (Czech UI) to serve the domestic market.

## 🚀 Key Features

* **Real-Time Data Fetching:** Automated retrieval of financial statements, price history, and key metrics via the `yfinance` API.
* **Interactive Price Charts:** Dynamic, user-friendly stock performance visualization using `Plotly`.
* **2-Stage DCF Model:** An interactive Discounted Cash Flow calculator with adjustable parameters (WACC, Growth Rates, FCF) to determine intrinsic value and Margin of Safety.
* **Reverse DCF:** Algorithmic calculation (binary search) to determine the market-implied growth rate based on the current stock price.
* **AI Analyst Verdicts:** Integration with the `Groq API` (Llama 3.3) for rapid, uncompromising, and data-driven investment verdicts (BUY, HOLD, SELL) based on fundamental health and macroeconomic moat.

## 🛠️ Tech Stack

* **Frontend & Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Data Source:** Yahoo Finance API (`yfinance`)
* **Visualization:** Plotly
* **LLM Integration:** Groq API (Llama-3.3-70b)

## ⚙️ Local Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/KUBINOO/python-streamlit-screener.git](https://github.com/KUBINOO/python-streamlit-screener.git)