# 📈 AI Stock Screener & DCF Valuation Tool

A comprehensive web application built with Python and Streamlit for fundamental stock analysis. The tool integrates real-time market data fetching, advanced quantitative financial modeling, and an AI-powered analyst to provide actionable investment insights.

> **Note:** The underlying codebase and prompt engineering are structured in English, while the User Interface (UI) and AI outputs are localized in Czech (Czech UI) to serve the domestic market.

## 🚀 Key Features

* **Real-Time Data Fetching:** Automated retrieval of financial statements, price history, and key metrics via the `yfinance` API and `Finnhub API` (for reliable EPS surprises).
* **Interactive Price Charts:** Dynamic, user-friendly stock performance visualization using `Plotly`.
* **2-Stage DCF Model:** An interactive Discounted Cash Flow calculator with adjustable parameters (WACC, Growth Rates, FCF) to determine intrinsic value and Margin of Safety.
* **Vectorized Monte Carlo Simulation:** Executes 10,000 stochastic DCF scenarios instantly using `NumPy`. This high-performance quantitative engine models probability distributions by adding statistical noise to growth rates, terminal values, and beta.
* **Reverse DCF:** Algorithmic calculation (binary search) to determine the market-implied growth rate based on the current stock price.
* **AI Analyst Verdicts:** Integration with the `Groq API` (Llama 3.3) for rapid, uncompromising, and data-driven investment verdicts (BUY, HOLD, SELL) based on fundamental health and macroeconomic moat.

## 🛠️ Tech Stack

* **Frontend & Framework:** Streamlit
* **Quantitative Engine & Processing:** Pandas, NumPy, SciPy (Strongly-typed using Dataclasses)
* **Data Sources:** Yahoo Finance API (`yfinance`), Finnhub API
* **Visualization:** Plotly
* **LLM Integration:** Groq API (Llama-3.3-70b)

## ⚙️ Local Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/KUBINOO/stock-screener-streamlit.git](https://github.com/KUBINOO/stock-screener-streamlit.git)


2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


3. Set up your API Keys:
   Create a `.streamlit/secrets.toml` file in the root directory and add your API keys:
   ```bash
   GROQ_API_KEY = "your_groq_api_key_here"
   FINNHUB_API_KEY = "your_finnhub_api_key_here"


4. Run the app:
   ```bash
   streamlit run app.py