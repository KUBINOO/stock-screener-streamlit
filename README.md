# 📈 AI Stock Screener & DCF Valuation Tool

A comprehensive web application built with Python and Streamlit for fundamental stock analysis. The tool integrates real-time market data fetching, advanced quantitative financial modeling, and an AI-powered analyst to provide actionable investment insights.

> **Note:** The underlying codebase and prompt engineering are structured in English, while the User Interface (UI) and AI outputs are localized in Czech (Czech UI) to serve the domestic market.

# 👉 Try it online
`https://akciovyscreener.streamlit.app/`


## 🚀 Key Features

* **Resilient Real-Time Data Pipeline:** Automated retrieval of financial statements, price history, and key metrics via the `yfinance` and `Finnhub` APIs. Features short-term Streamlit caching (`@st.cache_data`), throttled thread pool concurrency (`max_workers=2`), connection staggering/jitter, and browser impersonation (`curl_cffi`) with exponential backoff to reliably bypass Cloudflare rate-limit blocks (`HTTP 429` / `curl 28`).
* **Interactive Price Charts:** Dynamic, user-friendly stock performance visualization using `Plotly`.
* **2-Stage DCF Model & Strict CapEx Guardrail:** An interactive Discounted Cash Flow calculator with adjustable parameters (WACC, Growth Rates, FCF) to determine intrinsic value and Margin of Safety. Features strict Free Cash Flow (FCF) fallback validation requiring both Operating Cash Flow and Capital Expenditures (`CapEx`) (`ocf - abs(capex)`), flagging missing CapEx (`np.nan`) to prevent dangerous valuation overestimates alongside multi-currency guardrails.
* **Vectorized Monte Carlo Simulation:** Executes 10,000 stochastic DCF scenarios instantly using `NumPy`. This high-performance quantitative engine models probability distributions by adding statistical noise to growth rates, terminal values, and beta.
* **Reverse DCF:** Algorithmic calculation (binary search) to determine the market-implied growth rate based on the current stock price.
* **AI Analyst Verdicts & Gateway Architecture:** Integration with `AIGateway` utilizing the `Groq API` (`Llama-3.3-70b`) as the primary engine with automated Tier 2 fallback (`Google Gemini API`) for rapid, uncompromising, and data-driven investment verdicts (BUY, HOLD, SELL) based on fundamental health and macroeconomic moat.
* **Quantitative Peer Scoring Engine:** Vectorized peer scoring engine (`Quality_Score`, `Value_Score`, `Safety_Score`, `Total_Score`) ranking peer companies across normalized 0–100 percentile scales.
* **Glass Box CIO Analysis (Multi-Agent Committee):** Institutional-grade multi-agent debate simulation (Moat, Valuation, and Safety Agents) synthesizing deterministic peer scores into transparent chain-of-thought internal reasoning and a definitive CIO investment verdict.
* **Interactive Relative Valuation Matrix:** Customizable 2D `Plotly` scatter plots comparing peers across user-selected profitability/growth vs. valuation metrics, dynamically colored by `Total_Score`.
* **Executive Tear Sheet Export:** One-click generation of institutional HTML/PDF executive summary reports combining DCF valuation models, fundamental scorecards, and AI investment verdicts.

## 🛠️ Tech Stack

* **Frontend & Framework:** Streamlit
* **Quantitative Engine & Processing:** Pandas, NumPy, SciPy (Strongly-typed using Dataclasses)
* **Data Sources & Resiliency:** Yahoo Finance API (`yfinance`), Finnhub API, `curl_cffi` (Chrome Impersonation & Anti-Rate Limiting)
* **Visualization:** Plotly
* **LLM & AI Gateway Integration:** Groq API (`Llama-3.3-70b`), Google Gemini API (Tier 2 Fallback via `AIGateway`)

## ⚙️ Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KUBINOO/stock-screener-streamlit.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API Keys:
   Create a `.streamlit/secrets.toml` file in the root directory and add your API keys:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   FINNHUB_API_KEY = "your_finnhub_api_key_here"
   GEMINI_API_KEY = "your_gemini_api_key_here" # Optional: Tier 2 AI fallback
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```