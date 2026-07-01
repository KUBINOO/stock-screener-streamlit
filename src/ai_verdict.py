import streamlit as st
from groq import Groq
import time
import json

# Obtaining an API key directly from trusted Streamlit Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    raise ValueError("I can't find the key! Check to see if you have a .streamlit/secrets.toml file and if it contains GROQ_API_KEY")

# Client Initialization
client = Groq(api_key=api_key)


# Prompting AI
def get_ai_verdict(ticker, company_data):
    context = f"""
    Ticker: {ticker}
    --- VALUACE ---
    Forward P/E: {company_data.get('Forward P/E', 'N/A')}
    EV/EBITDA: {company_data.get('EV/EBITDA', 'N/A')}
    PEG Ratio: {company_data.get('PEG Ratio', 'N/A')}
    
    --- EFEKTIVITA A MARŽE ---
    ROIC: {company_data.get('ROIC (%)', 'N/A')}%
    ROA: {company_data.get('ROA (%)', 'N/A')}%
    Hrubá marže: {company_data.get('Hrubá marže (%)', 'N/A')}%
    Provozní marže: {company_data.get('Provozní marže (%)', 'N/A')}%
    Čistá marže: {company_data.get('Čistá marže (%)', 'N/A')}%
    
    --- RŮST A ZDRAVÍ ---
    Meziroční růst tržeb: {company_data.get('Tržby YoY Růst (%)', 'N/A')}%
    Dluh/Vlastní jmění (Debt/Equity): {company_data.get('Debt/Equity', 'N/A')}
    Current Ratio: {company_data.get('Current Ratio', 'N/A')}
    """

    system_prompt = """You are an expert, uncompromising quantitative financial analyst and stock market strategist.
You evaluate financial data against strict quantitative heuristics using a Chain-of-Thought (CoT) reasoning process before forming a final verdict.

STRICT QUANTITATIVE HEURISTICS TO OBEY:
1. Valuation:
   - PEG Ratio < 1.0 indicates undervalued / attractive growth valuation.
   - Forward P/E > 30 requires exceptional revenue growth (>20% YoY) to justify; otherwise it is considered expensive.
2. Efficiency & Profitability:
   - ROIC > 15% strongly indicates a competitive moat and superior capital allocation.
   - High margins (Gross, Operating, Net) reflect pricing power and operational efficiency.
3. Financial Health & Balance Sheet:
   - Debt/Equity > 1.5 combined with Current Ratio < 1.0 is a severe red flag indicating liquidity and solvency risk. If this condition is met, you MUST NOT issue a "STRONG BUY" verdict under any circumstances.
   - Current Ratio > 1.5 and low Debt/Equity (< 0.5) indicate a fortress balance sheet.

OUTPUT FORMAT REQUIREMENTS:
You MUST respond with a valid JSON object only. The JSON object must strictly adhere to this schema:
{
  "internal_reasoning": "Step-by-step quantitative evaluation of the raw metrics against the heuristics (e.g., analyzing valuation ratios, ROIC efficiency, debt safety, and moat). You must generate this field first as your private scratchpad.",
  "fundamental_summary": "Exactly 3 sentences in Czech summarizing the quantitative fundamental data (valuation, efficiency, profitability, and financial health).",
  "business_moat": "Exactly 3 sentences in Czech evaluating the business model, competitive moat (barriers to entry, pricing power), and macroeconomic risks.",
  "verdict": "Exactly one of the following English strings: BUY, STRONG BUY, DONT BUY, WAIT FOR BETTER PRICE."
}

CRITICAL RULES:
- The `internal_reasoning` field must be generated FIRST, detailing your step-by-step analytical deductions.
- Both `fundamental_summary` and `business_moat` MUST be written in the Czech language and MUST be EXACTLY 3 sentences each.
- The `verdict` must be strictly one of the 4 permitted English strings: BUY, STRONG BUY, DONT BUY, WAIT FOR BETTER PRICE.
- Do not output anything outside the JSON object."""

    user_prompt = f"""Evaluate the fundamental data for company {ticker}:
{context}

Perform step-by-step Chain-of-Thought (CoT) analysis in `internal_reasoning` checking the metrics against the quantitative heuristics.
Then write `fundamental_summary` (EXACTLY 3 sentences in Czech), `business_moat` (EXACTLY 3 sentences in Czech), and decide on the `verdict` (BUY, STRONG BUY, DONT BUY, or WAIT FOR BETTER PRICE).
Output ONLY the JSON object."""

    max_pokusu = 3
    
    for pokus in range(max_pokusu):
        try:
            # Calling Llama 3.3 model with Groq in JSON mode with determinism setup
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content or ""
            parsed_data = json.loads(raw_content.strip())
            
            return {
                "fundamental_summary": parsed_data.get("fundamental_summary", "Shrnutí fundamentů není k dispozici."),
                "business_moat": parsed_data.get("business_moat", "Analýza obchodního modelu není k dispozici."),
                "verdict": parsed_data.get("verdict", "WAIT FOR BETTER PRICE"),
                "internal_reasoning": parsed_data.get("internal_reasoning", ""),
                "error": None
            }
            
        except (json.JSONDecodeError, Exception) as e:
            chybova_hlaska = str(e)
            
            # If the API is overloaded (503), rate-limited (429), or JSON failed to decode
            if (isinstance(e, json.JSONDecodeError) or "429" in chybova_hlaska or "503" in chybova_hlaska) and pokus < max_pokusu - 1:
                cas_cekani = 3
                st.warning(f"Groq API nebo parsování JSON nabírá dech, AI zkusí znovu za {cas_cekani} vteřiny...")
                time.sleep(cas_cekani)
                continue
                
            return {
                "fundamental_summary": f"Chyba při komunikaci s AI nebo analýze dat: {chybova_hlaska}",
                "business_moat": "",
                "verdict": "ERROR",
                "internal_reasoning": "",
                "error": chybova_hlaska
            }

