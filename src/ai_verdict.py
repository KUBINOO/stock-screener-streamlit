import streamlit as st
from groq import Groq
import time

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

    prompt = f"""
    Here is the fundamental data for the company {ticker}:
    {context}
    
    Your analysis rules:
    1. Write EXACTLY three sentences summarizing the hard data: valuation, efficiency, and financial health based on the table.
    2. Create a second paragraph of EXACTLY three sentences evaluating the business model. Analyze the 'moat' (barriers to entry, pricing power) and major macroeconomic risks.
    3. IMPORTANT: The entire analysis in steps 1 and 2 MUST be written in the Czech language.
    4. On a new final line, write ONLY ONE of these verdicts (keep this in English): BUY, STRONG BUY, DONT BUY, WAIT FOR BETTER PRICE.
    5. Do not use any introductory or concluding filler phrases. Go straight to the point.
    """
    max_pokusu = 3
    
    for pokus in range(max_pokusu):
        try:
            # Calling Llama 3.3 modelu with Groq
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {"role": "system", "content": "Jsi expertní, nekompromisní a stručný finanční analytik akciového trhu."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            chybova_hlaska = str(e)
            
            # If the API is overloaded (503) or we've hit a rate limit (429)
            if ("429" in chybova_hlaska or "503" in chybova_hlaska) and pokus < max_pokusu - 1:
                cas_cekani = 3
                st.warning(f"Groq API nabírá dech, AI zkusí znovu za {cas_cekani} vteřiny...")
                time.sleep(cas_cekani)
                continue
                
            return f"Chyba při komunikaci s AI: {chybova_hlaska}"

