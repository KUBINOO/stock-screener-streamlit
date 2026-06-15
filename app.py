import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Imports from our modules 
from src.ai_verdict import get_ai_verdict
from src.data_fetcher import fetch_company_info, fetch_financial_history, fetch_eps_history, fetch_price_history
from src.dcf_model import get_dcf_base_data, calculate_dcf, calculate_reverse_dcf

# --- CACHING WRAPPERS ---
@st.cache_data(ttl=3600)
def get_cached_company_info(tickers):
    return fetch_company_info(tickers)

@st.cache_data(ttl=3600)
def get_cached_financial_history(ticker):
    return fetch_financial_history(ticker)

@st.cache_data(ttl=300) # Kratší cache (5 minut), aby cena byla aktuálnější
def get_cached_price_history(ticker, period):
    return fetch_price_history(ticker, period)

@st.cache_data(ttl=3600)
def get_cached_eps_history(ticker):
    return fetch_eps_history(ticker)

@st.cache_data(ttl=3600)
def cached_dcf_base_data(ticker):
    return get_dcf_base_data(ticker)

# --- UI  ---
st.set_page_config(page_title="Fundamental Screener", layout="wide")
st.title("Stock screener")

# Boční panel
st.sidebar.header("Nastavení screeneru")
tickers_input = st.sidebar.text_area("Zadejte tickery (oddělené čárkou):", "MSFT, AAPL, NVDA, INTC")

if tickers_input:
    # Organizing into tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Přehled & Cena", "Valuace & Ziskovost", "Finanční zdraví", "Historie výkazů", "Názor AI", "DCF Model"])

    # Download Key Metrics
    with st.spinner("Stahuji finanční data..."):
        # Let's take a closer look at the three values our fetcher now returns
        df, warnings, errors = get_cached_company_info(tickers_input)
        
        # Displaying any errors and warnings encountered during the download
        for w in warnings:
            st.warning(w)
        for e in errors:
            st.error(e)

# --- ZÁLOŽKA 1: Přehled a Cena ---
    with tab1:
        st.subheader("Aktuální vývoj ceny")
        ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        
        # Výběr tickeru (pokud jich uživatel zadal víc)
        selected_ticker_price = st.selectbox("Vyberte společnost pro graf ceny:", ticker_list, key="price_select")
        
        if selected_ticker_price:
            # Mapování tlačítek pro uživatele na formát, kterému rozumí yfinance
            period_options = {
                "1D": "1d", "5D": "5d", "1M": "1mo", "6M": "6mo", 
                "YTD": "ytd", "1R": "1y", "5R": "5y", "MAX": "max"
            }
            
            # Přepínač nad grafem
            selected_period = st.radio("Časové období:", list(period_options.keys()), horizontal=True, index=5) # index=5 znamená výchozí výběr '1R'
            
            # Získání dat
            hist_data = get_cached_price_history(selected_ticker_price, period_options[selected_period])
            
            if hist_data is not None and not hist_data.empty:
                # Výpočet aktuální ceny a změny za dané období
                current_price = hist_data['Close'].iloc[-1]
                start_price = hist_data['Close'].iloc[0]
                price_change = current_price - start_price
                pct_change = (price_change / start_price) * 100
                
                # Hezký vizuální ukazatel aktuální ceny
                st.metric(
                    label=f"Aktuální cena {selected_ticker_price}", 
                    value=f"${current_price:.2f}", 
                    delta=f"${price_change:.2f} ({pct_change:.2f}%) za zvolené období"
                )
                
                # Vykreslení Plotly grafu
                fig = go.Figure()
                
                # Zelená linka pokud to roste, červená pokud to klesá (jako na yfinance)
                line_color = '#2ca02c' if price_change >= 0 else '#d62728'
                
                fig.add_trace(go.Scatter(
                    x=hist_data.index, 
                    y=hist_data['Close'], 
                    mode='lines', 
                    name='Uzavírací cena', 
                    line=dict(color=line_color, width=2),
                    fill='tozeroy', # Vykreslí stín pod křivkou pro lepší design
                    fillcolor=line_color.replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in line_color else 'rgba(44, 160, 44, 0.1)' if price_change >=0 else 'rgba(214, 39, 40, 0.1)'
                ))
                
                # Skrytí víkendů a svátků na ose X pro kratší období (aby graf neměl mezery)
                if selected_period in ["1D", "5D", "1M"]:
                    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

                fig.update_layout(
                    xaxis_title="Datum", 
                    yaxis_title="Cena (USD)", 
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=30, b=0) # Zmenšení okrajů
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Nepodařilo se stáhnout historii cen pro {selected_ticker_price}.")

    # --- ZÁLOŽKA 2: Valuace ---
    with tab2:
        st.subheader("Ocenění a ziskovost společností")
        val_cols = ["Ticker", "Jméno", "Forward P/E", "EV/EBITDA", "PEG Ratio", "ROA (%)", "ROIC (%)", "Hrubá marže (%)", "Provozní marže (%)", "Čistá marže (%)"]
        
        st.dataframe(df[val_cols].style.format({
            "Forward P/E": "{:.2f}", "EV/EBITDA": "{:.2f}", "PEG Ratio": "{:.2f}",
            "ROA (%)": "{:.1f}%", "ROIC (%)": "{:.1f}%", "Hrubá marže (%)": "{:.1f}%",
            "Provozní marže (%)": "{:.1f}%", "Čistá marže (%)": "{:.1f}%"
        }, na_rep="N/A"), use_container_width=True)

    # --- ZÁLOŽKA 3: Finanční zdraví ---
    with tab3:
        st.subheader("Rozvaha a dluh")
        health_cols = ["Ticker", "Jméno", "Debt/Equity", "Current Ratio", "Tržby YoY Růst (%)"]
        
        st.dataframe(df[health_cols].style.format({
            "Debt/Equity": "{:.2f}", "Current Ratio": "{:.2f}", "Tržby YoY Růst (%)": "{:.1f}%"
        }, na_rep="N/A"), use_container_width=True)

    # --- ZÁLOŽKA 4: Grafy (Detail jedné firmy) ---
    with tab4:
        st.subheader("Historický vývoj financí")
        ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        selected_ticker = st.selectbox("Vyberte společnost pro detailní grafy:", ticker_list)

        if selected_ticker:
            # Získáme všechny datasety (roční i kvartální)
            inc_y, cf_y, inc_q, cf_q = get_cached_financial_history(selected_ticker)
            
            # Přepínač Roční vs. Kvartální
            period = st.radio("Frekvence výkazů:", ["Roční (Yearly)", "Kvartální (Quarterly)"], horizontal=True)
            
            # Nastavení aktivních datasetů podle výběru
            if period == "Roční (Yearly)":
                active_inc = inc_y
                active_cf = cf_y
                x_format = active_inc.index.year # Zobrazíme jen roky
            else:
                active_inc = inc_q
                active_cf = cf_q
                x_format = active_inc.index.strftime('%Y-Q%q') # Formát typu 2023-Q1
                
            # GRAF 1: VÝKAZY
            if "Total Revenue" in active_inc.columns:
                fig1 = go.Figure()
                
                # Tržby
                revenue = active_inc["Total Revenue"]
                fig1.add_trace(go.Bar(x=x_format, y=revenue.values, name="Tržby (Revenue)", marker_color='#1f77b4'))
                
                # Provozní zisk
                if "Operating Income" in active_inc.columns:
                    op_income = active_inc["Operating Income"]
                    fig1.add_trace(go.Bar(x=x_format, y=op_income.values, name="Provozní zisk (EBIT)", marker_color='#ff7f0e'))

                # Čistý zisk
                if "Net Income" in active_inc.columns:
                    net_income = active_inc["Net Income"]
                    fig1.add_trace(go.Bar(x=x_format, y=net_income.values, name="Čistý zisk", marker_color='#2ca02c'))

                # Free Cash Flow
                if "Free Cash Flow" in active_cf.columns:
                    fcf = active_cf["Free Cash Flow"]
                    fig1.add_trace(go.Bar(x=active_cf.index.strftime('%Y-%m') if period == "Kvartální (Quarterly)" else active_cf.index.year, y=fcf.values, name="Free Cash Flow", marker_color='#9467bd'))

                fig1.update_layout(
                    title=f"Vývoj Tržeb, Zisku a FCF: {selected_ticker}", 
                    barmode='group', 
                    xaxis_title="Období", 
                    yaxis_title="USD",
                    hovermode="x unified"
                )
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.warning("Data o tržbách nejsou k dispozici.")

            st.markdown("---")
            
            # GRAF 2: EPS SURPRISE
            st.subheader("Earnings per Share (EPS) - Odhad vs. Realita")
            eps_data = get_cached_eps_history(selected_ticker)
            
            if eps_data is not None and not eps_data.empty:
                fig2 = go.Figure()

                # Odhad (Šedá tečka/kruh)
                fig2.add_trace(go.Scatter(
                    x=eps_data.index.strftime('%Y-%m-%d'),
                    y=eps_data['EPS Estimate'],
                    mode='markers+lines', # Tečky spojené linkou pro trend
                    name='Odhad analytiků',
                    marker=dict(color='gray', size=10, symbol='circle-open'),
                    line=dict(color='gray', width=1, dash='dot')
                ))

                # Dynamické barvy pro realitu (Zelená = Beat, Červená = Miss)
                colors = ['#2ca02c' if rep >= est else '#d62728' for rep, est in zip(eps_data['Reported EPS'], eps_data['EPS Estimate'])]

                # Realita (Barevná tečka)
                fig2.add_trace(go.Scatter(
                    x=eps_data.index.strftime('%Y-%m-%d'),
                    y=eps_data['Reported EPS'],
                    mode='markers',
                    name='Reportovaná Realita',
                    marker=dict(color=colors, size=14, symbol='circle')
                ))

                fig2.update_layout(
                    title=f"EPS Překvapení (Posledních 8 kvartálů): {selected_ticker}",
                    xaxis_title="Datum vyhlášení",
                    yaxis_title="Zisk na akcii (USD)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Historie EPS (očekávání vs realita) není u tohoto tickeru k dispozici.")
            
    # --- ZÁLOŽKA 5: Ai názor ---
    with tab5:
        st.subheader("🤖 Rychlý AI Verdikt")
        ticker_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        selected_ticker_ai = st.selectbox("Vyber společnost pro AI analýzu:", ticker_list, key="ai_select")

        if selected_ticker_ai:
            # Získání dat pro konkrétní ticker z naší tabulky (ochrana pokud data chybí)
            if not df[df['Ticker'] == selected_ticker_ai].empty:
                firemni_data = df[df['Ticker'] == selected_ticker_ai].iloc[0].to_dict()
                
                if st.button(f"Vygenerovat verdikt pro {selected_ticker_ai}", type="primary"):
                    with st.spinner("AI studuje finanční výkazy..."):
                        vysledek = get_ai_verdict(selected_ticker_ai, firemni_data)
                        
                        st.info(vysledek)
                        
                        # Grafické odlišení posledního řádku (verdiktu)
                        lines = vysledek.strip().split('\n')
                        verdict = lines[-1].strip().upper()
                        
                        if "STRONG BUY" in verdict:
                            st.success("🔥 AI VERDIKT: " + verdict)
                        elif "BUY" in verdict:
                            st.success("📈 AI VERDIKT: " + verdict)
                        elif "WAIT" in verdict:
                            st.warning("⏳ AI VERDIKT: " + verdict)
                        elif "DONT" in verdict or "DON'T" in verdict:
                            st.error("🛑 AI VERDIKT: " + verdict)
            else:
                st.warning(f"Nejsou k dispozici data pro analýzu {selected_ticker_ai}.")
    
    # --- ZÁLOŽKA 6: DCF MODEL ---
    with tab6:
        st.subheader("Parametry DCF modelu")
        
        ticker_list_dcf = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        selected_ticker_dcf = st.selectbox("Vyberte společnost pro nacenění:", ticker_list_dcf, key="dcf_select")
        
        if selected_ticker_dcf:
            dcf_data = cached_dcf_base_data(selected_ticker_dcf)
            
            if dcf_data and dcf_data["shares_outstanding"] > 0:
                # Rozdělení na dva sloupce: vlevo slidery, vpravo výsledek
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("##### 🎛️ Nastavení parametrů")
                    
                    # 1. FCF TTM Slider (Výchozí je realita, posuvník jde od 0 do 2x reálné FCF pro stresstesty)
                    base_fcf = dcf_data["fcf_ttm"]
                    fcf_in_billions = base_fcf / 1e9
                    slider_fcf = st.slider(
                        "FCF TTM ($ Miliardy)", 
                        min_value=0.0, 
                        max_value=max(10.0, fcf_in_billions * 2), 
                        value=max(0.0, fcf_in_billions), 
                        step=0.1
                    )
                    
                    # 2. Délka Fáze 1
                    slider_years = st.slider("DÉLKA FÁZE 1 (roky)", min_value=3, max_value=10, value=10, step=1)
                    
                    # 3. Růst Fáze 1 (od -10% do 40%)
                    slider_growth = st.slider("RŮST FÁZE 1 (%)", min_value=-10.0, max_value=40.0, value=15.0, step=0.5)
                    
                    # 4. Terminální růst (od 0% do 5%)
                    slider_terminal = st.slider("TERMINÁLNÍ RŮST (%)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
                    
                    # 5. WACC / Diskontní míra (od 5% do 20%)
                    slider_wacc = st.slider("WACC (%)", min_value=5.0, max_value=20.0, value=10.0, step=0.1)
                
                with col2:
                    st.markdown("##### 🎯 Vnitřní hodnota (Fair Value)")
                    
                    # Přepočet hodnot ze sliderů do formátu pro matematiku (procenta na desetinná čísla)
                    calc_fcf = slider_fcf * 1e9
                    calc_growth = slider_growth / 100.0
                    calc_terminal = slider_terminal / 100.0
                    calc_wacc = slider_wacc / 100.0
                    
                    # Výpočet
                    intrinsic_value, ev = calculate_dcf(
                        calc_fcf, slider_years, calc_growth, 
                        calc_terminal, calc_wacc, 
                        dcf_data["shares_outstanding"], dcf_data["net_debt"]
                    )
                    
                    current_price = dcf_data["current_price"]
                    margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100 if current_price else 0
                    
                    # Vizuál výsledku
                    st.metric(label=f"Vypočítaná cena {selected_ticker_dcf}", value=f"${intrinsic_value:.2f}")
                    st.metric(label="Aktuální cena na trhu", value=f"${current_price:.2f}")
                    
                    # Barevné zhodnocení Margin of Safety
                    if intrinsic_value > current_price:
                        st.success(f"Akcie je PODHODNOCENÁ.\n\nMargin of Safety: +{margin_of_safety:.1f}%")
                    else:
                        st.error(f"Akcie je NADROHODNOCENÁ.\n\nPrémiová přirážka: {margin_of_safety:.1f}%")
            # --- ODDĚLOVAČ A REVERSE DCF ---
                st.markdown("---")
                st.markdown("#### 🟡 REVERSE DCF: Co trh implicitně očekává")
                
                # Výpočet implikovaného růstu
                implied_growth = calculate_reverse_dcf(
                    current_price, calc_fcf, slider_years, 
                    calc_terminal, calc_wacc, 
                    dcf_data["shares_outstanding"], dcf_data["net_debt"]
                )
                
                implied_growth_pct = implied_growth * 100
                
                # Hlavní hláška alá tvůj screen
                st.markdown(f"### Aktuální cena **${current_price:.2f}** implikuje **{implied_growth_pct:.1f}%** růst FCF po dobu {slider_years} let.")
                st.caption(f"Při tvých předpokladech (WACC {slider_wacc:.1f}%, terminální růst {slider_terminal:.1f}%) musí FCF růst tímto tempem, aby vnitřní hodnota přesně odpovídala aktuální ceně na trhu.")
                
                # Karty s porovnáním
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.metric(
                        label="TRH OČEKÁVÁ (Implied Growth)", 
                        value=f"{implied_growth_pct:.1f}%"
                    )
                with rc2:
                    diff_vs_market = slider_growth - implied_growth_pct
                    # Pokud jsi optimističtější než trh, je to zelené (+). Pokud ne, červené (-).
                    st.metric(
                        label="TVŮJ ODHAD", 
                        value=f"{slider_growth:.1f}%", 
                        delta=f"{diff_vs_market:.1f} p.b. vs trh"
                    )
                        
            else:
                st.warning("Nepodařilo se stáhnout potřebná data (FCF, počet akcií) pro tento DCF model.")