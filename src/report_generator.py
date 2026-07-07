"""
Executive Report Generator Module (Tear Sheet)

Generates self-contained, beautifully styled standalone HTML/PDF briefings for executive meetings.
Designed with zero system OS binary dependencies (no wkhtmltopdf, cairo, or pdfkit) to ensure
100% container-agnostic reliability on Streamlit Community Cloud.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional
import json
import pandas as pd
from src.logger_config import get_logger
from src.data_fetcher import get_currency_symbol

logger = get_logger(__name__)


class ExecutiveReportGenerator:
    """
    Generates standalone executive tear sheets in HTML format with inline styling,
    print-to-PDF optimization, and dynamic valuation highlight badges.
    """

    @staticmethod
    def generate_tear_sheet(
        ticker: str,
        ticker_data: pd.Series,
        ai_verdict: str,
        current_price: float
    ) -> str:
        """
        Compiles AI verdict and fundamental metrics into a self-contained HTML string.
        All user-facing text and labels are formatted in Czech as requested.
        """
        logger.info(f"Starting Executive Tear Sheet generation for ticker: {ticker}")
        start_time = time.perf_counter()
        
        try:
            company_name = ticker_data.get("Jméno", ticker_data.get("shortName", ticker_data.get("longName", ticker)))
            if pd.isna(company_name) or str(company_name).strip() in ["N/A", "None", ""]:
                company_name = ticker
            now_str = datetime.now().strftime("%d.%m.%Y %H:%M")
            
            currency_code = str(ticker_data.get("Měna", "USD")).strip()
            if pd.isna(currency_code) or currency_code in ["N/A", "None", ""]:
                currency_code = "USD"
            currency_symbol = get_currency_symbol(currency_code)
                
            # --- AI Verdict Formatting ---
            verdict_dict = {}
            if isinstance(ai_verdict, dict):
                verdict_dict = ai_verdict
            elif isinstance(ai_verdict, str):
                s_strip = ai_verdict.strip()
                if s_strip.startswith("{") and s_strip.endswith("}"):
                    try:
                        verdict_dict = json.loads(s_strip)
                    except Exception:
                        verdict_dict = {"verdict": s_strip, "fundamental_summary": s_strip, "business_moat": ""}
                else:
                    verdict_dict = {"verdict": s_strip, "fundamental_summary": s_strip, "business_moat": ""}
            else:
                verdict_dict = {"verdict": str(ai_verdict), "fundamental_summary": str(ai_verdict), "business_moat": ""}

            verdict_text = str(verdict_dict.get("verdict", "N/A")).upper()
            if "BUY" in verdict_text or "KUP" in verdict_text:
                verdict_badge_style = "background-color: #10b981; color: #ffffff;"
            elif "SELL" in verdict_text or "PRODEJ" in verdict_text:
                verdict_badge_style = "background-color: #ef4444; color: #ffffff;"
            elif "HOLD" in verdict_text or "DRŽ" in verdict_text:
                verdict_badge_style = "background-color: #f59e0b; color: #ffffff;"
            else:
                verdict_badge_style = "background-color: #6b7280; color: #ffffff;"
                
            fund_summary = verdict_dict.get("fundamental_summary", "Shrnutí fundamentů není k dispozici.")
            moat_summary = verdict_dict.get("business_moat", "Analýza konkurenčního příkopu není k dispozici.")
            
            # --- Key Metrics Processing ---
            roic_val = ticker_data.get("ROIC (%)", ticker_data.get("ROIC", "N/A"))
            pe_val = ticker_data.get("Forward P/E", ticker_data.get("P/E", "N/A"))
            margin_val = ticker_data.get("Provozní marže (%)", ticker_data.get("Provozní marže", "N/A"))
            de_val = ticker_data.get("Debt/Equity", ticker_data.get("Dluh/Vl.jmění", ticker_data.get("Dluh/Vl. jmění", "N/A")))
            
            def format_metric(val, is_pct=False, is_ratio=False):
                if val is None or val == "N/A":
                    return "N/A"
                try:
                    if pd.isna(val):
                        return "N/A"
                except Exception:
                    pass
                s_val = str(val).strip()
                if s_val in ["N/A", "nan", "NaN", "None", "", "<NA>"]:
                    return "N/A"
                try:
                    num = float(val)
                    if pd.isna(num) or num == 0.0:
                        return "N/A"
                    if is_pct:
                        return f"{num:.1f} %"
                    if is_ratio:
                        return f"{num:.2f}"
                    return f"{num:.2f}"
                except (ValueError, TypeError):
                    return s_val if s_val != "nan" else "N/A"

            roic_str = format_metric(roic_val, is_pct=True)
            pe_str = format_metric(pe_val)
            margin_str = format_metric(margin_val, is_pct=True)
            de_str = format_metric(de_val, is_ratio=True)

            # --- Self-Contained Styled HTML String ---
            html_content = f"""<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Tear Sheet - {ticker} ({company_name})</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --text-main: #0f172a;
            --text-muted: #475569;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --accent-color: #2563eb;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: #ffffff;
            padding: 32px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 4px solid var(--accent-color);
        }}
        .header-title h1 {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.025em;
            margin-bottom: 4px;
        }}
        .header-title p {{
            color: #94a3b8;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}
        .header-price {{
            text-align: right;
        }}
        .header-price .price {{
            font-size: 32px;
            font-weight: 800;
            color: #38bdf8;
        }}
        .header-price .currency {{
            font-size: 14px;
            color: #cbd5e1;
            display: block;
        }}
        .content {{
            padding: 40px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-color);
            display: flex;
            align-items: center;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 32px;
        }}
        .verdict-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }}
        .verdict-tag {{
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: 0.05em;
        }}
        .reasoning-block {{
            margin-bottom: 16px;
        }}
        .reasoning-block h4 {{
            font-size: 14px;
            text-transform: uppercase;
            color: var(--accent-color);
            margin-bottom: 6px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }}
        .reasoning-block p {{
            font-size: 15px;
            color: var(--text-muted);
            line-height: 1.7;
        }}
        .grid-table {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }}
        .metric-box {{
            background: #f1f5f9;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        .metric-box .label {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 4px;
            letter-spacing: 0.05em;
        }}
        .metric-box .val {{
            font-size: 22px;
            font-weight: 800;
            color: var(--text-main);
        }}
        .footer {{
            background: #f8fafc;
            padding: 20px 40px;
            border-top: 1px solid var(--border-color);
            font-size: 12px;
            color: #64748b;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        @media print {{
            body {{
                background-color: #ffffff !important;
                padding: 0 !important;
            }}
            .container {{
                box-shadow: none !important;
                border: none !important;
                max-width: 100% !important;
            }}
            .card, .val-badge, .metric-box {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <h1>{company_name}</h1>
                <p>EXECUTIVE TEAR SHEET &bull; TICKER: <strong>{ticker}</strong></p>
            </div>
            <div class="header-price">
                <span class="price">{current_price:,.2f}</span>
                <span class="currency">Měna: {currency_symbol} &bull; Aktuální kurz</span>
            </div>
        </div>
        
        <div class="content">
            <!-- 1. AI VERDICT & REASONING SECTION -->
            <div class="section-title">🤖 AI Strategický Verdikt (Llama 3.3 70B)</div>
            <div class="card">
                <div class="verdict-header">
                    <span style="font-weight: 700; color: var(--text-main);">Stanovisko AI analytika:</span>
                    <span class="verdict-tag" style="{verdict_badge_style}">{verdict_text}</span>
                </div>
                
                <div class="reasoning-block">
                    <h4>1. Shrnutí fundamentů (Fundamental Summary)</h4>
                    <p>{fund_summary}</p>
                </div>
                
                <div class="reasoning-block" style="margin-bottom: 0;">
                    <h4>2. Obchodní model a konkurenční příkop (Business Moat)</h4>
                    <p>{moat_summary}</p>
                </div>
            </div>

            <!-- 2. KEY METRICS TABLE -->
            <div class="section-title">📈 Klíčové finanční metriky (Key Financial Metrics)</div>
            <div class="grid-table">
                <div class="metric-box">
                    <div class="label">ROIC / Návratnost</div>
                    <div class="val">{roic_str}</div>
                </div>
                <div class="metric-box">
                    <div class="label">P/E Ratio (Valuace)</div>
                    <div class="val">{pe_str}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Provozní marže</div>
                    <div class="val">{margin_str}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Dluh / Vlastní jmění</div>
                    <div class="val">{de_str}</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div>Generováno systémem: <strong>Streamlit Quantitative & AI Financial Screener</strong></div>
            <div>Datum reportu: {now_str} &bull; Důvěrné pro exekutivní použití</div>
        </div>
    </div>
</body>
</html>
"""
            elapsed = time.perf_counter() - start_time
            logger.info(f"Successfully generated Executive Tear Sheet HTML for {ticker} in {elapsed:.4f}s.")
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to generate Executive Tear Sheet HTML for ticker {ticker}: {e}", exc_info=True)
            raise

    @staticmethod
    def generate_html_tearsheet(
        ticker: str,
        company_info: Dict[str, Any],
        dcf_result: Optional[Any],
        ai_verdict: Dict[str, Any],
        current_price: float,
        currency_symbol: str,
        key_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Backward-compatibility wrapper around generate_tear_sheet."""
        series_data = pd.Series(company_info) if isinstance(company_info, dict) else company_info
        if key_metrics:
            for k, v in key_metrics.items():
                series_data[k] = v
        verdict_str = json.dumps(ai_verdict) if isinstance(ai_verdict, dict) else str(ai_verdict)
        return ExecutiveReportGenerator.generate_tear_sheet(
            ticker=ticker,
            ticker_data=series_data,
            ai_verdict=verdict_str,
            current_price=current_price
        )


def generate_tear_sheet(
    ticker: str,
    ticker_data: pd.Series,
    ai_verdict: str,
    current_price: float,
    dcf_value: Optional[float] = None
) -> str:
    """Standalone module-level helper to generate the Executive Tear Sheet."""
    return ExecutiveReportGenerator.generate_tear_sheet(ticker, ticker_data, ai_verdict, current_price)


