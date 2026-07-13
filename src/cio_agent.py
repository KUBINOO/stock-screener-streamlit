"""
src/cio_agent.py
Qualitative Chief Investment Officer (CIO) Agent layer.
Synthesizes deterministic scoring engine outputs and simulates an institutional
investment committee debate (Moat, Safety, Valuation agents) to deliver actionable
investment recommendations in Czech.
"""

import json
from typing import Any, Dict
import pandas as pd
from src.ai_gateway import AIGateway
from src.logger_config import get_logger

logger = get_logger(__name__)

CIO_SYSTEM_PROMPT = """You are a seasoned Chief Investment Officer (CIO) leading an institutional investment committee at a premier quantitative asset management firm.
You are tasked with synthesizing quantitative scores and fundamental peer metrics into a definitive, institutional-grade comparative investment report.

### INVESTMENT COMMITTEE SIMULATION (MULTI-AGENT DEBATE)
In your internal reasoning, you must simulate an intensive, evidence-based debate among three specialized investment committee agents:
1. **Moat Agent (Quality Analyst)**: Advocates for companies with durable competitive moats (network effects, brand loyalty, high switching costs, scale economies) as reflected in top Quality_Score rankings (ROIC & Operating Margins).
2. **Valuation Agent (Value Analyst)**: Scrutinizes earnings multiples and pricing discipline based on Value_Score rankings (Forward P/E & EV/EBITDA multiples), identifying undervalued opportunities and warning against overpaying for growth.
3. **Safety Agent (Risk & Resilience Officer)**: Evaluates balance sheet resilience, debt leverage, and financial stability based on the Safety_Score (where a HIGHER score means lower debt, lower risk, and GREATER safety).

### STRICT OUTPUT INSTRUCTIONS
- All written textual content inside the JSON values MUST be written in professional, grammatically polished Czech language (čeština).
- You MUST output ONLY a valid JSON object matching the exact schema below. Do not include markdown fences, conversational filler, or intro/outro text outside the JSON object.

### MANDATORY JSON OUTPUT SCHEMA
{
  "internal_reasoning": "Detailed simulated debate between Moat, Safety, and Valuation agents weighing the quantitative scores and trade-offs. MUST be at least 200 words of rigorous analysis.",
  "moat_analysis": "Qualitative comparison of the companies' economic moats (Network effects, brand strength, pricing power, switching costs, and ROIC durability) in Czech.",
  "risk_analysis": "Comprehensive identification of key regulatory, competitive, balance-sheet, and execution risks among the peers in Czech, factoring in their Safety_Score.",
  "cio_verdict": "Final definitive ranking and investment recommendation explaining clear causal reasons WHY the winning ticker was selected over peers in Czech."
}"""


def _format_scored_df(scored_df: pd.DataFrame) -> str:
    """
    Formats the deterministic peer scoring DataFrame into a clean Markdown table string
    for LLM prompt context injection.
    """
    if scored_df is None or scored_df.empty:
        return "No peer scoring data provided."

    try:
        return scored_df.to_markdown()
    except Exception:
        # Fallback to standard string representation if tabulate is unavailable
        return scored_df.to_string()


def generate_cio_report(scored_df: pd.DataFrame, gateway: AIGateway) -> Dict[str, Any]:
    """
    Executes the qualitative CIO Agent layer over the deterministic peer scoring DataFrame.

    Args:
        scored_df: DataFrame containing Quality_Score, Value_Score, Safety_Score, Total_Score
                   for the peer group.
        gateway: Instance of AIGateway used to interact with LLM providers.

    Returns:
        Dict matching the enforced JSON schema:
        {
          "internal_reasoning": str,
          "moat_analysis": str,
          "risk_analysis": str,
          "cio_verdict": str
        }
    """
    logger.info("Initiating qualitative CIO Agent report generation.")

    if scored_df is None or scored_df.empty:
        logger.warning("Scored DataFrame passed to generate_cio_report is empty.")
        return {
            "internal_reasoning": "Vstupní data pro investiční komisi nejsou k dispozici.",
            "moat_analysis": "Analýza konkurenční výhody nemohla být provedena z důvodu chybějících dat.",
            "risk_analysis": "Hodnocení rizik nebylo provedeno z důvodu chybějících dat.",
            "cio_verdict": "CHYBĚJÍCÍ DATA: Pro vydání doporučení prosím načtěte data o konkurenci.",
        }

    formatted_table = _format_scored_df(scored_df)
    context_payload = f"""### DETERMINISTIC PEER SCORING MATRIX (0-100 Percentile Scores)
{formatted_table}

Analyze the peer group table above. Conduct the simulated investment committee debate and generate the final CIO verdict in strict JSON format."""

    try:
        raw_response = gateway.get_verdict(CIO_SYSTEM_PROMPT, context_payload)
    except Exception as e:
        logger.error(f"Error calling AIGateway inside generate_cio_report: {e}", exc_info=True)
        raw_response = {}

    # Guarantee all schema fields are present with high-quality fallbacks if needed
    report: Dict[str, Any] = {
        "internal_reasoning": raw_response.get(
            "internal_reasoning",
            "Simulace debaty investiční komise neproběhla (chyba API nebo nedostupnost LLM služby).",
        ),
        "moat_analysis": raw_response.get(
            "moat_analysis",
            "Kvalitativní srovnání konkurenčních výhod (ekonomického příkopu) není momentálně k dispozici.",
        ),
        "risk_analysis": raw_response.get(
            "risk_analysis",
            "Identifikace klíčových rizik a rozvahy není momentálně k dispozici.",
        ),
        "cio_verdict": raw_response.get(
            "cio_verdict",
            "Závěrečné doporučení CIO se nepodařilo vygenerovat. Zkontrolujte prosím připojení k AI bráně.",
        ),
    }

    logger.info("Successfully generated qualitative CIO Agent report.")
    return report
