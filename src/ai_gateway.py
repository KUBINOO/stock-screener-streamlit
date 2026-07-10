"""
src/ai_gateway.py
Fault-Tolerant AIGateway using the Adapter design pattern.
Handles LLM rate limits (HTTP 429/503) and failures by falling back across free-tier providers
(Groq -> Google Gemini) while enforcing strict JSON output schemas.
"""

import os
import json
import time
from typing import Any, Dict, Tuple, Union

# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from groq import Groq
try:
    # pyrefly: ignore [missing-import]
    import google.generativeai as genai
except ImportError:
    genai = None

from src.logger_config import get_logger

logger = get_logger(__name__)


class AIGateway:
    """
    Gateway class managing multi-provider LLM calls via isolated provider adapters.
    Ensures uninterrupted JSON verdict generation with graceful failover.
    """

    def __init__(self) -> None:
        self.groq_api_key = self._get_secret_or_env("GROQ_API_KEY")
        self.gemini_api_key = self._get_secret_or_env("GEMINI_API_KEY")

        self.groq_client = None
        if self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)

        if self.gemini_api_key and genai:
            try:
                genai.configure(api_key=self.gemini_api_key)
            except Exception as e:
                logger.error(f"Failed to configure Google GenAI SDK: {e}", exc_info=True)

    @staticmethod
    def _get_secret_or_env(key_name: str) -> Union[str, None]:
        """
        Securely retrieves an API key from Streamlit secrets or OS environment variables.
        """
        try:
            if hasattr(st, "secrets") and key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
        return os.environ.get(key_name)

    @staticmethod
    def _build_prompts(ticker: str, company_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Constructs the exact same instructions and schema expectations for all model providers.
        Returns (system_prompt, user_prompt).
        """
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

        return system_prompt, user_prompt

    @staticmethod
    def _normalize_verdict(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes and validates the JSON output payload returned by any adapter.
        """
        return {
            "fundamental_summary": parsed_data.get(
                "fundamental_summary", "Shrnutí fundamentů není k dispozici."
            ),
            "business_moat": parsed_data.get(
                "business_moat", "Analýza obchodního modelu není k dispozici."
            ),
            "verdict": parsed_data.get("verdict", "WAIT FOR BETTER PRICE"),
            "internal_reasoning": parsed_data.get("internal_reasoning", ""),
            "error": None,
        }

    def _call_groq(self, prompt: str, context: str) -> Dict[str, Any]:
        """
        Private adapter method for executing completion requests via Groq API.
        Requires strict JSON output mode.
        """
        if not self.groq_client:
            raise RuntimeError("Groq API key not found or client initialization failed.")

        model_name = "openai/gpt-oss-120b"
        logger.info(f"Initiating Groq API call using model {model_name}.")
        start_time = time.perf_counter()

        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context},
            ],
            temperature=0.0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - start_time
        logger.info(f"Groq API call completed successfully in {elapsed:.4f}s.")

        parsed_data = json.loads(raw_content.strip())
        return self._normalize_verdict(parsed_data)

    def _call_gemini(self, prompt: str, context: str) -> Dict[str, Any]:
        """
        Private adapter method for executing completion requests via Google GenAI SDK (model gemini-1.5-flash).
        Enforces strict JSON response MIME type.
        """
        if not self.gemini_api_key or genai is None:
            raise RuntimeError("GEMINI_API_KEY not found or SDK not installed/configured.")

        model_name = "gemini-1.5-flash"
        logger.info(f"Initiating Gemini API call using model {model_name}.")
        start_time = time.perf_counter()

        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0,
        )

        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=prompt,
                generation_config=generation_config,
            )
            response = model.generate_content(context)
        except TypeError:
            # Fallback for SDK versions where system_instruction is passed differently
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=generation_config,
            )
            response = model.generate_content(f"{prompt}\n\n{context}")

        raw_content = response.text or ""
        elapsed = time.perf_counter() - start_time
        logger.info(f"Gemini API call completed successfully in {elapsed:.4f}s.")

        parsed_data = json.loads(raw_content.strip())
        return self._normalize_verdict(parsed_data)

    def get_verdict(
        self,
        ticker_or_prompt: str,
        company_data_or_context: Union[Dict[str, Any], str] = "",
    ) -> Dict[str, Any]:
        """
        Public entry point for verdict generation with fallback routing across free-tier providers.
        Can accept either (ticker: str, company_data: dict) or (prompt: str, context: str).
        """
        if isinstance(company_data_or_context, dict):
            prompt, context = self._build_prompts(ticker_or_prompt, company_data_or_context)
        else:
            prompt = ticker_or_prompt
            context = str(company_data_or_context)

        # Attempt 1: Try calling _call_groq()
        try:
            return self._call_groq(prompt, context)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "503" in error_msg:
                logger.warning(
                    f"Groq failure ({error_msg}). Initiating Tier 2 Fallback to Gemini"
                )
            else:
                logger.warning(
                    f"Groq failure encountered ({error_msg}). Initiating Tier 2 Fallback to Gemini"
                )

        # Attempt 2: Try calling _call_gemini()
        try:
            return self._call_gemini(prompt, context)
        except Exception as e:
            logger.critical(
                f"Critical error during AI verdict generation after Tier 2 fallback failure: {e}",
                exc_info=True,
            )
            return {
                "internal_reasoning": "API Timeout",
                "fundamental_summary": "Data nedostupná",
                "business_moat": "Data nedostupná",
                "verdict": "WAIT FOR BETTER PRICE",
                "error": None,
            }
