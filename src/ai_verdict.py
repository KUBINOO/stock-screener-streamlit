"""
src/ai_verdict.py
Wrapper module for generating AI verdicts using the fault-tolerant AIGateway.
Direct Groq client initialization is deprecated in favor of AIGateway adapter pattern.
"""

from typing import Any, Dict
from src.ai_gateway import AIGateway
from src.logger_config import get_logger

logger = get_logger(__name__)


def get_ai_verdict(ticker: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper around AIGateway.get_verdict to maintain backwards compatibility
    with the rest of the application.
    """
    gateway = AIGateway()
    return gateway.get_verdict(ticker, company_data)
