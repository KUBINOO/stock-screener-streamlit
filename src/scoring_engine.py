"""
src/scoring_engine.py
Standalone module for computing normalized, weighted fundamental peer scores
(Quality, Value, Safety, Total) using fully vectorized Pandas operations.
"""

from typing import List
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from src.logger_config import get_logger

logger = get_logger(__name__)


def _clean_and_rank_metric(
    df: pd.DataFrame,
    candidate_cols: List[str],
    ascending: bool = True,
    positive_only: bool = False,
    non_negative_only: bool = False,
) -> pd.Series:
    """
    Extracts, cleans, imputes missing values with sector median, and ranks a metric
    vectorially into a 0 to 100 percentile score.

    Args:
        df: Input peer companies DataFrame.
        candidate_cols: List of potential column names for the metric.
        ascending: True if higher raw metric is better (e.g., ROIC),
                   False if lower raw metric is better (e.g., Forward P/E).
        positive_only: If True, values <= 0 are treated as NaN (e.g., negative P/E or EV/EBITDA).
        non_negative_only: If True, values < 0 are treated as NaN (e.g., negative Debt/Equity).

    Returns:
        pd.Series containing normalized scores in the [0.0, 100.0] range.
    """
    # Locate the first available column from candidate names
    selected_col = None
    for col in candidate_cols:
        if col in df.columns:
            selected_col = col
            break

    # If none of the candidate columns exist, return neutral 50.0 score for all rows
    if selected_col is None:
        logger.debug(f"No matching column found for candidates {candidate_cols}. Assigning neutral score 50.0.")
        return pd.Series(50.0, index=df.index, dtype=float)

    s = df[selected_col]

    # Clean object/string series vectorially (e.g., remove '%', '$', ',')
    if s.dtype == object or isinstance(s.dtype, pd.StringDtype):
        s = (
            s.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

    # Convert to numeric vectorially, coercing invalid values to NaN
    s = pd.to_numeric(s, errors="coerce")

    # Filter out invalid negative/zero values for valuation ratios or debt ratios
    if positive_only:
        s = s.where(s > 0, np.nan)
    if non_negative_only:
        s = s.where(s >= 0, np.nan)

    # Impute missing (NaN) raw values with the sector median vectorially
    median_val = s.median()
    s_imputed = s.fillna(median_val)

    # Convert raw metrics into a 0 to 1 percentile rank and scale to 0-100
    # Note: When ascending=False, lowest raw value receives the highest rank (1.0 -> 100.0)
    score = s_imputed.rank(pct=True, ascending=ascending) * 100.0

    # If all values were NaN (or median was NaN), fill remaining NaNs with neutral middle score 50.0
    score = score.fillna(50.0)

    return score.astype(float)


def calculate_peer_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes normalized, weighted fundamental scores (Quality, Value, Safety, Total)
    for a peer group of companies using fully vectorized Pandas operations.

    Args:
        df: Pandas DataFrame containing fundamental metrics as columns and Tickers as index
            (or 'Ticker' as a column).

    Returns:
        pd.DataFrame containing only calculated scores:
        ['Quality_Score', 'Value_Score', 'Safety_Score', 'Total_Score'] rounded to 1 decimal place.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame to calculate_peer_scores is empty or None.")
        return pd.DataFrame(
            columns=["Quality_Score", "Value_Score", "Safety_Score", "Total_Score"]
        )

    # Work on a copy and standardize index to Ticker if 'Ticker' is a column
    work_df = df.copy()
    if "Ticker" in work_df.columns and work_df.index.name != "Ticker":
        work_df = work_df.set_index("Ticker")

    # --- 1. Quality Category ---
    # Higher is better (ascending=True)
    roic_score = _clean_and_rank_metric(
        work_df,
        ["ROIC (%)", "ROIC", "returnOnEquity", "ROA (%)"],
        ascending=True,
    )
    op_margin_score = _clean_and_rank_metric(
        work_df,
        [
            "Provozní marže (%)",
            "Operating Margin (%)",
            "Operating Margin",
            "Operating Margins (%)",
            "Operating Margins",
            "operatingMargins",
            "Hrubá marže (%)",
            "Čistá marže (%)",
        ],
        ascending=True,
    )
    quality_score = (roic_score + op_margin_score) / 2.0

    # --- 2. Value Category ---
    # Lower is better (ascending=False -> lowest raw multiple gets highest rank)
    fwd_pe_score = _clean_and_rank_metric(
        work_df,
        ["Forward P/E", "Forward PE", "forwardPE", "P/E", "PE Ratio"],
        ascending=False,
        positive_only=True,
    )
    ev_ebitda_score = _clean_and_rank_metric(
        work_df,
        ["EV/EBITDA", "enterpriseToEbitda", "EV / EBITDA"],
        ascending=False,
        positive_only=True,
    )
    value_score = (fwd_pe_score + ev_ebitda_score) / 2.0

    # --- 3. Safety Category ---
    # Lower debt is better (ascending=False -> lowest debt/equity gets highest rank)
    safety_score = _clean_and_rank_metric(
        work_df,
        ["Debt/Equity", "Debt to Equity", "debtToEquity", "Celkový dluh"],
        ascending=False,
        non_negative_only=True,
    )

    # --- 4. Total Weighted Score ---
    # Weighted average: 40% Quality, 40% Value, 20% Safety
    total_score = (
        0.4 * quality_score
        + 0.4 * value_score
        + 0.2 * safety_score
    )

    # Construct clean output DataFrame containing strictly the required score columns
    scores_df = pd.DataFrame(
        {
            "Quality_Score": quality_score,
            "Value_Score": value_score,
            "Safety_Score": safety_score,
            "Total_Score": total_score,
        },
        index=work_df.index,
    )

    return scores_df.round(1)
