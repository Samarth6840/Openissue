"""
trends.py — Aggregate complaint theme counts over time and flag emerging issues.

Design decisions
----------------
* Weekly aggregation is the default. The period is configurable.
* Missing dates (NaT) are excluded from trend calculations with a clear count.
  We do NOT fill gaps with zero unless the period was actually observed in
  the data — zero-filling unobserved periods would misrepresent history.
* Emerging-issue flagging compares the most recent complete period with the
  previous one. This is a simple heuristic. A "doubling" is not a
  statistically significant claim; it is a review trigger.
* Every flag includes the counts behind it so a human can judge.

Responsibilities:
  - Aggregate theme counts by time period.
  - Report rows excluded due to missing dates.
  - Return a structured TrendResult.
  - Flag emerging themes with configurable thresholds.
  - Return a structured list of EmergingFlag with full evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class EmergingFlag:
    """One emerging-issue flag with the evidence behind it."""
    cluster_id: int
    top_terms: str
    last_period_count: int
    prev_period_count: int
    growth_factor: float             # last / prev (inf if prev == 0)
    last_period_label: str
    prev_period_label: str


@dataclass
class TrendResult:
    """Output of compute_trends."""
    pivot: pd.DataFrame = field(default_factory=pd.DataFrame)  # rows=week, cols=cluster_id, values=count
    rows_excluded_bad_date: int = 0  # rows dropped because date was NaT
    rows_used: int = 0
    period_freq: str = "W"           # pandas period alias


def compute_trends(
    df: pd.DataFrame,
    theme_col: str,
    date_col: str,
    freq: str = "W",
) -> TrendResult:
    """
    Aggregate complaint counts per theme per time period.

    Args:
        df:         DataFrame containing theme assignments and dates.
        theme_col:  Column name with cluster/theme IDs.
        date_col:   Column name with datetime values (may include NaT).
        freq:       Pandas period frequency string ("W", "M", "D").

    Returns:
        TrendResult.  pivot is empty if no valid dates exist.
    """
    result = TrendResult(period_freq=freq)
    df = df[[theme_col, date_col]].copy()

    # Convert to datetime if not already (errors → NaT).
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    bad_date_mask = df[date_col].isna()
    result.rows_excluded_bad_date = int(bad_date_mask.sum())
    df = df[~bad_date_mask].copy()
    result.rows_used = len(df)

    if df.empty:
        result.pivot = pd.DataFrame()
        return result

    df["_period"] = df[date_col].dt.to_period(freq).dt.start_time
    pivot = (
        df.groupby(["_period", theme_col])
        .size()
        .unstack(fill_value=0)
    )
    pivot.index.name = "period_start"
    result.pivot = pivot
    return result


def flag_emerging(
    trend: TrendResult,
    top_terms: dict[int, str],
    min_volume: int = 5,
    growth_factor: float = 2.0,
) -> list[EmergingFlag]:
    """
    Flag themes whose count in the last period meets or exceeds the growth
    threshold relative to the previous period.

    Rules:
      - last_count >= min_volume  (avoids flagging noise from tiny clusters)
      - last_count / prev_count >= growth_factor  OR  prev_count == 0
      - Requires at least two time periods.

    Args:
        trend:         TrendResult from compute_trends.
        top_terms:     Dict mapping cluster_id → top-terms string.
        min_volume:    Minimum complaint count in last period to flag.
        growth_factor: Ratio threshold (last / prev).

    Returns:
        List of EmergingFlag (may be empty). Each flag includes the counts.
    """
    pivot = trend.pivot
    if pivot.empty or len(pivot) < 2:
        return []

    last_row = pivot.iloc[-1]
    prev_row = pivot.iloc[-2]
    last_label = str(pivot.index[-1].date()) if hasattr(pivot.index[-1], "date") else str(pivot.index[-1])
    prev_label = str(pivot.index[-2].date()) if hasattr(pivot.index[-2], "date") else str(pivot.index[-2])

    flags: list[EmergingFlag] = []
    for col in pivot.columns:
        last_count = int(last_row[col])
        prev_count = int(prev_row[col])

        if last_count < min_volume:
            continue

        if prev_count == 0:
            gf = float("inf")
            flags.append(EmergingFlag(
                cluster_id=int(col),
                top_terms=top_terms.get(int(col), ""),
                last_period_count=last_count,
                prev_period_count=prev_count,
                growth_factor=gf,
                last_period_label=last_label,
                prev_period_label=prev_label,
            ))
        elif last_count / prev_count >= growth_factor:
            flags.append(EmergingFlag(
                cluster_id=int(col),
                top_terms=top_terms.get(int(col), ""),
                last_period_count=last_count,
                prev_period_count=prev_count,
                growth_factor=last_count / prev_count,
                last_period_label=last_label,
                prev_period_label=prev_label,
            ))

    return flags
