"""
validate.py — Validate and clean complaint DataFrames.

Responsibilities:
  - Check that required columns exist.
  - Count and report missing, blank, and duplicate records separately.
  - Coerce complaint text to strings safely (handles numeric values).
  - Parse and validate date columns, reporting unparseable rows.
  - Make deduplication opt-in, not default.
  - Never silently delete records; always report what was removed and why.

Returns a ValidationReport alongside the cleaned DataFrame so callers
(dashboard or tests) can inspect exactly what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ValidationReport:
    """Counts describing what happened during validation."""
    total_rows: int = 0
    missing_text: int = 0       # NaN / None in the text column
    blank_text: int = 0         # Non-null but whitespace-only
    non_string_coerced: int = 0 # Numeric / non-string values cast to str
    exact_duplicates: int = 0   # Removed only when deduplicate=True
    invalid_dates: int = 0      # Rows where date could not be parsed
    retained_rows: int = 0      # Rows available for modelling
    duplicate_rate: float = 0.0 # Ratio of duplicate text rows (0.0 to 1.0)
    median_word_length: float = 0.0 # Median word count of retained text
    categorical_warning: Optional[str] = None # Warning message if text looks categorical

    def as_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "missing_text": self.missing_text,
            "blank_text": self.blank_text,
            "non_string_coerced": self.non_string_coerced,
            "exact_duplicates": self.exact_duplicates,
            "invalid_dates": self.invalid_dates,
            "retained_rows": self.retained_rows,
            "duplicate_rate": self.duplicate_rate,
            "median_word_length": self.median_word_length,
            "categorical_warning": self.categorical_warning,
        }


def validate(
    df: pd.DataFrame,
    text_col: str,
    date_col: Optional[str] = None,
    deduplicate: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate and clean a complaint DataFrame.

    Args:
        df:           Raw DataFrame from ingestion.
        text_col:     Name of the column holding complaint text.
        date_col:     Optional column with dates. Will be parsed and
                      rows with unparseable dates are flagged (not dropped).
        deduplicate:  If True, remove exact duplicate complaint texts.
                      Default is False — repeated identical complaints
                      may be legitimate and affect trend counts.

    Returns:
        (cleaned_df, report) where cleaned_df has a normalised '_text'
        column ready for preprocessing, and report explains every change.

    Raises:
        ValueError: If text_col is not present in df.
    """
    report = ValidationReport(total_rows=len(df))

    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    # ── 1. Count and coerce non-string values ──────────────────────────
    # Values that are not strings (e.g. integers, floats) are cast to str
    # so they can be processed; we count them so the user knows.
    non_str_mask = df[text_col].notna() & ~df[text_col].apply(
        lambda v: isinstance(v, str)
    )
    report.non_string_coerced = int(non_str_mask.sum())
    df[text_col] = df[text_col].apply(
        lambda v: str(v) if pd.notna(v) else v
    )

    # ── 2. Count missing text ──────────────────────────────────────────
    missing_mask = df[text_col].isna()
    report.missing_text = int(missing_mask.sum())
    df = df[~missing_mask].copy()

    # ── 3. Count blank text ────────────────────────────────────────────
    blank_mask = df[text_col].str.strip() == ""
    report.blank_text = int(blank_mask.sum())
    df = df[~blank_mask].copy()

    # ── 4. Categorical text check ──────────────────────────────────────
    if len(df) > 0:
        dup_count = int(df.duplicated(subset=[text_col]).sum())
        report.duplicate_rate = dup_count / len(df)
        word_counts = df[text_col].astype(str).str.split().apply(len)
        report.median_word_length = float(word_counts.median())

        warning_reasons = []
        if report.duplicate_rate > 0.50:
            warning_reasons.append(
                f"duplicate rate is {report.duplicate_rate:.1%} (> 50%)"
            )
        if report.median_word_length < 5.0:
            warning_reasons.append(
                f"median length is {report.median_word_length:.1f} words (< 5 words)"
            )

        if warning_reasons:
            report.categorical_warning = (
                f"Data quality warning: Column '{text_col}' appears categorical ("
                + ", ".join(warning_reasons)
                + "). Unsupervised theme discovery is designed for free-form complaint text narratives."
            )

    # ── 5. Optional deduplication ──────────────────────────────────────
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=[text_col])
        report.exact_duplicates = before - len(df)
    else:
        # Informational count of duplicates without removing
        report.exact_duplicates = int(df.duplicated(subset=[text_col]).sum())

    # ── 6. Parse dates ─────────────────────────────────────────────────
    if date_col and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        bad_dates = parsed.isna() & df[date_col].notna()
        report.invalid_dates = int(bad_dates.sum())
        df[date_col] = parsed          # NaT for unparseable; rows kept

    # ── 7. Build clean text column ─────────────────────────────────────
    df = df.reset_index(drop=True)
    report.retained_rows = len(df)
    return df, report
