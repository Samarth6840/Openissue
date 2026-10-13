"""
preprocess.py — Text normalisation for complaint records.

Responsibilities:
  - Lowercase and strip whitespace.
  - Collapse repeated whitespace.
  - Preserve negation tokens (not, never, no) — important for complaints.
  - Preserve numeric terms (50%, 3 weeks) that carry issue-specific signal.
  - Apply normalisation to a pandas Series or a list of strings.

Does NOT vectorise text (see cluster.py).
"""

import re
import pandas as pd


# Regex: collapse internal whitespace to single space.
_WS = re.compile(r"\s+")

# Optional masked-token cleaning patterns (off by default; see clean_masked).
# Runs of 'x' used by CFPB to redact PII.
_MASK_RUN = re.compile(r"\bx{2,}\b", re.IGNORECASE)
# Date placeholders like XX/XX/XXXX or XX/XX/year>
_DATE_PLACEHOLDER = re.compile(r"\bxx/xx/(?:xxxx|year>)\b", re.IGNORECASE)
# Currency placeholders like {$40000.00}
_CURRENCY_PLACEHOLDER = re.compile(r"\{\$[\d.,]+\}", re.IGNORECASE)


def normalise_text(text: str, clean_masked: bool = False) -> str:
    """
    Normalise a single complaint string.

    Steps:
      1. Cast to str (safety for any stray non-string).
      2. Lowercase.
      3. Optionally collapse masked/redacted tokens (off by default):
         - runs of 'x' (XXXX) are removed
         - date placeholders (XX/XX/XXXX, XX/XX/year>) are removed
         - currency placeholders ({$40000.00}) become "money_amount"
      4. Strip leading/trailing whitespace.
      5. Collapse internal whitespace.

    Deliberately keeps (unless clean_masked=True):
      - Negation words (not, never, no, didn't …).
      - Percentages and numbers that describe the complaint.
      - Punctuation that may affect downstream TF-IDF token boundaries.

    Args:
        text: Raw complaint string.
        clean_masked: If True, apply the optional masked-token cleaning step.

    Returns:
        Normalised string. Empty string if input was blank after strip.
    """
    text = str(text).lower().strip()
    if clean_masked:
        text = _CURRENCY_PLACEHOLDER.sub(" money_amount ", text)
        text = _MASK_RUN.sub(" ", text)
        text = _DATE_PLACEHOLDER.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def normalise_series(series: pd.Series, clean_masked: bool = False) -> pd.Series:
    """
    Apply normalise_text to every element of a pandas Series.

    Args:
        series: Series of raw complaint strings.
        clean_masked: If True, apply the optional masked-token cleaning step.

    Returns:
        Series of normalised strings (same index).
    """
    return series.apply(lambda t: normalise_text(t, clean_masked=clean_masked))
