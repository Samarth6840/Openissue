"""
ingest.py — Load complaint data from CSV files or raw strings.

Responsibilities:
  - Read CSV bytes/strings into a DataFrame.
  - Return clear error messages for malformed files.
  - Never silently swallow parse errors.

Does NOT perform validation or cleaning (see validate.py).
"""

import io
import pandas as pd


def load_csv_bytes(raw_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Parse raw CSV bytes into a DataFrame.

    Args:
        raw_bytes: Raw bytes from a file upload or disk read.
        encoding:  Encoding to try first; falls back to latin-1.

    Returns:
        pd.DataFrame with the parsed contents.

    Raises:
        ValueError: If the file cannot be parsed as CSV or is empty.
    """
    if not raw_bytes or not raw_bytes.strip():
        raise ValueError("The uploaded file is empty (zero bytes).")

    # Try the specified encoding, then fall back to latin-1 (never silently corrupt).
    for enc in (encoding, "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            "Could not decode the file. "
            "Ensure it is a UTF-8 or Latin-1 encoded CSV."
        )

    try:
        df = pd.read_csv(io.StringIO(text))
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty (zero rows).")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file is empty (zero rows).")

    if df.columns.tolist() == [0] or len(df.columns) == 0:
        raise ValueError(
            "The file has no recognisable column headers. "
            "Ensure the first row contains column names."
        )

    return df


def load_csv_string(csv_text: str) -> pd.DataFrame:
    """
    Parse a CSV string (e.g. an inline sample) into a DataFrame.

    Raises:
        ValueError: If the string cannot be parsed or is empty.
    """
    if not csv_text or not csv_text.strip():
        raise ValueError("CSV string is empty (zero rows).")

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except pd.errors.EmptyDataError:
        raise ValueError("CSV string is empty (zero rows).")
    except Exception as exc:
        raise ValueError(f"Could not parse CSV string: {exc}") from exc

    if df.empty:
        raise ValueError("CSV string produced an empty DataFrame.")

    return df
