import pytest
import pandas as pd
import numpy as np
from openissue.validate import validate, ValidationReport


def test_validate_missing_column():
    df = pd.DataFrame({"other_col": [1, 2]})
    with pytest.raises(ValueError, match="not found"):
        validate(df, text_col="complaint_text")


def test_validate_missing_and_blank_text():
    df = pd.DataFrame({
        "complaint": ["Valid complaint", None, "   ", np.nan, "Another complaint"]
    })
    cleaned, report = validate(df, text_col="complaint")
    assert len(cleaned) == 2
    assert report.total_rows == 5
    assert report.missing_text == 2
    assert report.blank_text == 1
    assert report.retained_rows == 2


def test_validate_numeric_and_non_string():
    df = pd.DataFrame({
        "complaint": [12345, 99.5, "App froze", True]
    })
    cleaned, report = validate(df, text_col="complaint")
    assert report.non_string_coerced == 3
    assert len(cleaned) == 4
    assert cleaned["complaint"].iloc[0] == "12345"
    assert cleaned["complaint"].iloc[1] == "99.5"
    assert cleaned["complaint"].iloc[3] == "True"


def test_validate_duplicate_handling_off():
    df = pd.DataFrame({
        "complaint": ["Same issue", "Same issue", "Different issue"]
    })
    cleaned, report = validate(df, text_col="complaint", deduplicate=False)
    # Deduplication is off: duplicates are retained but counted for awareness
    assert len(cleaned) == 3
    assert report.exact_duplicates == 1
    assert report.retained_rows == 3


def test_validate_duplicate_handling_on():
    df = pd.DataFrame({
        "complaint": ["Same issue", "Same issue", "Different issue"]
    })
    cleaned, report = validate(df, text_col="complaint", deduplicate=True)
    assert len(cleaned) == 2
    assert report.exact_duplicates == 1
    assert report.retained_rows == 2


def test_validate_invalid_dates():
    df = pd.DataFrame({
        "complaint": ["Issue 1", "Issue 2", "Issue 3"],
        "date": ["2024-01-01", "not-a-date", "2024-01-03"]
    })
    cleaned, report = validate(df, text_col="complaint", date_col="date")
    # All 3 complaints are retained
    assert len(cleaned) == 3
    assert report.invalid_dates == 1
    assert pd.isna(cleaned["date"].iloc[1])
    assert pd.notna(cleaned["date"].iloc[0])


def test_export_consistency():
    df = pd.DataFrame({
        "complaint": ["Battery drain", "App crashed"],
        "date": ["2024-01-01", "2024-01-02"]
    })
    cleaned, report = validate(df, text_col="complaint", date_col="date")
    cleaned["theme_id"] = [0, 1]
    cleaned["theme_label"] = ["Theme 0: battery", "Theme 1: app"]
    
    export_cols = ["complaint", "theme_id", "theme_label", "date"]
    csv_bytes = cleaned[export_cols].to_csv(index=False)
    assert "theme_id" in csv_bytes
    assert "Theme 0: battery" in csv_bytes
    assert "2024-01-01" in csv_bytes


def test_validate_categorical_text_warning_using_issue_column():
    df = pd.DataFrame({
        "Issue": [
            "Incorrect information on report",
            "Incorrect information on report",
            "Incorrect information on report",
            "Improper use of report",
            "Improper use of report",
            "Improper use of report",
            "Debt collection",
        ],
        "date": ["2024-01-01"] * 7,
    })
    cleaned, report = validate(df, text_col="Issue", date_col="date")
    assert report.duplicate_rate > 0.50
    assert report.median_word_length < 5.0
    assert report.categorical_warning is not None
    assert "appears categorical" in report.categorical_warning
