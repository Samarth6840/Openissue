import pandas as pd
from openissue.preprocess import normalise_text, normalise_series


def test_normalise_text_basic():
    raw = "   App  Crashed   Unexpectedly!   "
    clean = normalise_text(raw)
    assert clean == "app crashed unexpectedly!"


def test_normalise_text_preserves_negations_and_numbers():
    raw = "Did NOT arrive in 3 days, 50% battery drop"
    clean = normalise_text(raw)
    assert "not" in clean
    assert "3" in clean
    assert "50%" in clean


def test_normalise_series():
    series = pd.Series(["  Slow response  ", "Device OVERHEATING "])
    result = normalise_series(series)
    assert result.iloc[0] == "slow response"
    assert result.iloc[1] == "device overheating"


def test_clean_masked_currency():
    raw = "refund of a payment ( {$40000.00} )"
    clean = normalise_text(raw, clean_masked=True)
    assert "money_amount" in clean
    assert "{$" not in clean


def test_clean_masked_date_and_xxxx():
    raw = "In XX/XX/year>, my email account"
    clean = normalise_text(raw, clean_masked=True)
    assert "xx/xx" not in clean
    assert clean == "in, my email account" or clean == "in my email account" or "my email account" in clean


def test_clean_masked_xxxx_runs():
    raw = "debt relief program with XXXX XXXX XXXX and paid {$7100.00}"
    clean = normalise_text(raw, clean_masked=True)
    assert "xxxx" not in clean
    assert "money_amount" in clean

def test_clean_masked_off_by_default():
    raw = "debt relief program with XXXX XXXX XXXX"
    clean = normalise_text(raw)
    assert "xxxx" in clean
