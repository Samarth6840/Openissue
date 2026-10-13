import pytest
import pandas as pd
from openissue.ingest import load_csv_bytes, load_csv_string


def test_load_csv_string_valid():
    csv_text = "text,date\nBattery dies,2024-01-01\nScreen cracked,2024-01-02"
    df = load_csv_string(csv_text)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["text", "date"]


def test_load_csv_string_empty():
    with pytest.raises(ValueError, match="empty"):
        load_csv_string("")


def test_load_csv_bytes_valid_utf8():
    raw = b"text,category\nSlow app,Mobile\nLate delivery,Shipping"
    df = load_csv_bytes(raw)
    assert len(df) == 2
    assert "text" in df.columns


def test_load_csv_bytes_latin1_fallback():
    raw = "text,category\nCafé service issue,Store\n".encode("latin-1")
    df = load_csv_bytes(raw)
    assert len(df) == 1
    assert "Café" in df["text"].iloc[0]


def test_load_csv_bytes_empty():
    with pytest.raises(ValueError, match="empty"):
        load_csv_bytes(b"")


def test_load_csv_bytes_malformed():
    # If bytes cannot be decoded or parsed as csv table
    with pytest.raises(ValueError):
        load_csv_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd", encoding="utf-8")
