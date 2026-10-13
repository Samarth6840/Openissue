import pandas as pd
from openissue.trends import compute_trends, flag_emerging, EmergingFlag, TrendResult


def test_compute_trends_weekly():
    df = pd.DataFrame({
        "theme_id": [0, 0, 1, 1, 0],
        "created_at": [
            "2024-01-05",  # Week 1
            "2024-01-06",  # Week 1
            "2024-01-07",  # Week 1
            "2024-01-15",  # Week 3
            "2024-01-16",  # Week 3
        ]
    })
    trend = compute_trends(df, theme_col="theme_id", date_col="created_at", freq="W")
    assert isinstance(trend, TrendResult)
    assert trend.rows_used == 5
    assert trend.rows_excluded_bad_date == 0
    assert not trend.pivot.empty
    assert 0 in trend.pivot.columns
    assert 1 in trend.pivot.columns


def test_compute_trends_unparseable_dates():
    df = pd.DataFrame({
        "theme_id": [0, 1, 2],
        "date": ["2024-01-05", "invalid-date", None]
    })
    trend = compute_trends(df, theme_col="theme_id", date_col="date", freq="W")
    assert trend.rows_used == 1
    assert trend.rows_excluded_bad_date == 2


def test_flag_emerging_growth_and_volume():
    # 2 weeks of data
    df = pd.DataFrame({
        "theme_id": [0] * 2 + [1] * 1 + [0] * 6 + [1] * 2,
        "date": (
            ["2024-01-01"] * 3 +  # Week 1: theme 0 = 2, theme 1 = 1
            ["2024-01-08"] * 8    # Week 2: theme 0 = 6 (3x, vol 6), theme 1 = 2 (2x, vol 2)
        )
    })
    trend = compute_trends(df, theme_col="theme_id", date_col="date", freq="W")
    top_terms = {0: "battery, drain", 1: "screen, crack"}

    # With min_volume=5, only theme 0 should flag (count=6 >= 5, growth=3.0 >= 2.0)
    flags = flag_emerging(trend, top_terms=top_terms, min_volume=5, growth_factor=2.0)
    assert len(flags) == 1
    flag = flags[0]
    assert flag.cluster_id == 0
    assert flag.last_period_count == 6
    assert flag.prev_period_count == 2
    assert flag.growth_factor == 3.0
    assert "battery" in flag.top_terms


def test_flag_emerging_single_period():
    df = pd.DataFrame({
        "theme_id": [0, 1],
        "date": ["2024-01-01", "2024-01-02"]
    })
    trend = compute_trends(df, theme_col="theme_id", date_col="date", freq="W")
    flags = flag_emerging(trend, top_terms={}, min_volume=1, growth_factor=1.5)
    # Only 1 period, cannot establish a comparison
    assert len(flags) == 0


def test_flag_emerging_synthetic_spike():
    # Synthetic spike: Theme 0 has 2 complaints in week 1, 10 complaints in week 2 (5x spike)
    # Theme 1 has 5 complaints in week 1, 5 complaints in week 2 (flat 1x)
    df = pd.DataFrame({
        "theme_id": [0] * 2 + [1] * 5 + [0] * 10 + [1] * 5,
        "date": (
            ["2024-01-01"] * 7 +   # Week 1: theme 0 = 2, theme 1 = 5
            ["2024-01-08"] * 15    # Week 2: theme 0 = 10, theme 1 = 5
        ),
    })
    trend = compute_trends(df, theme_col="theme_id", date_col="date", freq="W")
    top_terms = {0: "rapid, battery, drain", 1: "billing, overcharge"}

    flags = flag_emerging(trend, top_terms=top_terms, min_volume=5, growth_factor=2.0)
    assert len(flags) == 1
    flag = flags[0]
    assert flag.cluster_id == 0
    assert flag.last_period_count == 10
    assert flag.prev_period_count == 2
    assert flag.growth_factor == 5.0
    assert "battery" in flag.top_terms
