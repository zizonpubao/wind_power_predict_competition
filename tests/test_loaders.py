"""Unit tests for the two non-obvious behaviors in src/data/loaders.py:
sum-not-mean SCADA aggregation, and the hour-end kst_dtm shift.
"""
import pandas as pd

from src.data.loaders import aggregate_scada_to_hourly, group_scada_power


def test_aggregate_scada_to_hourly_sums_and_shifts_to_hour_end():
    # 6 readings at HH:00..HH:50 for hour 2026-01-01 05:00-05:50, plus one more
    # hour's worth, for a single turbine column.
    times = pd.date_range("2026-01-01 05:00", periods=6, freq="10min")
    times2 = pd.date_range("2026-01-01 06:00", periods=6, freq="10min")
    df = pd.DataFrame(
        {
            "kst_dtm": list(times) + list(times2),
            "wtg01_power_kw10m": [100, 100, 100, 100, 100, 100] + [10, 20, 30, 40, 50, 60],
        }
    )

    out = aggregate_scada_to_hourly(df, ["wtg01_power_kw10m"])

    # sum, not mean: 6*100 = 600, not 100
    row0 = out[out["kst_dtm"] == pd.Timestamp("2026-01-01 06:00")]
    assert row0["wtg01_power_kw10m"].iloc[0] == 600

    # second hour: readings at 06:00..06:50 belong to the hour ending 07:00
    row1 = out[out["kst_dtm"] == pd.Timestamp("2026-01-01 07:00")]
    assert row1["wtg01_power_kw10m"].iloc[0] == 210

    # no row should be tagged with the interval-start timestamp
    assert (out["kst_dtm"] == pd.Timestamp("2026-01-01 05:00")).sum() == 0


def test_group_scada_power_sums_per_group_turbines_via_mapping():
    times = pd.date_range("2026-01-01 05:00", periods=6, freq="10min")

    vestas_cols = {f"vestas_wtg{i:02d}_power_kw10m": [10.0] * 6 for i in range(1, 13)}
    vestas_df = pd.DataFrame({"kst_dtm": times, **vestas_cols})

    unison_cols = {f"unison_wtg{i:02d}_power_kw10m": [5.0] * 6 for i in range(1, 6)}
    unison_df = pd.DataFrame({"kst_dtm": times, **unison_cols})

    out = group_scada_power(vestas_df, unison_df)
    row = out[out["kst_dtm"] == pd.Timestamp("2026-01-01 06:00")].iloc[0]

    # group_1 = wtg01-06 (6 turbines), group_2 = wtg07-12 (6 turbines): each
    # turbine's hourly sum is 6*10=60, so each group is 6*60=360.
    assert row["kpx_group_1"] == 360
    assert row["kpx_group_2"] == 360
    # group_3 = unison wtg01-05 (5 turbines): each turbine hourly sum 6*5=30,
    # group total 5*30=150.
    assert row["kpx_group_3"] == 150
