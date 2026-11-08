import numpy as np
import pandas as pd

from src.data import TZ, add_clearsky, add_time_features, assign_segments, load_station, n_train_days


def _day_frame(start, n_days):
    idx = pd.date_range(start, periods=96 * n_days, freq="15min", tz=TZ)
    return pd.DataFrame({"power": 1.0}, index=idx)


def test_clearsky_night_and_noon():
    df = add_clearsky(_day_frame("2018-07-01", 1), lat=39.0, lon=116.0, capacity=10000.0)
    assert not df.loc["2018-07-01 02:00", "is_day"]
    assert df.loc["2018-07-01 12:15", "is_day"]
    assert df.loc["2018-07-01 12:15", "p_cs"] > 0.5
    assert np.isclose(df["y"].iloc[0], 0.1)  # 1 MW -> 1000 kW / 10000 kW


def test_fixed_segments():
    df = add_clearsky(_day_frame("2018-07-01", 366), 39.0, 116.0, 10.0)
    seg = assign_segments(df)["segment"]
    assert seg.loc["2018-07-15 12:00"] == "train"
    assert seg.loc["2018-12-15 12:00"] == "val"
    assert seg.loc["2019-02-15 12:00"] == "calib"
    assert seg.loc["2019-04-15 12:00"] == "test"
    assert seg.loc["2019-06-13 12:00"] == "test"
    assert n_train_days(assign_segments(df)) == 153


def test_load_station_utc_to_local(tmp_path):
    pd.DataFrame({"date_time": ["2018-08-15 16:00:00", "2018-08-15 16:15:00"],
                  "power": [0.0, 0.1]}).to_csv(tmp_path / "station00.csv", index=False)
    df = load_station(tmp_path, 0)
    assert df.index[0] == pd.Timestamp("2018-08-16 00:00", tz="Asia/Shanghai")
    assert len(df) == 96


def test_time_features_and_lag():
    df = _day_frame("2018-07-01", 3)
    df["power"] = np.arange(len(df), dtype=float)
    df = add_time_features(add_clearsky(df, 39.0, 116.0, 10.0))
    assert df["y_lag1d"].iloc[:96].isna().all()
    assert np.allclose(df["y_lag1d"].iloc[96:].to_numpy(), df["y"].iloc[:-96].to_numpy())
    assert np.allclose(df["y_lag1d_2"].iloc[98:].to_numpy(), df["y"].iloc[:-98].to_numpy())
    assert df["y_lag1d_2"].iloc[:98].isna().all()
    day = df[df["is_day"] & (df["date"] == df["date"].iloc[0])]
    assert np.isclose(day["rho"].min(), 0.0) and np.isclose(day["rho"].max(), 1.0)
