import numpy as np
import pandas as pd

from src.data import TZ
from src.weather import FIXED_THRESHOLDS, attach_weather, daily_k, label_days


def _frame(ratios, seg="train", start="2018-07-01"):
    idx = pd.date_range(start, periods=96 * len(ratios), freq="15min", tz=TZ)
    df = pd.DataFrame(index=idx)
    df["date"] = idx.normalize()
    df["is_day"] = (idx.hour >= 6) & (idx.hour < 18)
    df["ghi_cs"] = np.where(df["is_day"], 500.0, 0.0)
    r = np.repeat(ratios, 96)
    df["nwp_globalirrad"] = df["ghi_cs"] * r
    df["lmd_totalirrad"] = 123.0
    df["power"] = 1.0
    df["segment"] = seg
    return df


def test_daily_k_and_fixed_labels():
    df = _frame([0.2, 0.5, 0.9])
    k = daily_k(df)
    assert np.allclose(k.to_numpy(), [0.2, 0.5, 0.9])
    assert list(label_days(k, FIXED_THRESHOLDS)) == ["overcast", "cloudy", "clear"]


def test_labels_use_only_nwp():
    rng = np.random.default_rng(0)
    df = _frame(rng.uniform(0.1, 1.0, 30))
    a, _ = attach_weather(df)
    df2 = df.copy()
    df2["lmd_totalirrad"] = rng.uniform(0, 1000, len(df2))
    df2["power"] = rng.uniform(0, 1, len(df2))
    b, _ = attach_weather(df2)
    for col in ["wclass", "wclass_fixed", "wclass_kmeans"]:
        assert a[col].equals(b[col])


def test_tercile_thresholds_balanced():
    df = _frame(np.linspace(0.1, 1.0, 30))
    out, thr = attach_weather(df)
    counts = out.groupby("date")["wclass"].first().value_counts()
    assert set(counts.index) == {"overcast", "cloudy", "clear"}
    assert counts.max() - counts.min() <= 1


def test_partial_nan_nwp_day_is_nan():
    df = _frame([0.9, 0.5, 0.2])
    day0 = df["date"] == df["date"].iloc[0]
    day_rows = (day0 & df["is_day"])
    idx = day_rows[day_rows].index[: day_rows.sum() // 2]
    df.loc[idx, "nwp_globalirrad"] = np.nan
    k = daily_k(df)
    assert np.isnan(k.iloc[0])
    assert np.allclose(k.iloc[1:].to_numpy(), [0.5, 0.2])


def test_thresholds_fit_on_pretest_only():
    train = _frame(np.linspace(0.1, 1.0, 30), "train")
    after = (pd.Timestamp("2018-07-01") + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    test = _frame(np.full(10, 0.05), "test", start=after)
    calib = _frame(np.full(10, 0.05), "calib", start=after)
    _, thr_train = attach_weather(train)
    # (a) extra test-segment days with extreme k must not change thresholds
    _, thr_with_test = attach_weather(pd.concat([train, test]))
    assert np.allclose(thr_with_test, thr_train)
    # (b) extra calib-segment days are pre-test and must change thresholds
    _, thr_with_calib = attach_weather(pd.concat([train, calib]))
    assert not np.allclose(thr_with_calib, thr_train)


def test_label_days_duplicate_thresholds_ok():
    out = label_days(pd.Series([0.4, 0.5, 0.6]), (0.5, 0.5))
    assert list(out) == ["overcast", "overcast", "clear"]
