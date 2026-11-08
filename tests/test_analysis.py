import numpy as np
import pandas as pd

from src.analysis import class_stats, day_table, meas_strata, paired_diff


def _dt():
    rng = np.random.default_rng(1)
    rows = []
    for s in (0, 1):
        for i in range(12):
            rows.append(dict(station=s, date=i, cov=rng.uniform(0.6, 1), n=rng.integers(30, 45),
                             width=rng.uniform(0.1, 0.3), wclass=["clear", "cloudy", "overcast"][i % 3],
                             k_meas=rng.uniform(0.1, 1)))
    return pd.DataFrame(rows)


def test_pooled_arithmetic():
    dt = pd.DataFrame(dict(station=[0, 0, 1], date=[0, 1, 0], cov=[1.0, 0.5, 0.0], n=[10, 20, 10],
                           width=[1.0, 2.0, 3.0], wclass=["a", "a", "a"], k_meas=[1, 2, 3]))
    r = class_stats(dt, B=50).set_index("wclass")
    assert np.isclose(r.loc["a", "picp"], (10 + 10 + 0) / 40)
    assert np.isclose(r.loc["a", "pinaw"], (10 + 40 + 30) / 40)
    assert np.isclose(r.loc["all", "picp"], r.loc["a", "picp"])


def test_ci_contains_point_and_deterministic():
    dt = _dt()
    a, b = class_stats(dt, B=300), class_stats(dt, B=300)
    pd.testing.assert_frame_equal(a, b)
    assert ((a.picp_lo <= a.picp) & (a.picp <= a.picp_hi)).all()
    assert ((a.pinaw_lo <= a.pinaw) & (a.pinaw <= a.pinaw_hi)).all()


def test_meas_strata_balanced():
    out = meas_strata(_dt())
    assert (out.groupby(["station", "mstrat"]).size() == 4).all()
    lo = out[out.mstrat == "meas_low"].groupby("station")["k_meas"].max()
    hi = out[out.mstrat == "meas_high"].groupby("station")["k_meas"].min()
    assert (lo < hi).all()


def test_day_table_collapses_runs():
    d = pd.DataFrame(dict(station=[0, 0], date=[1, 1], wclass=["clear"] * 2, k_meas=[.9, .9],
                          n=[10, 10], n_cov=[10, 5], sum_width=[10., 30.], sum_winkler=[0., 0.],
                          base_model=["a", "b"], seed=[0, 0], method=["m"] * 2, alpha=[0.1] * 2))
    r = day_table(d, "m", 0.1).iloc[0]
    assert np.isclose(r["cov"], 0.75) and np.isclose(r["width"], 2.0)


def test_paired_diff_identical_is_zero():
    dt = _dt()
    d = paired_diff(dt, dt, B=200)
    assert np.allclose(d[["dpicp", "dpicp_lo", "dpicp_hi", "dpinaw", "dpinaw_lo", "dpinaw_hi"]].to_numpy(), 0)


def test_paired_diff_constant_shift():
    a = _dt()
    b = a.assign(cov=a["cov"] - 0.1, width=a["width"] + 0.05)
    d = paired_diff(a, b, B=200).set_index("wclass")
    assert np.allclose(d["dpicp"], 0.1) and np.allclose(d["dpicp_lo"], 0.1) and np.allclose(d["dpicp_hi"], 0.1)
    assert np.allclose(d["dpinaw"], -0.05)


def test_dates_drawn_jointly_across_stations():
    # Station 1's coverage is the complement of station 0's on each date (equal n), and it varies by
    # date: every resample pools to exactly 0.5 only if both stations' rows of a date move together.
    c = np.random.default_rng(3).uniform(0, 1, 30)
    dt = pd.DataFrame([dict(station=s, date=d, cov=c[d] if s == 0 else 1 - c[d], n=10, width=0.2,
                            wclass="a", k_meas=0.5) for s in (0, 1) for d in range(30)])
    r = class_stats(dt, B=300).set_index("wclass")
    assert np.isclose(r.loc["all", "picp_lo"], 0.5) and np.isclose(r.loc["all", "picp_hi"], 0.5)


def test_consecutive_date_blocks():
    # cov alternates 1, 0 on consecutive dates: every 2-day block pools to 0.5, single days do not.
    dt = pd.DataFrame(dict(station=0, date=range(20), cov=[float(i % 2) for i in range(20)], n=10,
                           width=0.2, wclass="a", k_meas=0.5))
    r = class_stats(dt, B=300, block_days=2).set_index("wclass")
    assert np.isclose(r.loc["all", "picp_lo"], 0.5) and np.isclose(r.loc["all", "picp_hi"], 0.5)
    r1 = class_stats(dt, B=300, block_days=1).set_index("wclass")
    assert r1.loc["all", "picp_hi"] - r1.loc["all", "picp_lo"] > 0.1


def test_block_counts_shape():
    from src.analysis import date_block_counts
    c = date_block_counts(15, B=500, seed=0, block_days=7)   # blocks of 7, 7, 1 dates
    assert c.shape == (500, 15)
    assert (c.sum(axis=1) >= 15).all() and (c.sum(axis=1) < 15 + 7).all()
    assert (c[:, :7] == c[:, [0]]).all() and (c[:, 7:14] == c[:, [7]]).all()
    assert np.array_equal(c, date_block_counts(15, B=500, seed=0, block_days=7))


def test_paired_diff_deterministic():
    a = _dt()
    b = a.assign(cov=np.random.default_rng(5).uniform(0.6, 1, len(a)))
    pd.testing.assert_frame_equal(paired_diff(a, b, B=200), paired_diff(a, b, B=200))


def test_winkler_pooled_when_present():
    dt = pd.DataFrame(dict(station=[0, 0], date=[0, 1], cov=[1.0, 0.5], n=[10, 30], width=[1.0, 2.0],
                           winkler=[1.0, 3.0], wclass=["a", "a"], k_meas=[1, 2]))
    r = class_stats(dt, B=50).set_index("wclass")
    assert np.isclose(r.loc["all", "winkler"], (10 + 90) / 40)
