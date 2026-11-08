import numpy as np
import pandas as pd

import pytest

from src.intervals import (ACI, CACP, CACP_COLS, CACP_NWP_COLS, CACP_WX_COLS, HI_GUARD, MondrianHour,
                           NWP_WEATHER_COLS, SplitCP, WCACI, cacp_k, conformal_quantile, run_rolling, to_interval)
from src.weather import CLASSES


def add_lags(f, per_day=24):
    for j in range(4):
        f["y_lag1d" if j == 0 else f"y_lag1d_{j}"] = f["y"].shift(per_day + j)
    return f


def synth(n_days=300, per_day=24, sigma=None, seed=0, sigma_test=None):
    """iid synthetic data: y = 0.5 + N(0, sigma_c), sigma set per weather class; the first 60 days are the calibration warm-up.
    If sigma_test is given, all classes use that sigma in the test period (to create drift). Includes the context columns CACP needs."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days * per_day, freq="h", tz="UTC")
    wc = np.repeat(rng.choice(CLASSES, size=n_days), per_day)
    sigma = sigma or {c: 0.1 for c in CLASSES}
    sig = np.array([sigma[c] for c in wc])
    is_calib = np.repeat(np.arange(n_days) < 60, per_day)
    if sigma_test is not None:
        sig = np.where(is_calib, sig, sigma_test)
    seg = np.where(is_calib, "calib", "test")
    f = pd.DataFrame({"y": 0.5 + rng.normal(0, 1, len(idx)) * sig, "y_hat": 0.5, "p_cs": 1.0,
                      "is_day": True, "segment": seg, "date": idx.normalize(), "wclass": wc},
                     index=idx)
    hour, doy = idx.hour.to_numpy(), idx.day_of_year.to_numpy()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["rho"] = hour / 23
    for c in NWP_WEATHER_COLS:
        f[c] = np.repeat(rng.uniform(0, 1, n_days), per_day)
    return add_lags(f, per_day)


def covered(r):
    return ((r["y"] >= r["lo"]) & (r["y"] <= r["hi"]))


def test_conformal_quantile_small_n_is_inf():
    assert np.isinf(conformal_quantile([0.1, 0.2], alpha=0.1))
    assert conformal_quantile(np.arange(1, 101) / 100, alpha=0.1) == 0.91


def test_split_cp_marginal_coverage_iid():
    r = run_rolling(synth(), SplitCP(alpha=0.1))
    assert abs(covered(r).mean() - 0.9) <= 0.01


def test_wcaci_fixes_class_undercoverage():
    f = synth(sigma={"clear": 0.03, "cloudy": 0.15, "overcast": 0.08})
    base = run_rolling(f, SplitCP(alpha=0.1))
    ours = run_rolling(f, WCACI(alpha=0.1, gamma=0.005))
    assert covered(base)[base["wclass"] == "cloudy"].mean() < 0.85
    for c in CLASSES:
        assert abs(covered(ours)[ours["wclass"] == c].mean() - 0.9) <= 0.03


def test_to_interval_no_upper_clip():
    today = pd.DataFrame({"y_hat": [0.9], "p_cs": [1.0]})
    lo, hi = to_interval(today, 0.3)
    assert np.isclose(hi[0], 1.2) and np.isclose(lo[0], 0.6)
    assert to_interval(today, np.inf)[1][0] == HI_GUARD
    assert np.isclose(to_interval(today, 3.0)[1][0], 3.9)   # numeric guard only, far above observed y


def test_cacp_k_is_fraction_of_history():
    assert cacp_k(0.1, 5000) == 500
    assert cacp_k(0.01, 1000) == 20          # floor of 20 neighbours
    assert cacp_k(0.01, 15) == 15            # capped at the history size
    assert cacp_k(0.5, 1000) == 500
    assert cacp_k(1.0, 1000) == 1000         # global endpoint: all history


METHODS = {
    "split": lambda: SplitCP(alpha=0.1),
    "aci": lambda: ACI(alpha=0.1, gamma=0.005),
    "mondrian": lambda: MondrianHour(alpha=0.1),
    "wcaci": lambda: WCACI(alpha=0.1, gamma=0.005),
    "cacp": lambda: CACP(0.1, CACP_COLS, k_frac=0.05),
    "cacp_nwp": lambda: CACP(0.1, CACP_NWP_COLS, k_frac=0.05),
    "cacp_wx": lambda: CACP(0.1, CACP_WX_COLS, k_frac=0.05),
}


@pytest.mark.parametrize("name", list(METHODS))
def test_no_lookahead(name):
    f = synth()
    cut = f["date"].unique()[200]
    f2 = f.copy()
    f2.loc[f2["date"] >= cut, "y"] = 99.0
    add_lags(f2)                      # lag columns recomputed from the perturbed y (lags of future days change too)
    a, b = run_rolling(f, METHODS[name]()), run_rolling(f2, METHODS[name]())
    m = a["date"] <= cut          # intervals on day cut may use only data before cut
    assert np.allclose(a.loc[m, ["lo", "hi"]], b.loc[m, ["lo", "hi"]])


def test_aci_widens_under_drift():
    f = synth(sigma_test=0.15, sigma={c: 0.05 for c in CLASSES})
    aci = ACI(alpha=0.1, gamma=0.05)
    r_aci = run_rolling(f, aci)
    r_cp = run_rolling(f, SplitCP(alpha=0.1))
    assert aci.a < 0.1
    last = sorted(r_aci["date"].unique())[-100:]
    cov = lambda r: covered(r[r["date"].isin(last)]).mean()
    assert cov(r_aci) > cov(r_cp)


def test_cacp_runs_and_covers():
    r = run_rolling(synth(), CACP(0.1, CACP_COLS, k_frac=0.1))
    assert abs(covered(r).mean() - 0.9) <= 0.03


def test_run_rolling_empty():
    f = synth()
    f["segment"] = "calib"
    r = run_rolling(f, SplitCP(alpha=0.1))
    assert r.empty and list(r.columns) == ["y", "wclass", "date", "lo", "hi"]


def test_cacp_nwp_context_is_cacp_plus_weather():
    assert set(CACP_NWP_COLS) - set(CACP_COLS) == set(NWP_WEATHER_COLS)
    assert CACP_NWP_COLS[:len(CACP_COLS)] == CACP_COLS


def test_tune_ignores_test_segment():
    from src.run import tune
    f = synth(n_days=90)
    f2 = f.copy()
    test = f2["segment"] == "test"
    f2.loc[test, "y"] = f2.loc[test, "y"] * 3 + 1.0
    add_lags(f2)
    assert tune(f) == tune(f2)
