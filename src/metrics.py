"""Interval metrics. All power is normalised by installed capacity, so PINAW is the mean interval width."""
import numpy as np
import pandas as pd


def picp(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


def pinaw(lo, hi):
    return float(np.mean(hi - lo))


def winkler(y, lo, hi, alpha):
    w = hi - lo + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)
    return float(np.mean(w))


def summarize(r: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """r has columns y, lo, hi, wclass. Returns picp / pinaw / winkler / n for wclass='all' and for each weather class."""
    r = r.dropna(subset=["y", "lo", "hi"])
    groups = [("all", r)] + [(c, g) for c, g in r.groupby("wclass", observed=True)]
    rows = []
    for name, g in groups:
        y, lo, hi = g["y"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy()
        rows.append({"wclass": name, "picp": picp(y, lo, hi), "pinaw": pinaw(lo, hi),
                     "winkler": winkler(y, lo, hi, alpha), "n": len(g)})
    return pd.DataFrame(rows)
