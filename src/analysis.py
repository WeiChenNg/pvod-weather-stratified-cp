"""Shared analysis helpers: per-day table, pooled class statistics with date-clustered block-bootstrap CIs."""
import numpy as np
import pandas as pd

CLASSES = ["overcast", "cloudy", "clear"]
STRATA = ["meas_low", "meas_mid", "meas_high"]


def day_table(daily, method, alpha):
    """One row per (station, date) for one method/alpha, averaged over base models and seeds.

    cov = mean over runs of n_cov/n; n = mean n; width = mean over runs of sum_width/n;
    winkler = mean over runs of sum_winkler/n.
    wclass (NWP-based) and k_meas (measured clear-sky index) are constant within a day.
    """
    d = daily[(daily["method"] == method) & np.isclose(daily["alpha"], alpha)].copy()
    d["cov"] = d["n_cov"] / d["n"]
    d["width"] = d["sum_width"] / d["n"]
    d["winkler"] = d["sum_winkler"] / d["n"]
    g = d.groupby(["station", "date"], observed=True)
    out = g.agg(cov=("cov", "mean"), n=("n", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
                wclass=("wclass", "first"), k_meas=("k_meas", "first")).reset_index()
    return out


def meas_strata(dt):
    """Add column 'mstrat' in {meas_low, meas_mid, meas_high}: terciles of the MEASURED daily
    clear-sky index k_meas, computed per station over its test days. Evaluation-only
    stratification: it uses realised irradiance and must never feed any forecasting method."""
    dt = dt.copy()
    dt["mstrat"] = None
    for _, idx in dt.groupby("station").groups.items():
        k = dt.loc[idx, "k_meas"].dropna()
        if len(k) == 0:
            continue
        q = pd.qcut(k.rank(method="first"), 3, labels=STRATA)
        dt.loc[k.index, "mstrat"] = q.astype(str)
    return dt


def date_block_counts(n_dates, B=2000, seed=0, block_days=7):
    """Moving-block bootstrap over calendar dates. The n_dates sorted dates are split into consecutive
    non-overlapping blocks of `block_days` (the last may be shorter); each replicate draws blocks with
    replacement until it holds >= n_dates dates. Returns a (B, n_dates) matrix of draw counts."""
    starts = np.arange(0, n_dates, block_days)
    rng = np.random.default_rng(seed)
    c = np.zeros((B, n_dates))
    for b in range(B):
        total = 0
        while total < n_dates:
            s0 = starts[rng.integers(len(starts))]
            e = min(s0 + block_days, n_dates)
            c[b, s0:e] += 1
            total += e - s0
    return c


def _date_sums(dt, values, by, labels):
    """Per-label (n_dates, k) sums of `values` (columns of dt) over the rows of each sorted date."""
    pos, dates = pd.factorize(dt["date"], sort=True)
    out = {}
    for l in labels:
        m = np.ones(len(dt)) if l == "all" else (dt[by].to_numpy() == l).astype(float)
        v = np.zeros((len(dates), len(values)))
        np.add.at(v, pos, np.stack(values, axis=1) * m[:, None])
        out[l] = v
    return len(dates), out


def class_stats(dt, by="wclass", B=2000, seed=0, block_days=7):
    """Pooled PICP = sum(cov*n)/sum(n), PINAW = sum(width*n)/sum(n) (and Winkler likewise when dt has a
    'winkler' column) per group of `by`, plus an 'all' row. 95% percentile CI by a date-clustered moving
    block bootstrap: calendar dates are resampled jointly across stations in blocks of `block_days`
    consecutive dates (date_block_counts), and every (station, date) row of a drawn date enters with
    the date's multiplicity. B replicates, rng seeded with `seed`."""
    dt = dt.dropna(subset=[by]).reset_index(drop=True)
    labels = list(dict.fromkeys(dt[by])) + ["all"]
    has_w = "winkler" in dt.columns
    n = dt["n"].to_numpy(float)
    vals = [dt["cov"].to_numpy() * n, dt["width"].to_numpy() * n, n] + ([dt["winkler"].to_numpy() * n] if has_w else [])
    n_dates, sums = _date_sums(dt, vals, by, labels)
    C = date_block_counts(n_dates, B, seed, block_days)
    rows = []
    for l in labels:
        p, b = sums[l].sum(axis=0), C @ sums[l]
        r = {by: l, "n": p[2]}
        for j, name in [(0, "picp"), (1, "pinaw")] + ([(3, "winkler")] if has_w else []):
            bs = b[:, j] / b[:, 2]
            r.update({name: p[j] / p[2], f"{name}_lo": np.percentile(bs, 2.5), f"{name}_hi": np.percentile(bs, 97.5)})
        rows.append(r)
    cols = [by] + [c for k in ["picp", "pinaw"] for c in (k, f"{k}_lo", f"{k}_hi")] + ["n"] + \
        (["winkler", "winkler_lo", "winkler_hi"] if has_w else [])
    return pd.DataFrame(rows)[cols]


def paired_diff(dt_a, dt_b, by="wclass", B=2000, seed=0, block_days=7):
    """Pooled PICP and PINAW difference (a - b) per group of `by` plus 'all', with a PAIRED
    date-clustered moving-block bootstrap (as in class_stats): the same resampled dates are used for
    both methods. dt_a, dt_b are day tables (one row per (station, date)); only (station, date) rows
    present in both are used, and the grouping column is taken from dt_a."""
    key = ["station", "date"]
    m = dt_a.merge(dt_b[key + ["cov", "n", "width"]], on=key, suffixes=("", "_b"))
    m = m.dropna(subset=[by]).reset_index(drop=True)
    labels = list(dict.fromkeys(m[by])) + ["all"]
    vals = [(m["cov"] * m["n"]).to_numpy(), (m["width"] * m["n"]).to_numpy(), m["n"].to_numpy(float),
            (m["cov_b"] * m["n_b"]).to_numpy(), (m["width_b"] * m["n_b"]).to_numpy(), m["n_b"].to_numpy(float)]
    n_dates, sums = _date_sums(m, vals, by, labels)
    C = date_block_counts(n_dates, B, seed, block_days)
    f = lambda t: (t[..., 0] / t[..., 2] - t[..., 3] / t[..., 5], t[..., 1] / t[..., 2] - t[..., 4] / t[..., 5])
    rows = []
    for l in labels:
        (pp, pw), (bp, bw) = f(sums[l].sum(axis=0)), f(C @ sums[l])
        rows.append({by: l, "dpicp": pp, "dpicp_lo": np.percentile(bp, 2.5), "dpicp_hi": np.percentile(bp, 97.5),
                     "dpinaw": pw, "dpinaw_lo": np.percentile(bw, 2.5), "dpinaw_hi": np.percentile(bw, 97.5)})
    return pd.DataFrame(rows)
