"""Tuning (second half of the calibration warm-up) and batch run of all interval methods.
Usage: python -m src.run [--stations 0 1] [--quick] [--merge]
With --stations: write only results/daily/station{XX}.parquet and the tuned-parameter files (run in parallel processes).
Without arguments or with --merge: merge into results/daily.parquet, results/results.csv, results/tuned_params.csv.
--quick: first seed of each model only; all output goes under results/quick/ and never overwrites or merges into the main results."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.intervals import ACI, CACP, CACP_COLS, CACP_NWP_COLS, CACP_WX_COLS, QR, WCACI, MondrianHour, SplitCP, run_rolling
from src.metrics import summarize, winkler

ALPHAS = [0.05, 0.1, 0.2]
SEEDS = {"lgbm": [0, 1, 2], "lstm": [0, 1, 2, 3, 4], "dlinear": [0, 1, 2, 3, 4]}
GAMMAS = [0.005, 0.01, 0.05, 0.1]   # ACI updates once per day, hence a coarser step grid
K_FRACS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]   # CACP K = k_frac x history size (see intervals.cacp_k)
KEYS = ["station", "base_model", "seed", "method", "alpha"]
RESULTS = Path("results")
QUICK = RESULTS / "quick"

METHODS = {
    "split_cp": (lambda a, t: SplitCP(a), None),
    "aci": (lambda a, t: ACI(a, gamma=t["aci_gamma"]), None),
    "mondrian_hour": (lambda a, t: MondrianHour(a), None),
    "cacp": (lambda a, t: CACP(a, CACP_COLS, k_frac=t["cacp_kfrac"]), None),
    "cacp_nwp": (lambda a, t: CACP(a, CACP_NWP_COLS, k_frac=t["cacp_nwp_kfrac"]), None),
    "cacp_wx": (lambda a, t: CACP(a, CACP_WX_COLS, k_frac=t["cacp_wx_kfrac"]), None),
    "wc_aci": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"]), None),
    "abl_no_norm": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"], normalize=False), None),
    "abl_no_fallback": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"], use_fallback=False), None),
    "abl_no_aci": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"], use_aci=False), None),
    "abl_fixed_thr": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"], class_col="wclass_fixed"), None),
    "abl_kmeans": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"], class_col="wclass_kmeans"), None),
    "abl_sliding60": (lambda a, t: WCACI(a, gamma=t["wc_aci_gamma"]), 60),
}


def _ws(r, alpha):
    r = r.dropna(subset=["y"])
    return winkler(r["y"].to_numpy(), r["lo"].to_numpy(), r["hi"].to_numpy(), alpha)


def tune(f, alpha=0.1):
    """Tune within the calibration warm-up: first half as history, second half for scoring; the test set is never touched."""
    calib_days = sorted(f.loc[f["segment"] == "calib", "date"].unique())
    tune_days = calib_days[len(calib_days) // 2:]
    run = lambda m: _ws(run_rolling(f, m, eval_segment="calib", warmup_segments=(), eval_days=tune_days), alpha)
    return {
        "aci_gamma": min(GAMMAS, key=lambda g: run(ACI(alpha, gamma=g))),
        "wc_aci_gamma": min(GAMMAS, key=lambda g: run(WCACI(alpha, gamma=g))),
        "cacp_kfrac": min(K_FRACS, key=lambda k: run(CACP(alpha, CACP_COLS, k_frac=k))),
        "cacp_nwp_kfrac": min(K_FRACS, key=lambda k: run(CACP(alpha, CACP_NWP_COLS, k_frac=k))),
        "cacp_wx_kfrac": min(K_FRACS, key=lambda k: run(CACP(alpha, CACP_WX_COLS, k_frac=k))),
    }


def k_meas_by_day(base):
    """Measured clear-sky index: used only for evaluation, never as input to any method."""
    day = base[base["is_day"]]
    g = day.groupby("date")
    num, den = g["lmd_totalirrad"].sum(), g["ghi_cs"].sum()
    complete = g["lmd_totalirrad"].apply(lambda s: s.notna().all())
    return (num / den.where(den > 0)).where(complete)


def daily_rows(r, alpha, kmeas):
    r = r.dropna(subset=["y", "lo", "hi"])
    y, lo, hi = r["y"].to_numpy(), r["lo"].to_numpy(), r["hi"].to_numpy()
    d = pd.DataFrame({
        "date": r["date"].to_numpy(), "wclass": r["wclass"].to_numpy(),
        "cov": ((y >= lo) & (y <= hi)).astype(float), "w": hi - lo,
        "wk": hi - lo + (2 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))})
    d["wclass"] = d["wclass"].astype(object)
    out = d.groupby("date").agg(wclass=("wclass", "first"), n=("cov", "size"), n_cov=("cov", "sum"),
                                sum_width=("w", "sum"), sum_winkler=("wk", "sum")).reset_index()
    out["n_cov"] = out["n_cov"].astype(int)
    out["k_meas"] = out["date"].map(kmeas)
    return out


def aggregate(daily):
    """Daily table -> summary per (run keys, wclass) plus an 'all' row. Days with NaN wclass count only towards 'all'."""
    s = ["n", "n_cov", "sum_width", "sum_winkler"]
    a = daily.groupby(KEYS, observed=True)[s].sum().reset_index().assign(wclass="all")
    c = daily.dropna(subset=["wclass"]).groupby(KEYS + ["wclass"], observed=True)[s].sum().reset_index()
    g = pd.concat([a, c], ignore_index=True)
    g["picp"], g["pinaw"] = g["n_cov"] / g["n"], g["sum_width"] / g["n"]
    g["winkler"] = g["sum_winkler"] / g["n"]
    return g


def merge(info, root=RESULTS):
    files = sorted((root / "daily").glob("station*.parquet"))
    daily = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    daily.to_parquet(root / "daily.parquet")
    res = aggregate(daily)
    res["main_analysis"] = res["station"].map(lambda s: info[int(s)]["main_analysis"])
    cols = ["wclass", "picp", "pinaw", "winkler", "n"] + KEYS + ["main_analysis"]
    res[cols].to_csv(root / "results.csv", index=False)
    tf = sorted((root / "daily").glob("tuned_station*.csv"))
    pd.concat([pd.read_csv(p) for p in tf], ignore_index=True).to_csv(root / "tuned_params.csv", index=False)
    print("merged into", root, len(files), "stations;", len(daily), "daily rows;", len(res), "result rows")


def run_station(sid, quick, root=RESULTS):
    t0 = time.time()
    base = pd.read_parquet(f"data/processed/station{sid:02d}.parquet")
    kmeas = k_meas_by_day(base)
    parts, tuned_log, checked = [], [], False
    for model, seeds in SEEDS.items():
        for seed in (seeds[:1] if quick else seeds):
            cache = pd.read_parquet(f"results/forecasts/station{sid:02d}_{model}_seed{seed}.parquet")
            f = base.join(cache)
            tuned = tune(f)
            tuned_log.append({"station": sid, "model": model, "seed": seed, **tuned})
            todo = dict(METHODS)
            if model == "lgbm":
                todo["qr"] = (lambda a, t: QR(a), None)
            for name, (make, window) in todo.items():
                for alpha in ALPHAS:
                    r = run_rolling(f, make(alpha, tuned), window=window)
                    d = daily_rows(r, alpha, kmeas).assign(
                        station=sid, base_model=model, seed=seed, method=name, alpha=alpha)
                    if not checked:   # daily-table aggregation must match metrics.summarize
                        s = summarize(r, alpha).set_index("wclass")
                        g = aggregate(d).set_index("wclass")
                        for c in s.index:
                            for m in ["picp", "pinaw", "winkler"]:
                                assert np.isclose(s.loc[c, m], g.loc[c, m]), (c, m)
                            assert s.loc[c, "n"] == g.loc[c, "n"]
                        checked = True
                    parts.append(d)
            print("done", sid, model, seed, tuned, f"{time.time() - t0:.0f}s", flush=True)
    daily = root / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(daily / f"station{sid:02d}.parquet")
    pd.DataFrame(tuned_log).to_csv(daily / f"tuned_station{sid:02d}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", type=int, nargs="*")
    ap.add_argument("--quick", action="store_true", help="first seed of each model only, as a smoke test; output goes to results/quick/")
    ap.add_argument("--merge", action="store_true", help="only merge existing per-station outputs")
    args = ap.parse_args()
    info = {int(k): v for k, v in json.loads(Path("results/m1_stations.json").read_text()).items()}
    main_st = [s for s, v in info.items() if v["main_analysis"]]
    root = QUICK if args.quick else RESULTS
    if not args.merge:
        for sid in (args.stations if args.stations else main_st):
            run_station(sid, args.quick, root)
    if args.merge or not args.stations:
        merge(info, root)


if __name__ == "__main__":
    main()
