"""M2: cache point forecasts and quantile-regression intervals in results/forecasts/.
Usage: python -m scripts.run_forecasts [station ...] [--force]
Without station numbers: run all main-analysis stations and write results/m2_point_skill.csv; with station numbers: run only those (for parallel use) and write no summary.
Existing caches are reused by default; --force ignores them, retrains and overwrites."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasters import fit_predict_lgbm, fit_predict_lgbm_quantiles, fit_predict_nn

ALPHAS = [0.05, 0.1, 0.2]
SEEDS = {"lgbm": [0, 1, 2], "lstm": [0, 1, 2, 3, 4], "dlinear": [0, 1, 2, 3, 4]}
OUT = Path("results/forecasts")
OUT.mkdir(parents=True, exist_ok=True)

stations = {int(k): v for k, v in json.loads(Path("results/m1_stations.json").read_text()).items()
            if v["main_analysis"]}
ap = argparse.ArgumentParser()
ap.add_argument("stations", type=int, nargs="*")
ap.add_argument("--force", action="store_true", help="ignore existing caches, retrain and overwrite")
args = ap.parse_args()
only = args.stations
rows = []
for sid in only or stations:
    df = pd.read_parquet(f"data/processed/station{sid:02d}.parquet")
    for model, seeds in SEEDS.items():
        for seed in seeds:
            path = OUT / f"station{sid:02d}_{model}_seed{seed}.parquet"
            if path.exists() and not args.force:
                cache = pd.read_parquet(path)
            else:
                cache = pd.DataFrame(index=df.index)
                cache["y_hat"] = (fit_predict_lgbm(df, seed) if model == "lgbm"
                                  else fit_predict_nn(df, model, seed))
                if model == "lgbm":
                    for a in ALPHAS:
                        cache[f"qr_lo_{a}"], cache[f"qr_hi_{a}"] = fit_predict_lgbm_quantiles(df, a, seed)
                cache.to_parquet(path)
            m = (df["segment"] == "test") & df["is_day"] & df["y"].notna() & cache["y_hat"].notna()
            nrmse = float(np.sqrt(np.mean((df.loc[m, "y"] - cache.loc[m, "y_hat"]) ** 2)))
            rows.append({"station": sid, "model": model, "seed": seed, "test_day_nrmse": nrmse})
            print(sid, model, seed, round(nrmse, 4), flush=True)

if not only:
    pd.DataFrame(rows).to_csv("results/m2_point_skill.csv", index=False)
