"""M1: build data/processed/station{XX}.parquet, results/m1_class_counts.csv and results/m1_stations.json."""
import json
from pathlib import Path

import pandas as pd

from src.data import add_clearsky, add_time_features, assign_segments, load_station, n_train_days
from src.weather import attach_weather

RAW, OUT = Path("data/raw/PVODdatasets_v1.0"), Path("data/processed")
MIN_TRAIN_DAYS = 60
OUT.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(exist_ok=True)

meta = pd.read_csv(RAW / "metadata.csv")
counts, info = [], {}
for sid in range(len(meta)):
    m = meta.iloc[sid]
    assert m["Station_ID"] == f"station{sid:02d}", m["Station_ID"]
    df = load_station(RAW, sid)
    df = add_time_features(add_clearsky(df, float(m["Latitude"]), float(m["Longitude"]), float(m["Capacity"])))
    df = assign_segments(df)
    ntr = n_train_days(df)
    info[sid] = {"n_days": int(df["date"].nunique()), "n_train_days": int(ntr), "main_analysis": bool(ntr >= MIN_TRAIN_DAYS)}
    if not info[sid]["main_analysis"]:
        print(f"station{sid:02d}: only {ntr} training days; excluded from the main analysis, skipped")
        continue
    df, thr = attach_weather(df)
    info[sid]["thresholds"] = [round(float(t), 4) for t in thr]
    df.to_parquet(OUT / f"station{sid:02d}.parquet")
    c = df.groupby(["segment", "wclass"])["date"].nunique().rename("n_days").reset_index()
    c["station"] = sid
    counts.append(c)

allc = pd.concat(counts)
allc.to_csv("results/m1_class_counts.csv", index=False)
Path("results/m1_stations.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
print(json.dumps(info, indent=2, ensure_ascii=False))
print(allc.pivot_table(index="station", columns=["segment", "wclass"], values="n_days", fill_value=0).to_string())
