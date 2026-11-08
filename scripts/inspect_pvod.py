"""M0: inspect the raw PVOD files (timestamps are UTC). Usage: python scripts/inspect_pvod.py data/raw/PVODdatasets_v1.0"""
import sys
from pathlib import Path

import pandas as pd

raw = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/PVODdatasets_v1.0")
Path("results").mkdir(exist_ok=True)
meta = pd.read_csv(raw / "metadata.csv")
rows = []
for i, p in enumerate(sorted(raw.glob("station*.csv"))):
    d = pd.read_csv(p)
    t = pd.to_datetime(d["date_time"])
    local = t.dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
    profile = d.groupby(local.dt.hour + local.dt.minute / 60)["power"].mean()
    cap_mw = float(meta["Capacity"].iloc[i]) / 1000
    rows.append({
        "file": p.name, "station_id_meta": meta["Station_ID"].iloc[i], "n_rows": len(d),
        "start_local": local.min(), "end_local": local.max(), "n_days": local.dt.normalize().nunique(),
        "peak_hour_local": profile.idxmax(), "capacity_mw": cap_mw, "max_power_mw": d["power"].max(),
        "max_over_capacity": d["power"].max() / cap_mw,
        **{f"na_{c}": d[c].isna().mean() for c in d.columns if c != "date_time"},
    })
out = pd.DataFrame(rows)
out.to_csv("results/m0_inspection.csv", index=False)
print(out[["file", "station_id_meta", "start_local", "end_local", "n_days", "peak_hour_local",
           "capacity_mw", "max_power_mw", "max_over_capacity"]].to_string())
print("max NaN fraction per column:", out.filter(like="na_").max().round(4).to_dict())
