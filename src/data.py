"""PVOD loading, alignment and splitting. TZ / POWER_SCALE / column names were fixed after the M0 data inspection."""
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

TZ = "Asia/Shanghai"   # CSV timestamps are UTC; converted to China Standard Time (checked in M0)
POWER_SCALE = 1000.0   # power is in MW, metadata Capacity in kW (checked in M0)
NWP_COLS = ["nwp_globalirrad", "nwp_directirrad", "nwp_temperature", "nwp_humidity",
            "nwp_windspeed", "nwp_winddirection", "nwp_pressure"]
LMD_COLS = ["lmd_totalirrad", "lmd_diffuseirrad", "lmd_temperature", "lmd_pressure",
            "lmd_winddirection", "lmd_windspeed"]
SEGMENTS = ["train", "val", "calib", "test"]
FIXED_SPLIT = {
    "train": ("2000-01-01", "2018-12-01"),
    "val": ("2018-12-01", "2019-01-01"),
    "calib": ("2019-01-01", "2019-03-01"),
    "test": ("2019-03-01", "2019-06-14"),
}


def load_station(raw_dir, sid: int) -> pd.DataFrame:
    """Read one station and reindex it to a complete 15-min grid (96 rows per day)."""
    raw = pd.read_csv(Path(raw_dir) / f"station{sid:02d}.csv")
    tcol = "date_time" if "date_time" in raw.columns else "datetime"
    idx = pd.DatetimeIndex(pd.to_datetime(raw[tcol])).round("15min").tz_localize("UTC").tz_convert(TZ)
    df = raw.drop(columns=[tcol]).set_index(idx).sort_index(kind="stable")
    df = df[~df.index.duplicated()]
    start = df.index[0].normalize()
    end = df.index[-1].normalize() + pd.Timedelta("1D") - pd.Timedelta("15min")
    return df.reindex(pd.date_range(start, end, freq="15min", tz=TZ))


def add_clearsky(df: pd.DataFrame, lat: float, lon: float, capacity: float) -> pd.DataFrame:
    """Clear-sky irradiance, clear-sky power with capacity 1, daytime mask, normalised power y, date key."""
    loc = pvlib.location.Location(lat, lon, tz=TZ)
    cs = loc.get_clearsky(df.index, model="ineichen")
    zenith = loc.get_solarposition(df.index)["apparent_zenith"]
    out = df.copy()
    out["ghi_cs"] = cs["ghi"].to_numpy()
    out["p_cs"] = out["ghi_cs"] / 1000.0
    out["is_day"] = (zenith < 85).to_numpy()
    out["y"] = out["power"] * POWER_SCALE / capacity
    out["date"] = out.index.normalize()
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sine/cosine of hour and day of year, solar-day position rho, and measured y_lag1d at the same time on the previous day. y_lag1d assumes forecasts are issued at the end of day D-1."""
    out = df.copy()
    hour = out.index.hour + out.index.minute / 60
    doy = out.index.day_of_year
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    out["doy_sin"], out["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    day = out[out["is_day"]]
    pos = day.groupby("date").cumcount()
    cnt = day.groupby("date")["is_day"].transform("size")
    out["rho"] = np.nan
    out.loc[day.index, "rho"] = (pos / (cnt - 1).clip(lower=1)).to_numpy()
    out["y_lag1d"] = out["y"].shift(96)
    for j in (1, 2, 3):   # 15/30/45 min before the same time on the previous day (CACP lag-block context)
        out[f"y_lag1d_{j}"] = out["y"].shift(96 + j)
    return out


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Split by the calendar dates in FIXED_SPLIT; rows outside every period get segment None."""
    out = df.copy()
    out["segment"] = None
    for seg, (a, b) in FIXED_SPLIT.items():
        m = (out["date"] >= pd.Timestamp(a, tz=TZ)) & (out["date"] < pd.Timestamp(b, tz=TZ))
        out.loc[m, "segment"] = seg
    return out


def n_train_days(df: pd.DataFrame) -> int:
    return int(df.loc[df["segment"] == "train", "date"].nunique())
