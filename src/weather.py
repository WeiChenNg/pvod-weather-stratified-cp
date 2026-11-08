"""Weather classes: computed only from NWP forecast irradiance and clear-sky irradiance; measured columns must never be used."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

CLASSES = ["overcast", "cloudy", "clear"]
FIXED_THRESHOLDS = (0.35, 0.65)


def daily_k(df: pd.DataFrame) -> pd.Series:
    """k_D = sum of daytime NWP irradiance / sum of daytime clear-sky irradiance; NaN if any daytime NWP value is missing or the clear-sky sum is 0."""
    day = df[df["is_day"]]
    g = day.groupby("date")
    complete = g["nwp_globalirrad"].apply(lambda s: s.notna().all())
    den = g["ghi_cs"].sum()
    k = g["nwp_globalirrad"].sum() / den.where(den > 0)
    return k.where(complete)


def label_days(k: pd.Series, thresholds) -> pd.Series:
    t0, t1 = thresholds
    lab = np.where(k <= t0, "overcast", np.where(k <= t1, "cloudy", "clear")).astype(object)
    return pd.Series(np.where(k.isna(), np.nan, lab), index=k.index, dtype=object)


def _kmeans_labels(k: pd.Series, k_fit: pd.Series) -> pd.Series:
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(k_fit.to_numpy().reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())
    rank = {c: CLASSES[i] for i, c in enumerate(order)}
    out = pd.Series(np.nan, index=k.index, dtype=object)
    ok = k.notna()
    out[ok] = [rank[c] for c in km.predict(k[ok].to_numpy().reshape(-1, 1))]
    return out


def attach_weather(df: pd.DataFrame):
    """Add k_nwp, wclass (tercile thresholds), wclass_fixed and wclass_kmeans. Returns (frame, tercile thresholds).
    Tercile thresholds and k-means are fitted only on all pre-test days (train+val+calib); the test segment never enters the fit. Days with NaN k_D get NaN in all three labels."""
    k = daily_k(df)
    fit_days = df.loc[df["segment"].isin(["train", "val", "calib"]), "date"].unique()
    k_fit = k[k.index.isin(fit_days)].dropna()
    assert k_fit.nunique() >= 3, "fewer than 3 distinct valid pre-test k_D values; cannot fit three weather classes"
    thr = tuple(np.quantile(k_fit, [1 / 3, 2 / 3]))
    out = df.copy()
    out["k_nwp"] = out["date"].map(k)
    out["wclass"] = out["date"].map(label_days(k, thr))
    out["wclass_fixed"] = out["date"].map(label_days(k, FIXED_THRESHOLDS))
    out["wclass_kmeans"] = out["date"].map(_kmeans_labels(k, k_fit))
    return out, thr
