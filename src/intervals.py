"""Interval methods. Common interface: predict_day(hist, today) -> (lo, hi); update(today, lo, hi).
hist: daytime rows with non-missing y up to day D-1; today: daytime rows of day D."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.weather import CLASSES

P_CS_FLOOR = 0.05          # floor on clear-sky power (5% of capacity) so scores do not blow up near sunrise and sunset
A_MIN, A_MAX = 0.001, 0.5
HI_GUARD = 5.0             # guards only against infinite intervals (measured y peaks near 1.34); the earlier 2.0 truncated about 0.2-0.3% of interval widths at alpha=0.05, so it was raised to 5.0
K_MIN = 20                 # lower bound on the CACP neighbour count
CACP_COLS = ["y_lag1d", "y_lag1d_1", "y_lag1d_2", "y_lag1d_3", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "rho"]
# Day-ahead NWP weather vector. Closest PVOD analogues of CACP v2's DSWRF, VBDSF, TMP, PRES;
# PVOD has no cloud-cover forecast, so humidity is included instead.
NWP_WEATHER_COLS = ["nwp_globalirrad", "nwp_directirrad", "nwp_temperature", "nwp_pressure", "nwp_humidity"]
CACP_NWP_COLS = CACP_COLS + NWP_WEATHER_COLS
# Exploratory (post hoc) dilution check: weather vector + hour embedding only, no lag block / doy / rho.
CACP_WX_COLS = NWP_WEATHER_COLS + ["hour_sin", "hour_cos"]


def scores(f, normalize=True):
    e = (f["y"] - f["y_hat"]).abs().to_numpy()
    return e / np.maximum(f["p_cs"].to_numpy(), P_CS_FLOOR) if normalize else e


def conformal_quantile(s, alpha):
    s = np.asarray(s, float)
    s = s[~np.isnan(s)]
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))   # k-th smallest score; 1e-9 guards against floating-point error
    return np.inf if n == 0 or k > n else float(np.sort(s)[k - 1])


def to_interval(today, q, normalize=True):
    y_hat = today["y_hat"].to_numpy()
    half = q * (np.maximum(today["p_cs"].to_numpy(), P_CS_FLOOR) if normalize else np.ones_like(y_hat))
    return np.maximum(y_hat - half, 0), np.minimum(y_hat + half, HI_GUARD)


def day_error(today, lo, hi):
    y = today["y"].to_numpy()
    m = ~np.isnan(y)
    return None if not m.any() else float(np.mean((y[m] < lo[m]) | (y[m] > hi[m])))


class Method:
    def __init__(self, alpha, normalize=True):
        self.alpha, self.normalize = alpha, normalize

    def update(self, today, lo, hi):
        pass


class QR(Method):
    def predict_day(self, hist, today):
        return today[f"qr_lo_{self.alpha}"].to_numpy(), today[f"qr_hi_{self.alpha}"].to_numpy()


class SplitCP(Method):
    def predict_day(self, hist, today):
        q = conformal_quantile(scores(hist, self.normalize), self.alpha)
        return to_interval(today, q, self.normalize)


class ACI(Method):
    def __init__(self, alpha, gamma=0.005, normalize=True):
        super().__init__(alpha, normalize)
        self.gamma, self.a = gamma, alpha

    def predict_day(self, hist, today):
        return to_interval(today, conformal_quantile(scores(hist, self.normalize), self.a), self.normalize)

    def update(self, today, lo, hi):
        err = day_error(today, lo, hi)
        if err is not None:
            self.a = float(np.clip(self.a + self.gamma * (self.alpha - err), A_MIN, A_MAX))


class MondrianHour(Method):
    def __init__(self, alpha, n_min=30, normalize=True):
        super().__init__(alpha, normalize)
        self.n_min = n_min

    def predict_day(self, hist, today):
        s, h_hist = scores(hist, self.normalize), hist.index.hour.to_numpy()
        q_g = conformal_quantile(s, self.alpha)
        qmap = {}
        for h in np.unique(today.index.hour):
            sh = s[h_hist == h]
            qmap[h] = conformal_quantile(sh, self.alpha) if len(sh) >= self.n_min else q_g
        return to_interval(today, np.array([qmap[h] for h in today.index.hour]), self.normalize)


def cacp_k(k_frac, n):
    """CACP neighbour count: k_frac times the history size n, at least K_MIN and at most n."""
    return min(max(K_MIN, int(round(k_frac * n))), n)


class CACP(Method):
    """CACP-style KNN localisation applied to the shared normalised absolute-residual score:
    following Moradi et al. (arXiv:2510.15780), Eqs. 7, 8, 13 and 16 (KNN indicator weights), take the K calibration
    samples nearest in context and compute the conformal quantile of their scores.
    cols=CACP_COLS: no weather context (lag block + hour/day-of-year embedding + rho);
    cols=CACP_NWP_COLS: the same context plus the day-ahead NWP weather vector (NWP_WEATHER_COLS); the two differ only by weather.
    K is the fraction k_frac of the history size (cacp_k), so it grows with the expanding history."""

    def __init__(self, alpha, cols, k_frac=0.05, normalize=True):
        super().__init__(alpha, normalize)
        self.cols, self.k_frac = cols, k_frac

    def predict_day(self, hist, today):
        s = scores(hist, self.normalize)
        H = hist[self.cols].to_numpy(float)
        ok = ~np.isnan(H).any(axis=1)
        H, s = H[ok], s[ok]
        if len(H) == 0:
            return to_interval(today, np.inf, self.normalize)
        mu, sd = H.mean(axis=0), H.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        k = cacp_k(self.k_frac, len(H))
        nn_ = NearestNeighbors(n_neighbors=k).fit((H - mu) / sd)
        T = np.nan_to_num((today[self.cols].to_numpy(float) - mu) / sd)
        _, idx = nn_.kneighbors(T)
        q = np.array([conformal_quantile(s[i], self.alpha) for i in idx])
        return to_interval(today, q, self.normalize)


class WCACI(Method):
    """WC-ACI: separate calibration per weather class + per-class ACI + shrinkage to the global quantile for sparse classes."""

    def __init__(self, alpha, gamma=0.005, n_min_days=20, class_col="wclass",
                 use_aci=True, use_fallback=True, normalize=True):
        super().__init__(alpha, normalize)
        self.gamma, self.n_min_days, self.class_col = gamma, n_min_days, class_col
        self.use_aci, self.use_fallback = use_aci, use_fallback
        self.a = {c: alpha for c in CLASSES}

    def predict_day(self, hist, today):
        c = today[self.class_col].iloc[0]
        a = self.a.get(c, self.alpha)
        s = scores(hist, self.normalize)
        q_g = conformal_quantile(s, a)
        if c not in self.a:                     # NWP missing, no weather class
            return to_interval(today, q_g, self.normalize)
        m = (hist[self.class_col] == c).to_numpy()
        n_days = hist.loc[m, "date"].nunique()
        q_c = conformal_quantile(s[m], a)
        if n_days == 0 or not np.isfinite(q_c):
            q = q_g
        elif self.use_fallback:
            lam = min(1.0, n_days / self.n_min_days)
            q = lam * q_c + (1 - lam) * q_g
        else:
            q = q_c
        return to_interval(today, q, self.normalize)

    def update(self, today, lo, hi):
        c = today[self.class_col].iloc[0]
        err = day_error(today, lo, hi)
        if self.use_aci and c in self.a and err is not None:
            self.a[c] = float(np.clip(self.a[c] + self.gamma * (self.alpha - err), A_MIN, A_MAX))


def run_rolling(f, method, eval_segment="test", warmup_segments=("calib",), eval_days=None, window=None):
    """Daily rolling protocol: day D is calibrated only on data before D, and D joins the history after its intervals are issued.
    window=None is an expanding window; window=60 a 60-day sliding window. Returns y, lo, hi, wclass, date for daytime rows of evaluated days."""
    day = f[f["is_day"] & f["y_hat"].notna()]
    pool = day[day["segment"].isin(list(warmup_segments) + [eval_segment])]
    if eval_days is None:
        eval_days = sorted(pool.loc[pool["segment"] == eval_segment, "date"].unique())
    out = []
    for d in eval_days:
        hist = pool[(pool["date"] < d) & pool["y"].notna()]
        if window is not None:
            hist = hist[hist["date"] >= d - pd.Timedelta(days=window)]
        today = pool[pool["date"] == d]
        if today.empty:
            continue
        lo, hi = method.predict_day(hist, today)
        method.update(today, lo, hi)
        r = today[["y", "wclass", "date"]].copy()
        r["lo"], r["hi"] = lo, hi
        out.append(r)
    if not out:
        return pd.DataFrame(columns=["y", "wclass", "date", "lo", "hi"])
    return pd.concat(out)
