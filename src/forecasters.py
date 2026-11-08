"""Point forecasters: inputs are the day-D NWP and time features only, no measured lags. Output is capacity-normalised y_hat."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # macOS: multithreaded OpenMP in lightgbm+torch segfaults or deadlocks

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import NWP_COLS

# No day-of-year features: training covers Jul-Nov and testing Mar-Jun, so they would extrapolate badly at test time.
FEATURES = NWP_COLS + ["p_cs", "hour_sin", "hour_cos"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _rows(df, seg):
    return (df["segment"] == seg) & df["is_day"] & df["y"].notna() & df[NWP_COLS].notna().all(axis=1)


def _post(df, pred):
    out = np.maximum(np.asarray(pred, dtype=float), 0)  # clip below only: y can exceed 1
    out[~df["is_day"].to_numpy()] = 0.0
    out[df[NWP_COLS].isna().any(axis=1).to_numpy()] = np.nan
    return out


def _lgbm(seed, **kw):
    return lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=31, subsample=0.8,
                             subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, **kw)


def fit_predict_lgbm(df, seed):
    tr, va = _rows(df, "train"), _rows(df, "val")
    m = _lgbm(seed).fit(df.loc[tr, FEATURES], df.loc[tr, "y"],
                        eval_set=[(df.loc[va, FEATURES], df.loc[va, "y"])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    return _post(df, m.predict(df[FEATURES].fillna(0)))


def fit_predict_lgbm_quantiles(df, alpha, seed):
    """Quantile-regression reference: fits the alpha/2 and 1-alpha/2 quantiles separately."""
    tr, va = _rows(df, "train"), _rows(df, "val")
    preds = []
    for q in (alpha / 2, 1 - alpha / 2):
        m = _lgbm(seed, objective="quantile", alpha=q).fit(
            df.loc[tr, FEATURES], df.loc[tr, "y"],
            eval_set=[(df.loc[va, FEATURES], df.loc[va, "y"])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(_post(df, m.predict(df[FEATURES].fillna(0))))
    lo, hi = np.minimum(*preds), np.maximum(*preds)
    return lo, hi


class LSTMNet(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o).squeeze(-1)


class DLinearNet(nn.Module):
    """DLinear adapted to exogenous inputs: each NWP series is split into trend and remainder, mapped linearly along time, then mixed linearly across features."""

    def __init__(self, n_feat, length=96, kernel=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.lin_s, self.lin_t = nn.Linear(length, length), nn.Linear(length, length)
        self.mix = nn.Linear(n_feat, 1)

    def forward(self, x):                     # x: (B, 96, F)
        xt = x.transpose(1, 2)                # (B, F, 96)
        trend = self.avg(xt)
        z = self.lin_s(xt - trend) + self.lin_t(trend)
        return self.mix(z.transpose(1, 2)).squeeze(-1)


def _day_tensors(df, days, mu, sd):
    sub = df[df["date"].isin(days)].sort_index()
    x = ((sub[FEATURES] - mu) / sd).to_numpy(np.float32).reshape(-1, 96, len(FEATURES))
    y = sub["y"].to_numpy(np.float32).reshape(-1, 96)
    m = (sub["is_day"].to_numpy() & ~np.isnan(sub["y"].to_numpy())).reshape(-1, 96)
    return torch.tensor(x), torch.tensor(np.nan_to_num(y)), torch.tensor(m, dtype=torch.float32)


def fit_predict_nn(df, kind, seed, epochs=300, patience=10, lr=1e-3, batch=16):
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = df.groupby("date")
    full = (g.size() == 96) & (df[NWP_COLS].notna().all(axis=1).groupby(df["date"]).sum() == 96)
    ok_days = full[full].index
    seg_of = g["segment"].first()
    tr_days = [d for d in ok_days if seg_of[d] == "train"]
    va_days = [d for d in ok_days if seg_of[d] == "val"]
    tr_rows = df["date"].isin(tr_days)
    mu, sd = df.loc[tr_rows, FEATURES].mean(), df.loc[tr_rows, FEATURES].std() + 1e-6
    xtr, ytr, mtr = (t.to(DEVICE) for t in _day_tensors(df, tr_days, mu, sd))
    xva, yva, mva = (t.to(DEVICE) for t in _day_tensors(df, va_days, mu, sd))
    net = (LSTMNet if kind == "lstm" else DLinearNet)(len(FEATURES)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = lambda p, y, m: ((p - y) ** 2 * m).sum() / m.sum().clamp(min=1)
    best, best_state, bad, best_ep = np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        for i in torch.randperm(len(xtr)).split(batch):
            opt.zero_grad()
            loss_fn(net(xtr[i]), ytr[i], mtr[i]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(xva), yva, mva).item()
        if v < best:
            best, best_state, bad, best_ep = v, {k: t.clone() for k, t in net.state_dict().items()}, 0, ep + 1
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError("validation loss was never finite; training failed")
    print(f"  [{kind} seed{seed}] best epoch {best_ep}, stopped at {ep + 1}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    x_all, _, _ = _day_tensors(df, list(ok_days), mu, sd)
    with torch.no_grad():
        p = net(x_all.to(DEVICE)).cpu().numpy().ravel()
    pred = pd.Series(np.nan, index=df.index)
    pred[df[df["date"].isin(ok_days)].sort_index().index] = p
    return _post(df, pred.to_numpy())
