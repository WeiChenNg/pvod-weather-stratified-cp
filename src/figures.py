"""Core figures and tables from results/daily.parquet. Usage: python -m src.figures"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.analysis import CLASSES, class_stats, day_table

MAIN = ["qr", "split_cp", "aci", "mondrian_hour", "cacp", "cacp_nwp", "cacp_wx", "wc_aci"]
ABL = ["wc_aci", "abl_no_norm", "abl_no_fallback", "abl_no_aci", "abl_fixed_thr", "abl_kmeans", "abl_sliding60"]
# Display names as used in the manuscript.
LABEL = {"qr": "QR (LightGBM only)", "split_cp": "Split CP", "aci": "ACI", "mondrian_hour": "Mondrian-hour",
         "cacp": "CACP-T", "cacp_nwp": "CACP-T+W", "cacp_wx": "CACP-W", "wc_aci": "WC-ACI"}
OUT = Path("results/figures")
ALPHA = 0.1


def stats_for(daily, methods):
    return pd.concat([class_stats(day_table(daily, m, ALPHA)).assign(method=m) for m in methods],
                     ignore_index=True)


def bar_by_class(st, metric, fname, target=None):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    w = 0.8 / len(MAIN)
    for i, m in enumerate(MAIN):
        g = st[st["method"] == m].set_index("wclass").loc[CLASSES]
        err = np.stack([g[metric] - g[f"{metric}_lo"], g[f"{metric}_hi"] - g[metric]])
        ax.bar(np.arange(3) + i * w, g[metric], w, yerr=err, label=LABEL.get(m, m), capsize=2)
    if target is not None:
        ax.axhline(target, color="k", ls="--", lw=1)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2, CLASSES)
    ax.set_ylabel(metric.upper())
    ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=300)
    fig.savefig((OUT / fname).with_suffix(".pdf"))  # vector copy for journal submission
    plt.close(fig)


def tradeoff(st, fname):
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    marks = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for ax, c in zip(axs, CLASSES):
        for i, m in enumerate(MAIN):
            g = st[(st["method"] == m) & (st["wclass"] == c)].iloc[0]
            ax.errorbar(g.pinaw, g.picp, xerr=[[g.pinaw - g.pinaw_lo], [g.pinaw_hi - g.pinaw]],
                        yerr=[[g.picp - g.picp_lo], [g.picp_hi - g.picp]], fmt=marks[i], ms=5, lw=0.8,
                        color=f"C{i}", label=LABEL.get(m, m))
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_title(c)
        ax.set_xlabel("PINAW")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))  # keep edge ticks of adjacent panels apart
    axs[0].set_ylabel("PICP")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / fname, dpi=300)
    fig.savefig((OUT / fname).with_suffix(".pdf"))  # vector copy for journal submission
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet("results/daily.parquet")
    st = stats_for(daily, list(dict.fromkeys(MAIN + ABL)))
    bar_by_class(st, "picp", "fig_coverage_by_class.png", target=0.9)
    bar_by_class(st, "pinaw", "fig_width_by_class.png")
    tradeoff(st, "fig_coverage_width_tradeoff.png")
    cols = ["method", "wclass", "picp", "picp_lo", "picp_hi", "pinaw", "pinaw_lo", "pinaw_hi", "winkler", "n"]
    for names, f in [(MAIN, "table_main.csv"), (ABL, "table_ablation.csv")]:
        t = st[st["method"].isin(names)][cols].copy()
        t["method"] = pd.Categorical(t["method"], names)
        t.sort_values(["method", "wclass"]).round(4).to_csv(OUT / f, index=False)


if __name__ == "__main__":
    main()
