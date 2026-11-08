"""Decisions 19 and 20 at alpha=0.1 (main-analysis stations). Prints and writes results/m3_m4_verdict.md."""
import numpy as np
import pandas as pd

from src.analysis import CLASSES, STRATA, class_stats, day_table, meas_strata, paired_diff

ALPHA = 0.1
METHODS = ["qr", "split_cp", "aci", "mondrian_hour", "cacp", "cacp_nwp", "cacp_wx", "wc_aci"]
LINES = []


def out(s=""):
    print(s)
    LINES.append(s)


def md(df, fmt="{:.3f}"):
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    out("| " + " | ".join(cols) + " |")
    out("|" + "---|" * len(cols))
    for i, row in df.iterrows():
        out("| " + " | ".join([str(i)] + [v if isinstance(v, str) else fmt.format(v) for v in row]) + " |")
    out()


LABEL = {"qr": "qr (LightGBM only)"}
lab = lambda m: LABEL.get(m, m)
BOOT = ("95% percentile CIs from a date-clustered moving-block bootstrap: calendar test dates are "
        "resampled jointly across stations in non-overlapping blocks of 7 consecutive dates, B=2000, seed 0")


def cell(x, g):
    """'PICP [CI] / PINAW [CI] / Winkler' for group g of a class_stats table."""
    return (f"{x.loc[g,'picp']:.3f} [{x.loc[g,'picp_lo']:.3f}, {x.loc[g,'picp_hi']:.3f}] / "
            f"{x.loc[g,'pinaw']:.3f} [{x.loc[g,'pinaw_lo']:.3f}, {x.loc[g,'pinaw_hi']:.3f}] / "
            f"{x.loc[g,'winkler']:.3f}")


CELL_HDR = "PICP [CI] / PINAW [CI] / Winkler"


def test_days(daily):
    out("## Evaluated test days per station")
    out()
    d = daily[(daily["method"] == "split_cp") & np.isclose(daily["alpha"], ALPHA)]
    d = d.drop_duplicates(["station", "date"])
    t = d.groupby("station").agg(days=("date", "nunique")).join(
        d.assign(wclass=d["wclass"].fillna("no class (NWP missing)"))
        .groupby(["station", "wclass"])["date"].nunique().unstack(fill_value=0))
    cols = ["days"] + [c for c in CLASSES + ["no class (NWP missing)"] if c in t.columns]
    t = t[cols]
    t.index.name = "station"
    md(t, fmt="{:.0f}")


def seed_variability(r_all):
    out("## Seed variability (station level, alpha = 0.1)")
    out()
    out("Per-station 'all' PICP and PINAW: mean +/- sample std (ddof=1) over seeds, per base model "
        "(lgbm 3 seeds; lstm, dlinear 5 seeds). Last column: mean over stations of the seed std.")
    out()
    r = r_all[np.isclose(r_all["alpha"], ALPHA) & r_all["main_analysis"] & (r_all["wclass"] == "all")
              & r_all["method"].isin(METHODS)]
    g = r.groupby(["base_model", "method", "station"])[["picp", "pinaw"]].agg(["mean", "std"])
    for metric in ["picp", "pinaw"]:
        out(f"### {metric.upper()}")
        out()
        mean, sd = g[(metric, "mean")].unstack(), g[(metric, "std")].unstack()
        t = mean.combine(sd, lambda a, b: a.map("{:.3f}".format) + " +/- " + b.map("{:.3f}".format))
        t["mean seed std"] = sd.mean(axis=1).map("{:.4f}".format)
        order = [(b, m) for b in ["lgbm", "lstm", "dlinear"] for m in METHODS if (b, m) in t.index]
        t = t.loc[order]
        t.index = [f"{b} / {lab(m)}" for b, m in t.index]
        t.columns = [str(c) for c in t.columns]
        t.index.name = "base model / method"
        md(t)


def qr_lgbm_table(daily):
    out("### QR vs split_cp and wc_aci on LightGBM only")
    out()
    out("QR is available only for the LightGBM base model; for a like-for-like comparison all three "
        "methods are restricted to base_model == lgbm (3 seeds).")
    out()
    d = daily[daily["base_model"] == "lgbm"]
    tabs = {m: class_stats(day_table(d, m, ALPHA)).set_index("wclass") for m in ["qr", "split_cp", "wc_aci"]}
    t = pd.DataFrame({g: {lab(m): cell(x, g) for m, x in tabs.items()} for g in CLASSES + ["all"]})
    t.index.name = f"method, lgbm only ({CELL_HDR})"
    md(t)


def rule_note(w):
    out("## Operational rule note")
    out()
    out("Decisions 19 and 20 above are decided on point estimates, as coded before the results were seen; "
        "that verdict is binding. As a sensitivity check only, the Decision 20 (c1) band [0.88, 0.92] is also "
        "compared with the wc_aci 95% CIs. This CI rule was made concrete after the results were seen and "
        "does not change the verdict.")
    out()
    for k in CLASSES:
        lo, hi = w.loc[k, "picp_lo"], w.loc[k, "picp_hi"]
        inside, overlap = lo >= 0.88 and hi <= 0.92, lo <= 0.92 and hi >= 0.88
        out(f"- wc_aci {k}: CI [{lo:.3f}, {hi:.3f}]; CI entirely inside [0.88, 0.92]: "
            f"{'yes' if inside else 'no'}; CI overlaps [0.88, 0.92]: {'yes' if overlap else 'no'}")
    out()


EXPL_ALPHAS = [0.05, 0.1, 0.2]
EXPL_METHODS = ["split_cp", "aci", "cacp", "cacp_nwp", "cacp_wx", "wc_aci"]
EXPL_VARIANTS = ["cacp", "cacp_nwp", "cacp_wx"]


def exploratory(daily, r_all):
    """Post hoc, not pre-registered. Target coverage is 1 - alpha throughout."""
    out("## Exploratory (post hoc): CACP vs CACP_NWP vs CACP_WX (dilution check)")
    out()
    out("Not pre-registered; the Decision 19/20 results above are unaffected by this section. "
        "Target coverage is 1 - alpha at each alpha. Same pooling and date-clustered block bootstrap as above "
        "(B=2000, seed 0). Paired differences resample the same calendar dates for both methods.")
    out()
    fmt = lambda x, g, m: (f"{x.loc[g,m]:.3f} [{x.loc[g,m+'_lo']:.3f}, {x.loc[g,m+'_hi']:.3f}]")
    for a in EXPL_ALPHAS:
        out(f"### alpha = {a} (target PICP {1 - a:.2f})")
        out()
        dts = {m: day_table(daily, m, a) for m in EXPL_METHODS}
        for title, by, groups, prep in [("By NWP weather class", "wclass", CLASSES + ["all"], lambda d: d),
                                        ("By measured k stratum", "mstrat", STRATA + ["all"], meas_strata)]:
            out(f"#### {title}")
            out()
            tabs = {m: class_stats(prep(dts[m]), by=by).set_index(by) for m in EXPL_METHODS}
            t = pd.DataFrame({g: {m: cell(x, g) for m, x in tabs.items()} for g in groups})
            t.index.name = f"method ({CELL_HDR}), target {1 - a:.2f}"
            md(t)
        for ma, mb in [("cacp_nwp", "cacp"), ("cacp_wx", "cacp"), ("cacp_wx", "cacp_nwp"), ("wc_aci", "cacp_wx")]:
            out(f"#### Paired difference {ma} - {mb} (paired date-clustered block bootstrap)")
            out()
            rows = {}
            for title, by, groups, prep in [("class", "wclass", CLASSES + ["all"], lambda d: d),
                                            ("stratum", "mstrat", STRATA + ["all"], meas_strata)]:
                d = paired_diff(prep(dts[ma]), prep(dts[mb]), by=by).set_index(by)
                for g in groups:
                    rows[f"{title}: {g}"] = {"dPICP [CI]": fmt(d, g, "dpicp"), "dPINAW [CI]": fmt(d, g, "dpinaw")}
            t = pd.DataFrame(rows).T
            t.index.name = "group"
            md(t)
        out("#### Per-station overcast PICP (mean over base models and seeds)")
        out()
        o = r_all[(np.isclose(r_all["alpha"], a)) & r_all["main_analysis"] & (r_all["wclass"] == "overcast")
                  & r_all["method"].isin(EXPL_VARIANTS + ["wc_aci"])]
        p = o.groupby(["station", "method"])["picp"].mean().unstack()[EXPL_VARIANTS + ["wc_aci"]]
        p["nwp-cacp"] = p["cacp_nwp"] - p["cacp"]
        p["wx-cacp"] = p["cacp_wx"] - p["cacp"]
        p["wx-nwp"] = p["cacp_wx"] - p["cacp_nwp"]
        p.index.name = "station"
        md(p)
        out(f"- stations where cacp_nwp > cacp (overcast PICP): {int((p['nwp-cacp'] > 0).sum())}/{len(p)}")
        out(f"- stations where cacp_wx > cacp: {int((p['wx-cacp'] > 0).sum())}/{len(p)}; "
            f"cacp_wx > cacp_nwp: {int((p['wx-nwp'] > 0).sum())}/{len(p)}")
        out(f"- stations below target {1 - a:.2f}: " + ", ".join(
            f"{m} {int((p[m] < 1 - a).sum())}/{len(p)}" for m in EXPL_VARIANTS + ["wc_aci"]))
        out()


def main():
    r = pd.read_csv("results/results.csv")
    r = r[(r["alpha"] == ALPHA) & r["main_analysis"] & r["wclass"].isin(CLASSES)]
    r_all = pd.read_csv("results/results.csv")
    daily = pd.read_parquet("results/daily.parquet")
    n_st = r["station"].nunique()
    out("# M3/M4 verdict")
    out()
    out(f"Generated by `python -m scripts.check_criteria`. alpha = {ALPHA}, {n_st} main-analysis stations. "
        "Station-level PICP averages over base models and seeds. Pooled statistics weight days by interval "
        f"count; {BOOT}. Winkler = pooled sum of Winkler scores / number of intervals. "
        "wclass = NWP-based class; meas strata = per-station terciles of measured daily k (evaluation only). "
        "qr is available for the LightGBM base model only.")
    out()
    test_days(daily)
    st = r.groupby(["method", "station", "wclass"])["picp"].mean().reset_index()

    out("## Decision 19: audit holds")
    out()
    sp = st[st["method"] == "split_cp"]
    ok = False
    for c in CLASSES:
        k = int((sp[sp["wclass"] == c]["picp"] < 0.85).sum())
        out(f"- {c}: stations with split_cp PICP < 0.85: {k}/{n_st}")
        ok |= k > n_st / 2
    out()
    out(f"**Decision 19 (some class fails at more than half the stations): {'PASS' if ok else 'FAIL'}**")
    out()
    for m in ["split_cp", "cacp"]:
        out(f"Per-station PICP, {m}:")
        out()
        t = st[st["method"] == m].pivot(index="station", columns="wclass", values="picp")[CLASSES]
        md(t)

    out("## Pooled class tables")
    out()
    pooled, strat = {}, {}
    for m in METHODS:
        dt = day_table(daily, m, ALPHA)
        pooled[m] = class_stats(dt).set_index("wclass")
        strat[m] = class_stats(meas_strata(dt), by="mstrat").set_index("mstrat")
    for title, tabs, groups in [("By NWP weather class", pooled, CLASSES + ["all"]),
                                ("By measured k stratum", strat, STRATA + ["all"])]:
        out(f"### {title}")
        out()
        t = pd.DataFrame({g: {lab(m): cell(x, g) for m, x in tabs.items()} for g in groups})
        t.index.name = f"method ({CELL_HDR})"
        md(t)
    qr_lgbm_table(daily)

    out("## Decision 20: method holds")
    out()
    w, s, c = pooled["wc_aci"], pooled["split_cp"], pooled["cacp"]
    c1 = all(abs(w.loc[k, "picp"] - 0.9) <= 0.02 for k in CLASSES)
    for k in CLASSES:
        out(f"- wc_aci {k}: PICP {w.loc[k,'picp']:.3f} [{w.loc[k,'picp_lo']:.3f}, {w.loc[k,'picp_hi']:.3f}]")
    out(f"- (c1) all three classes within 0.90 +/- 0.02: **{'PASS' if c1 else 'FAIL'}**")
    c2 = w.loc["clear", "pinaw"] < s.loc["clear", "pinaw"]
    out(f"- (c2) clear PINAW wc_aci {w.loc['clear','pinaw']:.3f} < split_cp {s.loc['clear','pinaw']:.3f}: "
        f"**{'PASS' if c2 else 'FAIL'}**")
    gap = lambda x: max(abs(x.loc[k, "picp"] - 0.9) for k in CLASSES)
    c3 = gap(w) <= gap(c)
    out(f"- (c3) max |PICP-0.9| wc_aci {gap(w):.3f} <= cacp {gap(c):.3f}: **{'PASS' if c3 else 'FAIL'}**")
    out()
    rule_note(w)
    seed_variability(r_all)
    exploratory(daily, r_all)
    open("results/m3_m4_verdict.md", "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
