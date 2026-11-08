"""Descriptive tables for the paper, computed from existing outputs only (no new runs).

python -m scripts.paper_extras  ->  results/paper_extras.md
All analyses here were defined after the main results were seen (exploratory / descriptive).
"""
import pandas as pd

from src.analysis import class_stats, day_table, meas_strata, paired_diff

METHODS = ["split_cp", "aci", "mondrian_hour", "cacp", "cacp_nwp", "cacp_wx", "wc_aci", "qr"]
CACP = ["cacp", "cacp_nwp", "cacp_wx", "wc_aci"]
A = 0.1


def main():
    daily = pd.read_parquet("results/daily.parquet")
    out = ["# Paper extras (descriptive, post hoc; alpha = 0.1 unless noted)\n"]

    out.append("## Per-class PINAW and Winkler (pooled; NWP classes)\n")
    rows = []
    for m in METHODS:
        s = class_stats(day_table(daily, m, A)).set_index("wclass")
        rows.append({"method": m, **{f"pinaw_{c}": s.loc[c, "pinaw"] for c in ["overcast", "cloudy", "clear", "all"]},
                     **{f"wink_{c}": s.loc[c, "winkler"] for c in ["overcast", "cloudy", "clear", "all"]}})
    out.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

    out.append("## Per base forecaster: overcast PICP and overall PINAW (point estimates)\n")
    rows = []
    for bm in ["lgbm", "lstm", "dlinear"]:
        sub = daily[daily["base_model"] == bm]
        for m in ["split_cp"] + CACP:
            s = class_stats(day_table(sub, m, A), B=1).set_index("wclass")
            rows.append({"base_model": bm, "method": m, "picp_overcast": s.loc["overcast", "picp"],
                         "picp_all": s.loc["all", "picp"], "pinaw_all": s.loc["all", "pinaw"]})
    out.append(pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

    out.append("## Per base forecaster: paired CACP-W minus CACP-T+W (95% date-block CI)\n")
    rows = []
    for bm in ["lgbm", "lstm", "dlinear"]:
        sub = daily[daily["base_model"] == bm]
        a, b = meas_strata(day_table(sub, "cacp_wx", A)), meas_strata(day_table(sub, "cacp_nwp", A))
        for by, lab in [("wclass", "overcast"), ("mstrat", "meas_low")]:
            r = paired_diff(a, b, by=by).set_index(by).loc[lab]
            rows.append({"base_model": bm, "stratum": lab, **r.round(3).to_dict()})
    out.append(pd.DataFrame(rows).to_markdown(index=False) + "\n")

    out.append("## Selected tuning values (104 runs = 8 stations x 13 base runs)\n")
    t = pd.read_csv("results/tuned_params.csv")
    for c in ["cacp_kfrac", "cacp_nwp_kfrac", "cacp_wx_kfrac", "aci_gamma", "wc_aci_gamma"]:
        out.append(f"- {c}: {t[c].value_counts().sort_index().to_dict()}")
    out.append("\nk_frac = 1.0 uses the whole history, i.e. the CACP quantile equals the split-CP quantile.\n")

    open("results/paper_extras.md", "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
