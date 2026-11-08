# Weather-stratified evaluation of conformal day-ahead PV power intervals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24050972.svg)](https://doi.org/10.5281/zenodo.24050972)

Code, results and reproduction commands for the paper

> W. C. Ng, Y. C. Lee, Y. K. Lee, Q. Xie, J. C. Kang, J. D. Tan.
> *Overcast-day coverage of conformal day-ahead photovoltaic power intervals: a weather-stratified
> evaluation of localized calibration and weather context on eight stations.* Submitted to the
> Journal of Power Sources.

Seven conformal methods and a quantile-regression reference are evaluated on eight stations of the
PV Output Dataset (PVOD; Hebei, China; 15-min resolution; spring 2019 test period) with three
point forecasters (LightGBM, LSTM, DLinear-style). Coverage is reported by numerical-weather-
prediction (NWP) weather class and by measured clear-sky-index tercile.

## Main result

Coverage (PICP) by NWP weather class at α = 0.1 (target 0.90), pooled over eight stations and 13
base runs per station (QR: LightGBM only), with the 95% date-clustered block-bootstrap CI for the
overcast class; PINAW is the overall mean width as a fraction of capacity. This is Table 3 of the
paper; all reported statistics are in `results/m3_m4_verdict.md` and `results/paper_extras.md`.

| Method | Overcast | Cloudy | Clear | All | PINAW (all) |
|---|---|---|---|---|---|
| QR (LightGBM, reference) | 0.758 [0.722, 0.793] | 0.802 | 0.723 | 0.758 | 0.393 |
| Split CP | 0.912 [0.891, 0.929] | 0.928 | 0.933 | 0.926 | 0.536 |
| ACI | 0.893 [0.874, 0.908] | 0.906 | 0.920 | 0.909 | 0.489 |
| Mondrian-hour | 0.911 [0.888, 0.930] | 0.936 | 0.956 | 0.938 | 0.508 |
| WC-ACI | 0.895 [0.880, 0.906] | 0.925 | 0.920 | 0.915 | 0.510 |
| CACP-T | 0.876 [0.850, 0.896] | 0.905 | 0.927 | 0.907 | 0.435 |
| CACP-T+W | 0.883 [0.858, 0.901] | 0.893 | 0.918 | 0.901 | 0.408 |
| CACP-W | 0.897 [0.875, 0.915] | 0.897 | 0.922 | 0.907 | 0.411 |

The two criteria fixed before testing, and their provenance, are in
[`docs/prespecified_criteria.md`](docs/prespecified_criteria.md). Neither was met; all other
comparisons, including every comparison among the CACP variants, are exploratory.

## Data

PVOD v1.0 is published on the Science Data Bank, DOI
[10.57760/sciencedb.01094](https://doi.org/10.57760/sciencedb.01094). **The raw data are not
redistributed here.** Released under CC BY 4.0 according to its Science Data Bank page; the
providers additionally ask that it be used for scientific research only and that the data paper
be cited:

> T. Yao, J. Wang, H. Wu, P. Zhang, S. Li, Y. Wang, X. Chi, M. Shi. A photovoltaic power output
> dataset: Multi-source photovoltaic power output dataset with Python toolkit. *Solar Energy* 230
> (2021) 122–130. https://doi.org/10.1016/j.solener.2021.09.050

Place the extracted files (`metadata.csv`, `station00.csv` … `station09.csv`) in
`data/raw/PVODdatasets_v1.0/`.

Two facts about the raw files that the code already handles: timestamps are UTC (converted to China
Standard Time), and `power` is in MW while the metadata `Capacity` is in kW.

## Environment

Python 3.12.7; exact package versions are pinned in `requirements.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OMP_NUM_THREADS=1      # results were produced single-threaded per process
```

`OMP_NUM_THREADS=1` is also set inside `src/forecasters.py`: on macOS, multithreaded OpenMP in
LightGBM and PyTorch together can crash or deadlock.

## Reproducing the results

### From the committed results (no data download)

These read only files under `results/` and regenerate the verdict, the supplementary tables and
the figures:

```bash
python -m scripts.check_criteria    # -> results/m3_m4_verdict.md
python -m scripts.paper_extras      # -> results/paper_extras.md
python -m src.figures               # -> results/figures/
```

### Full pipeline (needs PVOD)

Run from the repository root, in this order:

```bash
# M0: inspect raw files -> results/m0_inspection.csv
python scripts/inspect_pvod.py data/raw/PVODdatasets_v1.0

# M1: processed frames, weather classes, class counts
#     -> data/processed/, results/m1_stations.json, results/m1_class_counts.csv
python -m scripts.build_frames

# M2: point forecasts and LightGBM quantile intervals -> results/m2_point_skill.csv
python -m scripts.run_forecasts

# M3/M4: tuning and all interval methods, one process per main-analysis station
for s in 0 1 2 4 6 7 8 9; do python -m src.run --stations $s & done; wait
python -m src.run --merge           # -> results/daily.parquet, results.csv, tuned_params.csv

# Criteria, exploratory analyses, supplementary tables, figures
python -m scripts.check_criteria
python -m scripts.paper_extras
python -m src.figures
```

**Point-forecast cache.** `results/forecasts/` holds the 104 point forecasts used in the paper
(8 stations × 13 seeded runs: LightGBM 3 seeds, LSTM 5, DLinear 5). `run_forecasts` reuses them
by default, so the interval results can be reproduced exactly without retraining. Pass `--force`
to retrain; retrained neural forecasters may differ slightly across hardware and library builds.

`python -m src.run --quick` runs one seed per base model as a smoke test and writes only under
`results/quick/`.

Tests: `pytest -q`.

## Layout

- `src/` data loading (`data.py`), weather classes (`weather.py`), point forecasters
  (`forecasters.py`), interval methods (`intervals.py`), metrics, runner and tuning (`run.py`),
  bootstrap and paired comparisons (`analysis.py`), figures
- `scripts/` pipeline entry points (M0, M1, M2, criteria check, supplementary tables)
- `tests/` unit tests: look-ahead, synthetic coverage, metrics, bootstrap, tuning isolation
- `results/daily.parquet` the per-(run, station, date) table from which all reported statistics
  are computed; `results/tuned_params.csv` the tuned γ and CACP neighbourhood per run
- `docs/prespecified_criteria.md` the pre-specified criteria and their provenance

Method names in the code differ from the paper: `cacp` = CACP-T, `cacp_nwp` = CACP-T+W,
`cacp_wx` = CACP-W, `wc_aci` = WC-ACI, `mondrian_hour` = Mondrian-hour, `qr` = QR.

## Citation

If you use this code, please cite the paper (citation will be updated on publication) and this
archive, https://doi.org/10.5281/zenodo.24050972 (all versions); see `CITATION.cff`.

## License

Code: MIT, see `LICENSE`. The PVOD data are not covered by this license.
