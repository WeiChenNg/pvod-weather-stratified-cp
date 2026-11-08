# Paper extras (descriptive, post hoc; alpha = 0.1 unless noted)

## Per-class PINAW and Winkler (pooled; NWP classes)

| method        |   pinaw_overcast |   pinaw_cloudy |   pinaw_clear |   pinaw_all |   wink_overcast |   wink_cloudy |   wink_clear |   wink_all |
|:--------------|-----------------:|---------------:|--------------:|------------:|----------------:|--------------:|-------------:|-----------:|
| split_cp      |            0.458 |          0.57  |         0.555 |       0.536 |           0.553 |         0.639 |        0.622 |      0.61  |
| aci           |            0.429 |          0.518 |         0.502 |       0.489 |           0.546 |         0.612 |        0.583 |      0.583 |
| mondrian_hour |            0.442 |          0.536 |         0.525 |       0.508 |           0.561 |         0.618 |        0.58  |      0.588 |
| cacp          |            0.393 |          0.458 |         0.441 |       0.435 |           0.565 |         0.578 |        0.531 |      0.555 |
| cacp_nwp      |            0.401 |          0.426 |         0.398 |       0.408 |           0.56  |         0.567 |        0.504 |      0.539 |
| cacp_wx       |            0.426 |          0.423 |         0.391 |       0.411 |           0.553 |         0.565 |        0.492 |      0.532 |
| wc_aci        |            0.436 |          0.563 |         0.512 |       0.51  |           0.55  |         0.637 |        0.595 |      0.598 |
| qr            |            0.342 |          0.412 |         0.409 |       0.393 |           0.554 |         0.599 |        0.683 |      0.623 |

## Per base forecaster: overcast PICP and overall PINAW (point estimates)

| base_model   | method   |   picp_overcast |   picp_all |   pinaw_all |
|:-------------|:---------|----------------:|-----------:|------------:|
| lgbm         | split_cp |           0.901 |      0.924 |       0.514 |
| lgbm         | cacp     |           0.871 |      0.907 |       0.426 |
| lgbm         | cacp_nwp |           0.867 |      0.897 |       0.393 |
| lgbm         | cacp_wx  |           0.887 |      0.906 |       0.399 |
| lgbm         | wc_aci   |           0.895 |      0.914 |       0.486 |
| lstm         | split_cp |           0.913 |      0.921 |       0.496 |
| lstm         | cacp     |           0.884 |      0.907 |       0.42  |
| lstm         | cacp_nwp |           0.886 |      0.899 |       0.387 |
| lstm         | cacp_wx  |           0.899 |      0.906 |       0.389 |
| lstm         | wc_aci   |           0.892 |      0.911 |       0.481 |
| dlinear      | split_cp |           0.919 |      0.932 |       0.589 |
| dlinear      | cacp     |           0.871 |      0.907 |       0.455 |
| dlinear      | cacp_nwp |           0.889 |      0.906 |       0.438 |
| dlinear      | cacp_wx  |           0.9   |      0.909 |       0.439 |
| dlinear      | wc_aci   |           0.898 |      0.92  |       0.553 |

## Per base forecaster: paired CACP-W minus CACP-T+W (95% date-block CI)

| base_model   | stratum   |   dpicp |   dpicp_lo |   dpicp_hi |   dpinaw |   dpinaw_lo |   dpinaw_hi |
|:-------------|:----------|--------:|-----------:|-----------:|---------:|------------:|------------:|
| lgbm         | overcast  |   0.02  |      0.014 |      0.03  |    0.024 |       0.016 |       0.031 |
| lgbm         | meas_low  |   0.017 |      0.011 |      0.023 |    0.017 |       0.01  |       0.024 |
| lstm         | overcast  |   0.014 |      0.009 |      0.022 |    0.018 |       0.013 |       0.024 |
| lstm         | meas_low  |   0.01  |      0.004 |      0.016 |    0.012 |       0.006 |       0.017 |
| dlinear      | overcast  |   0.012 |      0.01  |      0.014 |    0.032 |       0.027 |       0.037 |
| dlinear      | meas_low  |   0.011 |      0.007 |      0.015 |    0.025 |       0.018 |       0.031 |

## Selected tuning values (104 runs = 8 stations x 13 base runs)

- cacp_kfrac: {0.05: 12, 0.1: 12, 0.2: 25, 0.5: 35, 1.0: 20}
- cacp_nwp_kfrac: {0.01: 1, 0.02: 2, 0.05: 14, 0.1: 27, 0.2: 27, 0.5: 24, 1.0: 9}
- cacp_wx_kfrac: {0.01: 1, 0.02: 9, 0.05: 17, 0.1: 27, 0.2: 23, 0.5: 14, 1.0: 13}
- aci_gamma: {0.005: 43, 0.01: 9, 0.05: 16, 0.1: 36}
- wc_aci_gamma: {0.005: 47, 0.01: 12, 0.05: 8, 0.1: 37}

k_frac = 1.0 uses the whole history, i.e. the CACP quantile equals the split-CP quantile.
