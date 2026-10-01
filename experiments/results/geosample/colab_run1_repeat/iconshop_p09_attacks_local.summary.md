# Geosample evaluation (120 marked, 120 plain samples)

Truncated at the token limit: 38 marked, 38 plain. Marked samples with no descriptor: 0. All are kept in the denominators.

Rates are over completed attacks; `errors` counts attacks that could not run (attack/detect).

| Attack | TPR@1e-6 | TPR@1e-3 | conditional@1e-6 | median log10 p | attempted | errors |
|---|---|---|---|---|---|---|
| identity | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| svgo_default | 0.66 | 0.85 | 1.00 | -10.0 | 120 | 0/0 |
| svgo_multipass | 0.66 | 0.85 | 1.00 | -10.0 | 120 | 0/0 |
| svgo_p2 | 0.66 | 0.85 | 1.00 | -10.0 | 120 | 0/0 |
| scour_default | 0.66 | 0.85 | 1.00 | -10.0 | 120 | 0/0 |
| scour_p3 | 0.66 | 0.85 | 1.00 | -10.0 | 120 | 0/0 |
| picosvg | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| round_2dp | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| round_1dp | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| svgo_p1 | 0.65 | 0.85 | 0.99 | -10.0 | 120 | 0/0 |
| round_0dp | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| translate_5pct | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| scale_0.37 | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| scale_3 | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| rotate_30 | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| rotate_90 | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| mirror | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| group_transform | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| aspect_1.2 | 0.00 | 0.03 | 0.00 | -0.7 | 120 | 0/0 |
| reorder | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| reverse | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| restart | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| subdivide | 0.00 | 0.01 | 0.00 | -0.5 | 120 | 0/0 |
| merge_paths | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| split_subpaths | 0.66 | 0.85 | 1.00 | -10.2 | 120 | 0/0 |
| delete_25pct | 0.68 | 0.86 | 0.99 | -8.9 | 120 | 0/0 |
| delete_50pct | 0.64 | 0.81 | 0.95 | -7.9 | 120 | 0/0 |
| crop_half | 0.40 | 0.65 | 0.61 | -4.8 | 120 | 0/0 |
| compose_grid | 0.47 | 0.72 | 0.71 | -5.8 | 120 | 0/0 |
| noise_0.1pct | 0.02 | 0.15 | 0.03 | -1.4 | 120 | 0/0 |
| noise_0.3pct | 0.00 | 0.00 | 0.00 | -0.4 | 120 | 0/0 |
| noise_1pct | 0.00 | 0.00 | 0.00 | -0.2 | 120 | 0/0 |
| polyline_0.05pct | 0.00 | 0.00 | 0.00 | -0.2 | 120 | 0/0 |
| polyline_0.2pct | 0.00 | 0.01 | 0.00 | -0.4 | 120 | 0/0 |
| revectorize_512 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |
| revectorize_1024 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |

## Nulls

| Null | n | FPR@1e-2 | FPR@1e-3 | FPR@1e-6 |
|---|---|---|---|---|
| plain_with_key | 120 | 0.017 | 0.000 | 0.000 |
| marked_wrong_key | 120 | 0.033 | 0.000 | 0.000 |
| plain_wrong_key | 120 | 0.008 | 0.000 | 0.000 |

## Clean detection vs distinct vertices (marked)

| distinct vertices | n | median log10 p | TPR@1e-6 | TPR@1e-3 |
|---|---|---|---|---|
| 10-19 | 1 | -0.1 | 0.00 | 0.00 |
| 20-39 | 9 | -2.7 | 0.00 | 0.44 |
| 40-79 | 29 | -4.1 | 0.31 | 0.66 |
| 80-9999 | 81 | -12.8 | 0.86 | 0.98 |

## Distribution preservation (plain vs marked, Mann-Whitney U)

| Statistic | plain median | marked median | p |
|---|---|---|---|
| tokens | 353.0 | 392.0 | 0.581 |
| vertices | 161.0 | 183.0 | 0.475 |
| distinct_vertices | 100.5 | 111.0 | 0.564 |
| paths | 5.0 | 6.0 | 0.647 |
| contours | 5.0 | 6.0 | 0.647 |
