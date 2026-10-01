# Geosample evaluation (120 marked, 120 plain samples)

Truncated at the token limit: 37 marked, 37 plain. Marked samples with no descriptor: 0. All are kept in the denominators.

Rates are over completed attacks; `errors` counts attacks that could not run (attack/detect).

| Attack | TPR@1e-6 | TPR@1e-3 | conditional@1e-6 | median log10 p | attempted | errors |
|---|---|---|---|---|---|---|
| identity | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| svgo_default | not run | not run | — | — | 120 | 120/0 |
| svgo_multipass | not run | not run | — | — | 120 | 120/0 |
| svgo_p2 | not run | not run | — | — | 120 | 120/0 |
| scour_default | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| scour_p3 | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| picosvg | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| round_2dp | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| round_1dp | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| svgo_p1 | not run | not run | — | — | 120 | 120/0 |
| round_0dp | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| translate_5pct | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| scale_0.37 | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| scale_3 | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| rotate_30 | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| rotate_90 | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| mirror | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| group_transform | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| aspect_1.2 | 0.00 | 0.00 | 0.00 | -0.4 | 120 | 0/0 |
| reorder | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| reverse | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| restart | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| subdivide | 0.00 | 0.00 | 0.00 | -0.2 | 120 | 0/0 |
| merge_paths | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| split_subpaths | 0.07 | 0.38 | 1.00 | -2.4 | 120 | 0/0 |
| delete_25pct | 0.07 | 0.34 | 0.88 | -2.1 | 120 | 0/0 |
| delete_50pct | 0.04 | 0.33 | 0.62 | -1.9 | 120 | 0/0 |
| crop_half | 0.04 | 0.19 | 0.62 | -1.0 | 120 | 0/0 |
| compose_grid | 0.03 | 0.17 | 0.50 | -1.4 | 120 | 0/0 |
| noise_0.1pct | 0.00 | 0.00 | 0.00 | -0.6 | 120 | 0/0 |
| noise_0.3pct | 0.00 | 0.00 | 0.00 | -0.3 | 120 | 0/0 |
| noise_1pct | 0.00 | 0.00 | 0.00 | -0.2 | 120 | 0/0 |
| polyline_0.05pct | 0.00 | 0.00 | 0.00 | -0.1 | 120 | 0/0 |
| polyline_0.2pct | 0.00 | 0.00 | 0.00 | -0.3 | 120 | 0/0 |
| revectorize_512 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |
| revectorize_1024 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |

## Nulls

| Null | n | FPR@1e-2 | FPR@1e-3 | FPR@1e-6 |
|---|---|---|---|---|
| plain_with_key | 120 | 0.000 | 0.000 | 0.000 |
| marked_wrong_key | 120 | 0.025 | 0.008 | 0.000 |
| plain_wrong_key | 120 | 0.017 | 0.000 | 0.000 |

## Clean detection vs distinct vertices (marked)

| distinct vertices | n | median log10 p | TPR@1e-6 | TPR@1e-3 |
|---|---|---|---|---|
| 10-19 | 4 | -1.1 | 0.00 | 0.00 |
| 20-39 | 17 | -1.2 | 0.00 | 0.00 |
| 40-79 | 61 | -2.3 | 0.00 | 0.34 |
| 80-9999 | 38 | -3.3 | 0.21 | 0.63 |

## Distribution preservation (plain vs marked, Mann-Whitney U)

| Statistic | plain median | marked median | p |
|---|---|---|---|
| tokens | 361.0 | 377.0 | 0.844 |
| vertices | 165.0 | 169.0 | 0.468 |
| distinct_vertices | 58.5 | 63.5 | 0.106 |
| paths | 5.0 | 6.0 | 0.675 |
| contours | 5.0 | 6.0 | 0.675 |
