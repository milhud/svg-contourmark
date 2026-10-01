# Geosample evaluation (120 marked, 120 plain samples)

Truncated at the token limit: 43 marked, 37 plain. Marked samples with no descriptor: 0. All are kept in the denominators.

Rates are over completed attacks; `errors` counts attacks that could not run (attack/detect).

| Attack | TPR@1e-6 | TPR@1e-3 | conditional@1e-6 | median log10 p | attempted | errors |
|---|---|---|---|---|---|---|
| identity | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| scour_default | 0.12 | 0.49 | 1.00 | -2.9 | 120 | 0/0 |
| scour_p3 | 0.12 | 0.49 | 1.00 | -2.9 | 120 | 0/0 |
| picosvg | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| round_2dp | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| round_1dp | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| round_0dp | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| translate_5pct | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| scale_0.37 | 0.08 | 0.47 | 0.77 | -2.8 | 120 | 0/0 |
| scale_3 | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| rotate_30 | 0.09 | 0.44 | 0.85 | -2.7 | 120 | 0/0 |
| rotate_90 | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| mirror | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| group_transform | 0.08 | 0.43 | 0.77 | -2.7 | 120 | 0/0 |
| aspect_1.2 | 0.00 | 0.01 | 0.00 | -0.6 | 120 | 0/0 |
| reorder | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| reverse | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| restart | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| subdivide | 0.00 | 0.00 | 0.00 | -0.5 | 120 | 0/0 |
| merge_paths | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| split_subpaths | 0.11 | 0.47 | 1.00 | -2.8 | 120 | 0/0 |
| delete_25pct | 0.08 | 0.47 | 0.77 | -2.7 | 120 | 0/0 |
| delete_50pct | 0.09 | 0.40 | 0.69 | -2.6 | 120 | 0/0 |
| crop_half | 0.07 | 0.27 | 0.62 | -1.5 | 120 | 0/0 |
| compose_grid | 0.03 | 0.24 | 0.31 | -1.5 | 120 | 0/0 |
| noise_0.1pct | 0.00 | 0.00 | 0.00 | -0.5 | 120 | 0/0 |
| noise_0.3pct | 0.00 | 0.00 | 0.00 | -0.3 | 120 | 0/0 |
| noise_1pct | 0.00 | 0.00 | 0.00 | -0.2 | 120 | 0/0 |
| polyline_0.05pct | 0.00 | 0.00 | 0.00 | -0.3 | 120 | 0/0 |
| polyline_0.2pct | 0.00 | 0.00 | 0.00 | -0.3 | 120 | 0/0 |
| revectorize_512 | 0.00 | 0.00 | 0.00 | -0.1 | 120 | 0/0 |
| revectorize_1024 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |

## Nulls

| Null | n | FPR@1e-2 | FPR@1e-3 | FPR@1e-6 |
|---|---|---|---|---|
| plain_with_key | 120 | 0.008 | 0.000 | 0.000 |
| marked_wrong_key | 120 | 0.000 | 0.000 | 0.000 |
| plain_wrong_key | 120 | 0.000 | 0.000 | 0.000 |

## Clean detection vs distinct vertices (marked)

| distinct vertices | n | median log10 p | TPR@1e-6 | TPR@1e-3 |
|---|---|---|---|---|
| 10-19 | 1 | -1.4 | 0.00 | 0.00 |
| 20-39 | 18 | -1.8 | 0.06 | 0.28 |
| 40-79 | 66 | -2.4 | 0.02 | 0.35 |
| 80-9999 | 35 | -4.4 | 0.31 | 0.80 |

## Distribution preservation (plain vs marked, Mann-Whitney U)

| Statistic | plain median | marked median | p |
|---|---|---|---|
| tokens | 361.0 | 379.0 | 0.585 |
| vertices | 165.0 | 147.0 | 0.715 |
| distinct_vertices | 56.5 | 56.0 | 0.232 |
| paths | 5.0 | 5.0 | 0.815 |
| contours | 5.0 | 5.0 | 0.815 |
