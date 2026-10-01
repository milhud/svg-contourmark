# Geosample evaluation (120 marked, 120 plain samples)

Truncated at the token limit: 40 marked, 38 plain. Marked samples with no descriptor: 0. All are kept in the denominators.

Rates are over completed attacks; `errors` counts attacks that could not run (attack/detect).

| Attack | TPR@1e-6 | TPR@1e-3 | conditional@1e-6 | median log10 p | attempted | errors |
|---|---|---|---|---|---|---|
| identity | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| scour_default | 0.62 | 0.90 | 1.00 | -7.7 | 120 | 0/0 |
| scour_p3 | 0.62 | 0.90 | 1.00 | -7.7 | 120 | 0/0 |
| picosvg | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| round_2dp | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| round_1dp | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| round_0dp | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| translate_5pct | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| scale_0.37 | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| scale_3 | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| rotate_30 | 0.61 | 0.90 | 0.99 | -7.7 | 120 | 0/0 |
| rotate_90 | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| mirror | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| group_transform | 0.61 | 0.89 | 0.99 | -7.6 | 120 | 0/0 |
| aspect_1.2 | 0.00 | 0.03 | 0.00 | -0.8 | 120 | 0/0 |
| reorder | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| reverse | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| restart | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| subdivide | 0.00 | 0.02 | 0.00 | -0.8 | 120 | 0/0 |
| merge_paths | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| split_subpaths | 0.62 | 0.91 | 1.00 | -7.7 | 120 | 0/0 |
| delete_25pct | 0.57 | 0.90 | 0.92 | -6.9 | 120 | 0/0 |
| delete_50pct | 0.47 | 0.80 | 0.77 | -5.6 | 120 | 0/0 |
| crop_half | 0.39 | 0.67 | 0.64 | -4.7 | 120 | 0/0 |
| compose_grid | 0.40 | 0.69 | 0.65 | -5.1 | 120 | 0/0 |
| noise_0.1pct | 0.01 | 0.13 | 0.01 | -1.5 | 120 | 0/0 |
| noise_0.3pct | 0.00 | 0.00 | 0.00 | -0.5 | 120 | 0/0 |
| noise_1pct | 0.00 | 0.00 | 0.00 | -0.1 | 120 | 0/0 |
| polyline_0.05pct | 0.00 | 0.00 | 0.00 | -0.4 | 120 | 0/0 |
| polyline_0.2pct | 0.00 | 0.00 | 0.00 | -0.5 | 120 | 0/0 |
| revectorize_512 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |
| revectorize_1024 | 0.00 | 0.00 | 0.00 | 0.0 | 120 | 0/0 |

## Nulls

| Null | n | FPR@1e-2 | FPR@1e-3 | FPR@1e-6 |
|---|---|---|---|---|
| plain_with_key | 120 | 0.000 | 0.000 | 0.000 |
| marked_wrong_key | 120 | 0.025 | 0.017 | 0.000 |
| plain_wrong_key | 120 | 0.025 | 0.000 | 0.000 |

## Clean detection vs distinct vertices (marked)

| distinct vertices | n | median log10 p | TPR@1e-6 | TPR@1e-3 |
|---|---|---|---|---|
| 0-9 | 1 | 0.0 | 0.00 | 0.00 |
| 20-39 | 8 | -4.5 | 0.00 | 0.75 |
| 40-79 | 37 | -4.7 | 0.30 | 0.81 |
| 80-9999 | 74 | -11.3 | 0.85 | 0.99 |

## Distribution preservation (plain vs marked, Mann-Whitney U)

| Statistic | plain median | marked median | p |
|---|---|---|---|
| tokens | 353.0 | 383.0 | 0.505 |
| vertices | 161.0 | 171.0 | 0.494 |
| distinct_vertices | 101.5 | 99.5 | 0.907 |
| paths | 5.0 | 6.0 | 0.432 |
| contours | 5.0 | 6.0 | 0.432 |
