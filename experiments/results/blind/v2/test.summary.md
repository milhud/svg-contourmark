# Blind watermark evaluation (900 assets, decision threshold p <= 1e-6)

## Detection rate (TPR at p <= 1e-6) by attack

| Attack | spectral | fd_vertex | bezier_split | numeric_lsb | xml_comment | metadata_element |
|---|---|---|---|---|---|---|
| identity | 0.83 | 0.00 | 0.09 | 0.80 | 1.00 | 1.00 |
| svgo_default | 0.79 | 0.00 | 0.00 | 0.54 | 0.00 | 0.00 |
| svgo_multipass | 0.79 | 0.00 | 0.00 | 0.54 | 0.00 | 0.00 |
| svgo_p2 | 0.44 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| scour_default | 0.83 | 0.00 | 0.09 | 0.76 | 1.00 | 1.00 |
| scour_p3 | 0.70 | 0.00 | 0.05 | 0.22 | 1.00 | 1.00 |
| picosvg | 0.65 | 0.00 | 0.06 | 0.59 | 0.00 | 0.00 |
| round_2dp | 0.64 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 |
| round_1dp | 0.13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| svgo_p1 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| round_0dp | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| translate_5pct | 0.83 | 0.00 | 0.09 | 0.34 | 0.00 | 0.00 |
| scale_0.37 | 0.83 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| scale_3 | 0.83 | 0.00 | 0.09 | 0.70 | 0.00 | 0.00 |
| rotate_30 | 0.83 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| rotate_90 | 0.83 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| mirror | 0.83 | 0.00 | 0.09 | 0.70 | 0.00 | 0.00 |
| group_transform | 0.83 | 0.00 | 0.09 | 0.80 | 0.00 | 1.00 |
| aspect_1.2 | 0.00 | 0.00 | 0.08 | 0.29 | 0.00 | 0.00 |
| reorder | 0.83 | 0.00 | 0.05 | 0.46 | 0.00 | 0.00 |
| reverse | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| restart | 0.80 | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 |
| subdivide | 0.83 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| merge_paths | 0.83 | 0.00 | 0.09 | 0.70 | 0.00 | 0.00 |
| split_subpaths | 0.83 | 0.00 | 0.09 | 0.70 | 0.00 | 0.00 |
| delete_25pct | 0.82 | 0.00 | 0.04 | 0.59 | 0.00 | 0.00 |
| delete_50pct | 0.69 | 0.00 | 0.01 | 0.60 | 0.00 | 0.00 |
| crop_half | 0.69 | 0.00 | 0.01 | 0.31 | 0.00 | 0.00 |
| compose_grid | 0.74 | 0.00 | 0.09 | 0.00 | 0.00 | 0.00 |
| noise_0.1pct | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| noise_0.3pct | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| noise_1pct | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| polyline_0.05pct | 0.69 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 |
| polyline_0.2pct | 0.24 | 0.00 | 0.00 | 0.04 | 0.00 | 0.00 |
| revectorize_512 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| revectorize_1024 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## Attack accounting

Rates above are over completed attacks. Attacks that could not run or whose detector raised:

None: every attack completed for every method.

## Nulls, fidelity, size

| Method | embedded | FPR@1e-2 orig | FPR@1e-6 orig | FPR@1e-6 wrong key | PSNR@1024 med | SSIM@1024 med | bytes x | SVGO bytes x | gzip x |
|---|---|---|---|---|---|---|---|---|---|
| spectral | 900 | 0.006 | 0.000 | 0.000 | 25.4 | 0.9796 | 1.86 | 1.76 | 1.86 |
| fd_vertex | 900 | 0.001 | 0.000 | 0.000 | 83.1 | 1.0000 | 1.01 | 1.02 | 1.03 |
| bezier_split | 900 | 0.000 | 0.000 | 0.000 | 61.9 | 1.0000 | 1.36 | 1.13 | 1.32 |
| numeric_lsb | 900 | 0.004 | 0.000 | 0.000 | 54.4 | 1.0000 | 1.07 | 1.11 | 1.07 |
| xml_comment | 900 | 0.000 | 0.000 | 0.000 | 99.0 | 1.0000 | 1.11 | 1.00 | 1.14 |
| metadata_element | 900 | 0.000 | 0.000 | 0.000 | 99.0 | 1.0000 | 1.14 | 1.00 | 1.19 |

## Spectral watermark by source

| Source | n | clean TPR | SVGO TPR | round-2dp TPR | median log10 p | median marked contours | PSNR@1024 |
|---|---|---|---|---|---|---|---|
| bootstrap | 100 | 0.87 | 0.78 | 0.48 | -16.1 | 3 | 24.0 |
| fontawesome | 100 | 0.88 | 0.88 | 0.88 | -20.1 | 3 | 24.4 |
| lucide | 100 | 0.70 | 0.67 | 0.57 | -14.4 | 2 | 25.2 |
| material | 100 | 0.82 | 0.77 | 0.54 | -16.9 | 3 | 26.4 |
| octicons | 100 | 0.86 | 0.80 | 0.56 | -16.8 | 3 | 24.2 |
| openmoji | 100 | 0.92 | 0.92 | 0.91 | -59.1 | 11 | 28.7 |
| remix | 100 | 0.84 | 0.75 | 0.60 | -16.1 | 3 | 26.4 |
| simple-icons | 100 | 0.90 | 0.87 | 0.73 | -24.9 | 3 | 25.0 |
| tabler | 100 | 0.69 | 0.63 | 0.52 | -13.5 | 2 | 25.5 |
