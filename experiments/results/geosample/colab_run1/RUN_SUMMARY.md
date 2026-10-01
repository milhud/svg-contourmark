# Run summary: colab_run1

Detection columns are the fraction of marked samples with p <= 1e-6 (clean, then after each transformation).

| Config | sampler | top-p | key | n | clean 1e-6 | clean 1e-3 | SVGO | round 1dp | rotate | scale | subdivide | compose | noise 0.1% | FPR plain 1e-3 | CLIP top-1 marked / plain | truncated | descriptors | attack errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iconshop_p05 | gumbel-mask | 0.5 | evaluation | 120 | 0.11 | 0.47 | 0.12 | 0.11 | 0.09 | 0.08 | 0.00 | 0.03 | 0.00 | 0.000 | 0.44 / 0.45 | 43 | 56 | 0 |
| iconshop_p09 | gumbel-mask | 0.9 | evaluation | 120 | 0.62 | 0.91 | 0.62 | 0.62 | 0.61 | 0.62 | 0.00 | 0.40 | 0.01 | 0.000 | 0.40 / 0.40 | 40 | 100 | 0 |
