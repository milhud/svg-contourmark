# Run summary: colab_run1_repeat

Detection columns are the fraction of marked samples with p <= 1e-6 (clean, then after each transformation).

| Config | sampler | top-p | key | n | clean 1e-6 | clean 1e-3 | SVGO | round 1dp | rotate | scale | subdivide | compose | noise 0.1% | FPR plain 1e-3 | CLIP top-1 marked / plain | truncated | descriptors | attack errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iconshop_p05 | gumbel-mask | 0.5 | evaluation | 120 | 0.07 | 0.38 | 0.07 | 0.07 | 0.07 | 0.07 | 0.00 | 0.03 | 0.00 | 0.000 | 0.45 / 0.45 | 37 | 64 | 0 |
| iconshop_p09 | gumbel-mask | 0.9 | evaluation | 120 | 0.66 | 0.85 | 0.66 | 0.66 | 0.66 | 0.66 | 0.00 | 0.47 | 0.02 | 0.000 | 0.39 / 0.40 | 38 | 111 | 0 |
