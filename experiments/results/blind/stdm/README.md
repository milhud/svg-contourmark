# Spread-transform dither modulation vs the informed re-embedding attacker

54 dev icons (`--per-source 6` of `corpus/dev.json`), per-asset keys, threshold p <= 1e-6, delta = 0.004.
Survival rates are conditional on clean detection. Alignment is the owner's statistic divided by its maximum (1 = on the lattice, 0 = unmarked).
Boundary distance is the symmetric p95 as a fraction of the diagonal; PSNR is at 1024 px. Attack distortion is measured against the marked file.

## Owner side

| Owner scheme | Clean detected | Median clean log10 p | After SVGO | After round_2dp | Embed PSNR | Embed boundary p95 | Contours skipped (mask / seed unstable) |
|---|---|---|---|---|---|---|---|
| plain QIM (spread=0) | 47/54 (87%) | -17.4 | 96% | 85% | 25.7 dB | 0.164% | 0 / 2 of 248 |
| STDM spread=4 | 45/54 (83%) | -12.9 | 87% | 53% | 26.4 dB | 0.128% | 13 / 3 of 248 |
| STDM spread=2 | 29/54 (54%) | -6.2 | 66% | 45% | 28.0 dB | 0.112% | 14 / 3 of 248 |

## Attacker side (re-embedding with a random key)

Each cell: survival at 1e-6 / survival at 1e-3 / median alignment left / attack PSNR / attack boundary p95.

| Owner scheme | QIM at delta | QIM at 2 delta | QIM at 4 delta | same scheme at delta | same scheme at 2 delta |
|---|---|---|---|---|---|
| plain QIM (spread=0) | 4% / 4% / 0.04 / 26.6 dB / 0.144% | 4% / 6% / 0.03 / 22.2 dB / 0.278% | 6% / 9% / 0.03 / 18.6 dB / 0.564% | 4% / 4% / 0.04 / 26.6 dB / 0.144% | 4% / 6% / 0.03 / 22.2 dB / 0.278% |
| STDM spread=4 | 0% / 2% / 0.30 / 26.5 dB / 0.148% | 4% / 11% / 0.04 / 22.4 dB / 0.291% | 2% / 13% / -0.06 / 18.4 dB / 0.636% | 4% / 13% / 0.24 / 28.3 dB / 0.114% | 9% / 16% / 0.07 / 22.9 dB / 0.252% |
| STDM spread=2 | 0% / 14% / 0.42 / 25.7 dB / 0.130% | 0% / 3% / 0.10 / 22.2 dB / 0.285% | 0% / 14% / 0.00 / 18.8 dB / 0.543% | 0% / 10% / 0.41 / 29.0 dB / 0.099% | 0% / 14% / 0.11 / 23.9 dB / 0.198% |

## Mask sensitivity

STDM rows depend on the verifier's selection mask. Fraction of marked-file contours whose (seed, mask) pair is unchanged by a benign transform (clean-detectable files):

| Owner scheme | SVGO default | round_2dp |
|---|---|---|
| plain QIM (spread=0) | 99.1% | 96.1% |
| STDM spread=4 | 100.0% | 97.4% |
| STDM spread=2 | 99.4% | 98.3% |

## Reading

* STDM does not raise the informed attacker's cost at the 1e-6 operating point: a random-key re-embedding at the owner's own step (plain QIM or the same STDM scheme) removes 96-100% of marks under every owner scheme, at a distortion comparable to the embedding's (same-scheme attack: 1-2 dB less; plain-QIM attack on spread=2: 2.3 dB more).
* The mechanism does work per carrier: the alignment left after a one-step attack rises from about 0.04 (plain QIM) to about 0.3 (spread=4) and 0.4 (spread=2), as the sqrt(L/P) larger step predicts. A contour then carries only P terms instead of up to 8, and icons have too few contours for the remaining partial alignment to reach 1e-6. Doubling the attacker's step (about 3-4 dB more distortion) removes the residue as well.
* The cost is paid by the owner: fewer terms mean lower clean detection (87% -> 83% -> 54%) and much less margin, so benign survival falls (SVGO 96% -> 87% -> 66%; two-decimal rounding 85% -> 53% -> 45%) even though alignment after those transforms stays above 0.97. About 5% of contours are skipped because the verifier's mask is not a fixed point of embedding.
* Embedding distortion is slightly lower with STDM in this sample (0.11-0.13% vs 0.16% boundary p95; the step was chosen for equal magnitude-domain energy), so a matched-distortion comparison would let STDM use a somewhat larger delta; that does not change the attacker result, which is limited by term count.
* n = 54 icons (29-47 clean-detectable); differences of a few points between cells are within noise.
