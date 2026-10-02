"""CPU lab for inference-sampler variants (no model, no GPU).

Compares, on the toy point-token generator at several per-step entropies:

* ``gumbel-mask``  distribution-preserving sampler (default);
* ``gumbel-allow`` naive score reuse (biased; ablation);
* ``bias-<delta>`` green-list sampler with the matched binomial detector;
* ``polygon-mask`` / ``polygon-allow`` the control-polygon scheme (every
  handle and endpoint keyed, in context with its neighbour).

For each variant it reports detection power, the measured group entropy per
keyed step, the distribution shift the sampler itself introduces (KL per
step; zero for the Gumbel modes by construction), survival under a few
attacks, and the null on unmarked drawings.

Use ``--smoke`` for a seconds-long run; the default takes a few minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from contourmark.attacks import standard_suite
from contourmark.geosample import GeoParameters, GeoWatermark, detect
from contourmark.toygen import toy_drawing

KEY = hashlib.sha256(b"sampler-lab-key").digest()
ATTACKS = ["svgo_default", "round_1dp", "rotate_30", "subdivide", "noise_0.1pct"]


POLYGON = GeoParameters(scheme="polygon")


def variant(name: str, seed: int) -> tuple[GeoWatermark | None, str]:
    if name == "plain":
        return None, "gamma"
    if name == "polygon-mask":
        return GeoWatermark(KEY, POLYGON, seed=seed), "gamma"
    if name == "polygon-allow":
        return GeoWatermark(KEY, POLYGON, seed=seed, reuse="allow"), "gamma"
    if name == "gumbel-mask":
        return GeoWatermark(KEY, seed=seed), "gamma"
    if name == "gumbel-allow":
        return GeoWatermark(KEY, seed=seed, reuse="allow"), "gamma"
    if name.startswith("bias-"):
        return GeoWatermark(KEY, seed=seed, mode="bias", delta=float(name.split("-")[1])), "green"
    raise SystemExit(f"unknown variant {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/geosample/sampler_lab.json"))
    parser.add_argument("--drawings", type=int, default=30)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.6, 1.0, 3.0])
    parser.add_argument("--sizes", nargs="+", default=["2x8", "5x14"], help="shapes x vertices per drawing")
    parser.add_argument("--variants", nargs="+", default=["gumbel-mask", "gumbel-allow", "bias-1", "bias-2", "bias-4"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.drawings, args.sigmas, args.sizes = 3, [1.0], ["2x8"]
    suite = standard_suite()
    rows = []
    for size in args.sizes:
        shapes, vertices = (int(v) for v in size.split("x"))
        for sigma in args.sigmas:
            plain = [toy_drawing(None, seed, shapes, vertices, sigma)[0].svg() for seed in range(args.drawings)]
            for name in args.variants:
                log_p, entropy, kl, distinct, attacked = [], [], [], [], {a: [] for a in ATTACKS}
                statistic = "gamma"
                for seed in range(args.drawings):
                    watermark, statistic = variant(name, seed)
                    state, stats = toy_drawing(watermark, seed, shapes, vertices, sigma)
                    svg = state.svg()
                    params = watermark.params if watermark is not None else GeoParameters()
                    result = detect(svg, KEY, params, statistic=statistic)
                    log_p.append(result["log10_p_value"])
                    distinct.append(result["distinct_vertices"])
                    entropy += stats["entropy"]
                    kl += stats["kl"]
                    for attack in ATTACKS:
                        try:
                            attacked[attack].append(detect(suite[attack](svg), KEY, params, statistic=statistic)["log10_p_value"])
                        except Exception:
                            attacked[attack].append(0.0)
                null = [detect(svg, KEY, params, statistic=statistic)["log10_p_value"] for svg in plain]
                rows.append({
                    "size": size, "sigma": sigma, "variant": name, "statistic": statistic, "drawings": args.drawings,
                    "entropy_nats_per_keyed_step": float(np.mean(entropy)) if entropy else None,
                    "kl_nats_per_keyed_step": float(np.mean(kl)) if kl else 0.0,
                    "distinct_descriptors_median": float(np.median(distinct)),
                    "median_log10_p": float(np.median(log_p)),
                    "tpr_1e-6": float(np.mean(np.array(log_p) <= -6)), "tpr_1e-3": float(np.mean(np.array(log_p) <= -3)),
                    "attack_tpr_1e-3": {a: float(np.mean(np.array(v) <= -3)) for a, v in attacked.items()},
                    "null_min_log10_p": float(min(null)), "null_fpr_1e-2": float(np.mean(np.array(null) <= -2)),
                })
                r = rows[-1]
                print(f"{size:5s} sigma={sigma:<4} {name:13s} H={r['entropy_nats_per_keyed_step'] or 0:.2f} KL={r['kl_nats_per_keyed_step']:.3f} "
                      f"distinct={r['distinct_descriptors_median']:.0f} median={r['median_log10_p']:7.1f} TPR6={r['tpr_1e-6']:.2f} TPR3={r['tpr_1e-3']:.2f} "
                      f"svgo={r['attack_tpr_1e-3']['svgo_default']:.2f} subdiv={r['attack_tpr_1e-3']['subdivide']:.2f} nullmin={r['null_min_log10_p']:.1f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"key_id": hashlib.sha256(KEY).hexdigest()[:16], "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
