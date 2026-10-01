"""Prompt fidelity of generated icons, measured with CLIP.

Detection strength can be bought with worse drawings (for example by raising
top-p), so every sampler setting needs a quality number next to its detection
rate.  For each generated icon this renders the SVG and computes:

* ``similarity``: CLIP cosine similarity between the image and its own prompt;
* ``top1`` / ``top5``: whether the icon's own prompt is the best (or among the
  five best) matches out of all prompts in the run (retrieval accuracy).

Results are reported for plain and marked samples separately, with
prompt-level bootstrap intervals and the paired difference.  CLIP is a coarse
judge of line art and misses geometric damage; treat it as a sanity check on
prompt adherence, not as a perceptual study.

Needs a renderer (rsvg-convert) and downloads openai/clip-vit-base-patch32
(about 600 MB) on first use.  GPU optional.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from contourmark.attacks import render_png

TEMPLATE = "a simple black icon of a {}"


def embed_images(model, processor, images: list[Image.Image], device, batch: int = 32) -> np.ndarray:
    out = []
    for start in range(0, len(images), batch):
        inputs = processor(images=images[start:start + batch], return_tensors="pt").to(device)
        features = model.get_image_features(**inputs)
        features = features if isinstance(features, torch.Tensor) else features.pooler_output
        out.append(torch.nn.functional.normalize(features, dim=-1).cpu().numpy())
    return np.concatenate(out)


def bootstrap(values: dict[str, list[float]], rng, resamples: int = 2000) -> list[float]:
    """Percentile interval of the mean, resampling prompts (the independent unit)."""
    prompts = list(values)
    means = []
    for _ in range(resamples):
        chosen = rng.choice(len(prompts), len(prompts))
        means.append(np.mean([v for i in chosen for v in values[prompts[i]]]))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--limit", type=int, help="use only the first N samples per directory (smoke test)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device(args.device)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    rng = np.random.default_rng(20261001)
    report = {"model": args.model, "template": TEMPLATE, "runs": {}}
    for directory in args.samples_dir:
        records = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines() if line.strip()]
        records = [r for r in records if (directory / r["file"]).exists()]
        if args.limit:
            records = records[: args.limit]
        prompts = sorted({r["prompt"] for r in records})
        text = processor(text=[TEMPLATE.format(p) for p in prompts], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text)
        text_features = text_features if isinstance(text_features, torch.Tensor) else text_features.pooler_output
        text_features = torch.nn.functional.normalize(text_features, dim=-1).cpu().numpy()
        images, kept = [], []
        for record in records:
            try:
                images.append(Image.open(io.BytesIO(render_png((directory / record["file"]).read_bytes(), 224))).convert("RGB"))
                kept.append(record)
            except Exception:
                continue  # unrenderable output: counted below as a failure
        scores = embed_images(model, processor, images, device) @ text_features.T
        entry: dict = {"samples": len(records), "rendered": len(kept)}
        groups = {}
        for label, marked in (("plain", False), ("marked", True)):
            rows = [(r, scores[i]) for i, r in enumerate(kept) if r["marked"] == marked]
            if not rows:
                continue
            similarity, top1, top5 = {}, {}, {}
            for record, row in rows:
                own = prompts.index(record["prompt"])
                rank = int((row > row[own]).sum())
                similarity.setdefault(record["prompt"], []).append(float(row[own]))
                top1.setdefault(record["prompt"], []).append(float(rank == 0))
                top5.setdefault(record["prompt"], []).append(float(rank < 5))
            groups[label] = (similarity, top1, top5)
            flat = lambda d: [v for values in d.values() for v in values]  # noqa: E731
            entry[label] = {
                "n": len(rows),
                "similarity": {"mean": float(np.mean(flat(similarity))), "ci95": bootstrap(similarity, rng)},
                "top1": {"mean": float(np.mean(flat(top1))), "ci95": bootstrap(top1, rng)},
                "top5": {"mean": float(np.mean(flat(top5))), "ci95": bootstrap(top5, rng)},
            }
        if "plain" in groups and "marked" in groups:
            shared = sorted(set(groups["plain"][1]) & set(groups["marked"][1]))
            differences = []
            for _ in range(2000):
                chosen = rng.choice(len(shared), len(shared))
                differences.append(np.mean([np.mean(groups["marked"][1][shared[i]]) - np.mean(groups["plain"][1][shared[i]]) for i in chosen]))
            entry["top1_marked_minus_plain"] = {"mean": float(np.mean(differences)), "ci95": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))]}
        report["runs"][str(directory)] = entry
        for label in ("plain", "marked"):
            if label in entry:
                e = entry[label]
                print(f"{directory.name:28s} {label:6s} n={e['n']:4d} similarity={e['similarity']['mean']:.3f} top1={e['top1']['mean']:.2f} [{e['top1']['ci95'][0]:.2f}, {e['top1']['ci95'][1]:.2f}] top5={e['top5']['mean']:.2f}")
        if "top1_marked_minus_plain" in entry:
            d = entry["top1_marked_minus_plain"]
            print(f"{directory.name:28s} marked - plain top1: {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
