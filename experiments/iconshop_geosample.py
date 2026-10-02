"""Run the IconShop checkpoint with and without the inference-time watermark.

For each prompt and seed, the same model, prompt, truncation and RNG seed are
used twice: once with ordinary sampling and once with the keyed geosample
choice at segment endpoints.  Outputs are SVG files plus a JSONL record per
sample (tokens, timing, keyed-step count, detection).

The checkpoint (m1357l/iconshop-svg-generator, CC BY-NC-SA 4.0) is downloaded
to models/iconshop, which is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import importlib.metadata
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/iconshop"))

from contourmark.geosample import GeoParameters, GeoWatermark, detect  # noqa: E402
from contourmark.point_token_models import DecodeState, IconShopGrammar, watermarked_step  # noqa: E402
from run_identity import evaluation_key, file_digest, prepare_run  # noqa: E402

PROMPTS = [
    "star", "heart", "house", "car", "rocket", "calendar", "cat", "dog", "tree", "flower",
    "camera", "bell", "cloud", "sun", "moon", "fish", "bird", "key", "lock", "phone",
    "book", "music", "umbrella", "cup", "bicycle", "airplane", "gift", "shopping cart", "envelope", "lightning",
]


def load_model(device: torch.device):
    from model.decoder import SketchDecoder
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    cfg = json.loads((ROOT / "models/iconshop/config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_name"])
    model = SketchDecoder(
        config={k: cfg[k] for k in ("hidden_dim", "embed_dim", "num_layers", "num_heads", "dropout_rate")},
        pix_len=cfg["pix_len"], text_len=cfg["text_len"], num_text_token=tokenizer.vocab_size,
        word_emb_path=str(ROOT / "models/iconshop/word_embedding_512.pt"), pos_emb_path=None,
    )
    state = load_file(str(ROOT / "models/iconshop/model.safetensors"))
    if any(k.startswith("model.") for k in state):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"checkpoint mismatch: {incompatible}")
    return model.to(device).eval(), tokenizer, cfg


def pixel_to_xy(pixel: int) -> list[int]:
    """IconShop's input coordinate embedding for a sampled token (see model/decoder.py)."""
    if pixel >= 6:
        q = pixel - 6
        return [q % 200 + 6, q // 200 + 6]
    return [pixel, pixel]


@torch.no_grad()
def generate(model, tokenizer, cfg, prompts: list[str], seeds: list[int], watermark: GeoWatermark | None,
             device: torch.device, top_p: float, temperature: float, max_tokens: int) -> list[dict]:
    grammar = IconShopGrammar()
    batch = len(prompts)
    text = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=cfg["text_len"],
                     return_token_type_ids=False)["input_ids"].to(device)
    num_text = tokenizer.vocab_size
    rngs = [np.random.default_rng(seed) for seed in seeds]
    # Each sample owns every random stream. Sharding/batch membership must
    # not consume another sample's free-group or within-group randomness.
    watermarks = [GeoWatermark(watermark.key, watermark.params, seed=seed, reuse=watermark.reuse, mode=watermark.mode,
                               delta=watermark.delta, gamma=watermark.gamma) if watermark else None for seed in seeds]
    states = [DecodeState(grammar) for _ in range(batch)]
    tokens: list[list[int]] = [[] for _ in range(batch)]
    keyed = [0] * batch
    alive = list(range(batch))
    pixel_seq = xy_seq = None
    started = time.time()
    for _ in range(min(max_tokens, cfg["pix_len"])):
        if not alive:
            break
        logits = model.forward([None] * len(alive) if pixel_seq is None else pixel_seq, [None, None] * len(alive) if xy_seq is None else xy_seq, None, text[alive])[:, -1, num_text:]
        logits = logits.float().cpu().numpy()
        next_tokens = []
        for row, sample in enumerate(alive):
            token, was_keyed = watermarked_step(states[sample], logits[row], watermarks[sample], rngs[sample], top_p=top_p, temperature=temperature)
            keyed[sample] += was_keyed
            tokens[sample].append(token)
            if token != 0:
                states[sample].feed(token)
            next_tokens.append(token)
        step_pix = torch.tensor(next_tokens, device=device).view(-1, 1)
        step_xy = torch.tensor([pixel_to_xy(t) for t in next_tokens], device=device).unsqueeze(1)
        pixel_seq = step_pix if pixel_seq is None else torch.cat([pixel_seq, step_pix], 1)
        xy_seq = step_xy if xy_seq is None else torch.cat([xy_seq, step_xy], 1)
        keep = [row for row, token in enumerate(next_tokens) if token != 0]
        alive = [alive[row] for row in keep]
        pixel_seq, xy_seq = pixel_seq[keep], xy_seq[keep]
    elapsed = time.time() - started
    return [{"prompt": p, "seed": s, "tokens": t, "keyed_steps": k, "svg": st.svg(), "seconds": elapsed / batch,
             "completed": bool(t and t[-1] == 0), "truncated": not bool(t and t[-1] == 0)}
            for p, s, t, k, st in zip(prompts, seeds, tokens, keyed, states)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--prompts", nargs="*", default=PROMPTS)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--shard", type=int, default=0, help="this worker's index (e.g. one per GPU)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--mode", choices=["gumbel", "bias"], default="gumbel", help="gumbel preserves the distribution; bias trades it for power")
    parser.add_argument("--delta", type=float, default=2.0, help="bias strength in nats (bias mode)")
    parser.add_argument("--gamma", type=float, default=0.5, help="green fraction (bias mode)")
    parser.add_argument("--reuse", choices=["mask", "allow"], default="mask", help="allow: naive score reuse (ablation, not distribution-preserving)")
    parser.add_argument("--key-label", default="evaluation", help="selects a public benchmark key")
    parser.add_argument("--scheme", choices=["vertex", "polygon"], default="vertex", help="polygon keys every control-polygon point, in context")
    parser.add_argument("--marked-only", action="store_true", help="skip plain samples (they do not depend on the sampler variant)")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be in [0, --num-shards)")
    if args.batch < 1 or args.samples_per_prompt < 1 or args.max_tokens < 1:
        raise SystemExit("batch, samples-per-prompt and max-tokens must be positive")
    device = torch.device(args.device)
    params = GeoParameters(scheme=args.scheme)
    key = evaluation_key(args.key_label)
    wrong = evaluation_key(args.key_label, wrong=True)
    statistic = "green" if args.mode == "bias" else "gamma"
    sampler = {"mode": args.mode, "reuse": args.reuse}
    if args.mode == "bias":
        sampler.update({"delta": args.delta, "gamma": args.gamma})
    args.output.mkdir(parents=True, exist_ok=True)
    # One log per shard so concurrent GPU workers never write the same file;
    # merge with: cat samples.shard*.jsonl > samples.jsonl
    log = args.output / (f"samples.shard{args.shard}.jsonl" if args.num_shards > 1 else "samples.jsonl")
    files = sorted((ROOT / "src/contourmark").glob("*.py")) + [Path(__file__), ROOT / "experiments/run_identity.py"]
    files += sorted((ROOT / "models/iconshop").rglob("*.py"))
    files += [ROOT / "models/iconshop" / name for name in ("config.json", "model.safetensors", "word_embedding_512.pt")]
    configuration = {
        "schema": "iconshop-geosample-run-v2", "prompts": args.prompts,
        "seed_base": args.seed_base, "samples_per_prompt": args.samples_per_prompt,
        "top_p": args.top_p, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "shard": args.shard, "num_shards": args.num_shards, "device": str(device),
        "watermark": asdict(params), "key_id": hashlib.sha256(key).hexdigest(),
        "sampler": sampler, "key_label": args.key_label, "marked_only": args.marked_only,
        "files": {str(path.relative_to(ROOT)): file_digest(path) for path in files},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy")},
    }
    run_id = prepare_run(log, configuration)
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
    if any(r.get("run_id") != run_id for r in records):
        raise ValueError("sample log contains mismatched run identities")
    done = {(r["prompt"], r["seed"], r["marked"]) for r in records if (args.output / r["file"]).exists()}
    if len(done) != len(records):
        raise ValueError("duplicate or missing sample artifacts; repair log before resuming")
    model, tokenizer, cfg = load_model(device)
    jobs = [(p, args.seed_base + i) for p in args.prompts for i in range(args.samples_per_prompt)]
    jobs = jobs[args.shard::args.num_shards]
    for marked in ((True,) if args.marked_only else (False, True)):
        todo = [(p, s) for p, s in jobs if (p, s, marked) not in done]
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            watermark = GeoWatermark(key, params, seed=chunk[0][1], reuse=args.reuse, mode=args.mode, delta=args.delta, gamma=args.gamma) if marked else None
            results = generate(model, tokenizer, cfg, [p for p, _ in chunk], [s for _, s in chunk], watermark, device, args.top_p, args.temperature, args.max_tokens)
            with log.open("a") as stream:
                for result in results:
                    name = f"{'wm' if marked else 'plain'}_{result['prompt'].replace(' ', '_')}_{result['seed']}.svg"
                    (args.output / name).write_bytes(result["svg"])
                    check = detect(result["svg"], key, params, statistic=statistic, gamma=args.gamma)
                    record = {k: v for k, v in result.items() if k != "svg"}
                    record.update({"run_id": run_id, "marked": marked, "file": name, "log10_p": check["log10_p_value"], "distinct_vertices": check["distinct_vertices"],
                                   "wrong_key_log10_p": detect(result["svg"], wrong, params, statistic=statistic, gamma=args.gamma)["log10_p_value"], "top_p": args.top_p, "temperature": args.temperature})
                    stream.write(json.dumps(record) + "\n")
            print(f"{'marked' if marked else 'plain'} {start + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
