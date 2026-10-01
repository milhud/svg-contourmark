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
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models/iconshop"))

from contourmark.geosample import GeoParameters, GeoWatermark, detect  # noqa: E402
from contourmark.point_token_models import DecodeState, IconShopGrammar, watermarked_step  # noqa: E402

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
    model.load_state_dict(state, strict=False)
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
            token, was_keyed = watermarked_step(states[sample], logits[row], watermark, rngs[sample], top_p=top_p, temperature=temperature)
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
    return [{"prompt": p, "seed": s, "tokens": t, "keyed_steps": k, "svg": st.svg(), "seconds": elapsed / batch}
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
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be in [0, --num-shards)")
    device = torch.device(args.device)
    model, tokenizer, cfg = load_model(device)
    key = hashlib.sha256(b"iconshop-geosample-evaluation-key").digest()
    wrong = hashlib.sha256(b"iconshop-geosample-wrong-key").digest()
    args.output.mkdir(parents=True, exist_ok=True)
    # One log per shard so concurrent GPU workers never write the same file;
    # merge with: cat samples.shard*.jsonl > samples.jsonl
    log = args.output / (f"samples.shard{args.shard}.jsonl" if args.num_shards > 1 else "samples.jsonl")
    done = {(r["prompt"], r["seed"], r["marked"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    jobs = [(p, args.seed_base + i) for p in args.prompts for i in range(args.samples_per_prompt)]
    jobs = jobs[args.shard::args.num_shards]
    for marked in (False, True):
        todo = [(p, s) for p, s in jobs if (p, s, marked) not in done]
        for start in range(0, len(todo), args.batch):
            chunk = todo[start:start + args.batch]
            watermark = GeoWatermark(key, seed=chunk[0][1]) if marked else None
            results = generate(model, tokenizer, cfg, [p for p, _ in chunk], [s for _, s in chunk], watermark, device, args.top_p, args.temperature, args.max_tokens)
            with log.open("a") as stream:
                for result in results:
                    name = f"{'wm' if marked else 'plain'}_{result['prompt'].replace(' ', '_')}_{result['seed']}.svg"
                    (args.output / name).write_bytes(result["svg"])
                    check = detect(result["svg"], key)
                    record = {k: v for k, v in result.items() if k != "svg"}
                    record.update({"marked": marked, "file": name, "log10_p": check["log10_p_value"], "distinct_vertices": check["distinct_vertices"],
                                   "wrong_key_log10_p": detect(result["svg"], wrong)["log10_p_value"], "top_p": args.top_p, "temperature": args.temperature})
                    stream.write(json.dumps(record) + "\n")
            print(f"{'marked' if marked else 'plain'} {start + len(chunk)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
