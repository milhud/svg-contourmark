"""Build a self-contained perceptual study (no server, no dependencies).

Question: can people tell a marked drawing from its original?  The task is a
same/different judgement on a pair shown side by side, which measures
detectability of the change rather than preference.

Trials:

* ``marked``     original next to its marked version (the condition of interest);
* ``identical``  the same image twice (catch trial: "different" is a false alarm);
* ``obvious``    original next to a heavily distorted version (catch trial:
                 "same" means the participant is not looking).

Each trial is rendered at an icon size and at a large size, order and side
are randomized per participant in the browser, and answers download as CSV.
Analysis: sensitivity d' from hits on ``marked`` and false alarms on
``identical``; exclude participants who miss the ``obvious`` trials.

Output: ``<out>/index.html``, ``<out>/trials.json``, ``<out>/stimuli/*.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from contourmark.attacks import gaussian_noise, render_png
from contourmark.spectral import SpectralParameters, embed

ROOT = Path(__file__).resolve().parents[1]
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Are these two drawings the same?</title>
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#d8d7d2; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#ffffff; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; } }
body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px; }
.pair { display:flex; gap:16px; justify-content:center; flex-wrap:wrap; margin:16px 0; }
.pair img { background:#fff; border:1px solid var(--line); image-rendering:auto; max-width:100%; }
button { font:inherit; padding:10px 22px; margin:6px; border:1px solid var(--line); border-radius:8px; background:transparent; color:var(--ink); cursor:pointer; }
button:hover, button:focus-visible { border-color:var(--accent); outline:2px solid var(--accent); }
.muted { color:var(--muted); }
</style></head><body><main>
<h1>Are these two drawings the same?</h1>
<p class="muted" id="status"></p>
<div id="stage"></div>
<script>
const trials = __TRIALS__;
const order = trials.map((t, i) => i).sort(() => Math.random() - 0.5);
const answers = []; let position = 0; let shown = 0;
const stage = document.getElementById('stage'), status = document.getElementById('status');
function intro() {
  stage.innerHTML = '<p>You will see pairs of drawings. For each pair, decide whether the two images are exactly the same or different in any way. Some pairs are identical. There is no time limit. Use the buttons or press S (same) / D (different).</p><label>Participant code <input id="who" autocomplete="off"></label><p><button id="go">Start</button></p>';
  document.getElementById('go').onclick = () => { window.participant = document.getElementById('who').value || 'anonymous'; next(); };
}
function next() {
  if (position >= order.length) return done();
  const t = trials[order[position]]; const flip = Math.random() < 0.5;
  const left = flip ? t.b : t.a, right = flip ? t.a : t.b;
  status.textContent = 'Pair ' + (position + 1) + ' of ' + order.length;
  stage.innerHTML = '<div class="pair"><img alt="left drawing" width="' + t.size + '" height="' + t.size + '" src="' + left + '"><img alt="right drawing" width="' + t.size + '" height="' + t.size + '" src="' + right + '"></div><p><button id="same">Same (S)</button><button id="diff">Different (D)</button></p>';
  shown = performance.now();
  document.getElementById('same').onclick = () => record(t, flip, 'same');
  document.getElementById('diff').onclick = () => record(t, flip, 'different');
}
function record(t, flip, answer) {
  answers.push({participant: window.participant, trial: t.id, kind: t.kind, asset: t.asset, size: t.size, flipped: flip, answer: answer, ms: Math.round(performance.now() - shown), index: position});
  position += 1; next();
}
document.addEventListener('keydown', (e) => { const b = document.getElementById(e.key.toLowerCase() === 's' ? 'same' : e.key.toLowerCase() === 'd' ? 'diff' : ''); if (b) b.click(); });
function done() {
  const header = Object.keys(answers[0]);
  const csv = [header.join(',')].concat(answers.map(a => header.map(h => JSON.stringify(a[h])).join(','))).join('\\n');
  const url = URL.createObjectURL(new Blob([csv], {type: 'text/csv'}));
  status.textContent = 'Finished. Thank you.';
  stage.innerHTML = '<p><a download="study_' + window.participant + '.csv" href="' + url + '"><button>Download answers (CSV)</button></a></p>';
}
intro();
</script></main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "experiments/corpus/dev.json")
    parser.add_argument("--assets", type=int, default=24)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--delta", type=float, default=SpectralParameters.delta)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    items = json.loads(args.corpus.read_text())["items"]
    rng.shuffle(items)
    stimuli = args.output / "stimuli"
    stimuli.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(b"perceptual-study-key").digest()
    params = SpectralParameters(delta=args.delta)
    trials = []
    used = 0
    for item in items:
        if used >= args.assets:
            break
        source = (ROOT / item["path"]).read_bytes()
        try:
            marked, report = embed(source, key, params)
            if not report["marked_contours"]:
                continue
            obvious = gaussian_noise(0.02, seed=used)(source)
            name = hashlib.sha256(item["path"].encode()).hexdigest()[:10]
            for size in args.sizes:
                files = {}
                for label, data in (("orig", source), ("marked", marked), ("obvious", obvious)):
                    path = stimuli / f"{name}_{label}_{size}.png"
                    path.write_bytes(render_png(data, size))
                    files[label] = f"stimuli/{path.name}"
                trials.append({"id": f"{name}-m-{size}", "kind": "marked", "asset": item["path"], "size": size, "a": files["orig"], "b": files["marked"]})
                if used % 3 == 0:
                    trials.append({"id": f"{name}-i-{size}", "kind": "identical", "asset": item["path"], "size": size, "a": files["orig"], "b": files["orig"]})
                if used % 6 == 0:
                    trials.append({"id": f"{name}-o-{size}", "kind": "obvious", "asset": item["path"], "size": size, "a": files["orig"], "b": files["obvious"]})
        except Exception:
            continue
        used += 1
    (args.output / "trials.json").write_text(json.dumps({"delta": args.delta, "seed": args.seed, "trials": trials}, indent=1) + "\n")
    (args.output / "index.html").write_text(PAGE.replace("__TRIALS__", json.dumps(trials)))
    kinds = {k: sum(t["kind"] == k for t in trials) for k in ("marked", "identical", "obvious")}
    print(f"wrote {len(trials)} trials {kinds} for {used} assets to {args.output}/index.html")


if __name__ == "__main__":
    main()
