"""Seeded Monte Carlo calibration and power study for assisted detection.

The conditional test is valid only when recognition does not depend on whether
the observed choice matches the secret keyed winner.  This experiment includes
an intentionally outcome-biased erasure mechanism to quantify that boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Callable

from contourmark.inference import _binomial_tail


ALPHAS = (0.01, 0.001)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def categorical(rng: random.Random, probabilities: tuple[float, ...]) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def summarize(counts: dict[float, int], trials: int) -> dict[str, dict[str, float | int | list[float]]]:
    return {
        str(alpha): {
            "detections": counts[alpha],
            "trials": trials,
            "rate": counts[alpha] / trials,
            "wilson_95": wilson(counts[alpha], trials),
        }
        for alpha in ALPHAS
    }


def binary_null(
    rng: random.Random,
    steps: int,
    trials: int,
    keep_match: float,
    keep_mismatch: float,
) -> dict:
    conditional_counts = {alpha: 0 for alpha in ALPHAS}
    conservative_counts = {alpha: 0 for alpha in ALPHAS}
    tail_cache: dict[tuple[int, int], float] = {}

    def tail(total: int, matches: int) -> float:
        key = (total, matches)
        if key not in tail_cache:
            tail_cache[key] = _binomial_tail([0.5] * total, matches) if total else 1.0
        return tail_cache[key]

    recognized_total = 0
    for _ in range(trials):
        matched = 0
        recognized = 0
        for _step in range(steps):
            is_match = rng.getrandbits(1) == 1
            keep_probability = keep_match if is_match else keep_mismatch
            if rng.random() < keep_probability:
                recognized += 1
                matched += int(is_match)
        recognized_total += recognized
        conditional = tail(recognized, matched)
        conservative = tail(steps, matched)
        for alpha in ALPHAS:
            conditional_counts[alpha] += conditional <= alpha
            conservative_counts[alpha] += conservative <= alpha
    return {
        "steps": steps,
        "keep_match": keep_match,
        "keep_mismatch": keep_mismatch,
        "mean_recognized": recognized_total / trials,
        "conditional": summarize(conditional_counts, trials),
        "conservative": summarize(conservative_counts, trials),
    }


def weighted_null(rng: random.Random, steps: int, trials: int, probabilities: tuple[float, ...]) -> dict:
    counts = {alpha: 0 for alpha in ALPHAS}
    for _ in range(trials):
        null_probabilities = []
        matches = 0
        for _step in range(steps):
            expected = categorical(rng, probabilities)
            observed = categorical(rng, probabilities)
            null_probabilities.append(probabilities[expected])
            matches += expected == observed
        p_value = _binomial_tail(null_probabilities, matches)
        for alpha in ALPHAS:
            counts[alpha] += p_value <= alpha
    return {
        "steps": steps,
        "candidate_probabilities": probabilities,
        "conditional": summarize(counts, trials),
    }


def binary_power(rng: random.Random, steps: int, trials: int, match_probability: float) -> dict:
    counts = {alpha: 0 for alpha in ALPHAS}
    tails = [_binomial_tail([0.5] * steps, matches) for matches in range(steps + 1)]
    for _ in range(trials):
        matches = sum(rng.random() < match_probability for _ in range(steps))
        p_value = tails[matches]
        for alpha in ALPHAS:
            counts[alpha] += p_value <= alpha
    return {
        "steps": steps,
        "match_probability": match_probability,
        "power": summarize(counts, trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100_000)
    parser.add_argument("--weighted-trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.weighted_trials < 1:
        raise SystemExit("trial counts must be positive")
    rng = random.Random(args.seed)

    erasure_models: dict[str, tuple[float, float]] = {
        "none": (1.0, 1.0),
        "independent_75_percent": (0.75, 0.75),
        "outcome_biased": (0.95, 0.25),
    }
    null_results = {
        name: [binary_null(rng, steps, args.trials, *keep) for steps in (16, 32, 64)]
        for name, keep in erasure_models.items()
    }
    weighted_results = [
        weighted_null(rng, steps, args.weighted_trials, (0.1, 0.3, 0.6))
        for steps in (16, 32)
    ]
    power_results = [
        binary_power(rng, steps, args.trials, probability)
        for steps in (16, 32, 64)
        for probability in (1.0, 0.9, 0.75, 0.6)
    ]
    report = {
        "seed": args.seed,
        "binary_trials_per_cell": args.trials,
        "weighted_trials_per_cell": args.weighted_trials,
        "alphas": ALPHAS,
        "binary_null": null_results,
        "weighted_null": weighted_results,
        "binary_power": power_results,
        "interpretation": {
            "independent_erasure": "Conditional p-values should remain super-uniform when recognition is independent of match status.",
            "outcome_biased_erasure": "Conditional p-values are not valid when recognition preferentially retains matches; this is an intentional counterexample.",
            "conservative": "Counting every unrecognized step as a failure avoids that selection effect at the cost of power.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
