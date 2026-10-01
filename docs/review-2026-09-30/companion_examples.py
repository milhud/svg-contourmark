"""Executable teaching examples, NOT a replacement watermark implementation.

Run: .venv/bin/python docs/review-2026-09-30/companion_examples.py
All checks are CPU-only. No models, network requests or production edits.
"""
from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import i0e
from scipy.spatial import cKDTree
from scipy.stats import gamma


def appearance_issues(source: bytes) -> list[str]:
    """Conservative unsupported-feature gate, not a visibility oracle."""
    if b"<!DOCTYPE" in source.upper() or b"<!ENTITY" in source.upper():
        return ["DTD/entity"]
    issues = []
    if b"<?xml-stylesheet" in source.lower():
        issues.append("external stylesheet")
    root = ET.fromstring(source)
    for node in root.iter():
        name = node.tag.rsplit("}", 1)[-1]
        if name in {"style", "script", "text", "image", "foreignObject", "use",
                    "animate", "animateTransform", "animateMotion", "set"}:
            issues.append(name)
        if name == "svg" and node is not root:
            issues.append("nested viewport")
        props = dict(node.attrib)
        for declaration in node.get("style", "").split(";"):
            if ":" in declaration:
                k, v = declaration.split(":", 1)
                props[k.strip().lower()] = v.strip()
        for k in ("clip-path", "mask", "filter"):
            if props.get(k, "none").strip().lower() != "none":
                issues.append(k)
        if any("var(" in v.lower() for v in props.values()):
            issues.append("CSS variable")
    return sorted(set(issues))


def gamma_evidence(descriptors, key: bytes) -> dict:
    """Illustrate deduplication and the ideal-PRF Gamma null, no window search."""
    from contourmark.geosample import keyed_uniform
    unique = set(descriptors)
    if not unique:
        return {"n": 0, "p": 1.0}
    score = sum(-math.log1p(-keyed_uniform(key, d)) for d in unique)
    return {"n": len(unique), "p": float(gamma.sf(score, len(unique)))}


def combine_tests(p_values) -> float:
    """Bonferroni for a fixed, predeclared family (arbitrary dependence)."""
    values = list(p_values)
    if not values or any(not 0 <= p <= 1 for p in values):
        raise ValueError("provide valid p-values for every declared test")
    return min(1.0, len(values) * min(values))


def fixed_chernoff_log_bound(statistic, weights, theta=1.0):
    """Conservative analytic tail form; floating-point error is NOT certified.

    For any fixed theta >= 0, Markov gives exp(-theta*t) prod I0(theta*w).
    Optimizing theta tightens it but is not required for mathematical validity.
    """
    weights = np.asarray(weights, dtype=float)
    if theta < 0 or np.any(weights < 0):
        raise ValueError("nonnegative theta and weights required")
    z = theta * weights
    return min(0.0, float(-theta * statistic + np.sum(np.log(i0e(z)) + np.abs(z))))


def equal_arclength(polyline, count=128):
    """Uniform arc-length samples of a polyline; curves need adaptive flattening."""
    p = np.asarray(polyline, dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-12]
    p = p[keep]
    if len(p) < 2:
        raise ValueError("degenerate curve")
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    target = np.linspace(0, s[-1], count)
    return np.column_stack([np.interp(target, s, p[:, axis]) for axis in (0, 1)])


def boundary_distances(a, b):
    """Symmetric sampled-boundary metrics; NOT certified curve Hausdorff."""
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    both = np.r_[ab, ba]
    return {"mean": float(both.mean()), "p95": float(np.quantile(both, .95)),
            "max": float(both.max())}


def composite_rgba(rgba, background):
    rgba = np.asarray(rgba, dtype=float)
    alpha = rgba[..., 3:4]
    return rgba[..., :3] * alpha + np.asarray(background) * (1 - alpha)


def appearance_metrics(a, b):
    """Inputs: equally sized straight-alpha RGBA images in [0,1]."""
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError("matching RGBA arrays required")
    out = {"alpha_mae": float(np.abs(a[..., 3] - b[..., 3]).mean())}
    for name, bg in (("white", [1, 1, 1]), ("dark", [.05, .05, .05])):
        mse = float(np.mean((composite_rgba(a, bg) - composite_rgba(b, bg)) ** 2))
        out[name + "_mse"] = mse
    return out


def constrained_proposal(x0, objective, distortion, epsilon, acceptable):
    """Sketch: optimize, then independently verify and roll back on failure.

    `acceptable` must implement topology/style/render checks for the actual
    application. SLSQP convergence alone does not certify any such property.
    """
    x0 = np.asarray(x0, dtype=float)
    result = minimize(objective, x0, method="SLSQP", constraints=[
        {"type": "ineq", "fun": lambda x: epsilon - distortion(x0, x)}])
    candidate = result.x
    accepted = bool(result.success and np.isfinite(candidate).all()
                    and distortion(x0, candidate) <= epsilon
                    and acceptable(candidate))
    return (candidate if accepted else x0.copy()), accepted


def budgeted_attack(x0, propose, quality_ok, score, queries=20):
    """Score-oracle attack example: lower score means less owner evidence.

    Owner-score queries are counted, including the initial score. The caller
    also needs a separate proposal/render budget for expensive quality checks.
    """
    if queries < 1:
        raise ValueError("at least one score query required")
    best = np.asarray(x0, dtype=float).copy()
    best_score = score(best)
    used = 1
    for candidate in propose(best.copy()):
        if used >= queries:
            break
        if not quality_ok(x0, candidate):
            continue
        value = score(candidate)
        used += 1
        if value < best_score:
            best, best_score = candidate.copy(), value
    return {"candidate": best, "score": best_score, "queries": used}


def cluster_bootstrap_difference(rows, seed=7, repeats=1000):
    """Paired mean(B-A), resampling prompt clusters with all their observations.

    Rows contain prompt, a, b. Extend to hierarchical prompt/key sampling for
    real experiments; a single clustering axis is only an example.
    """
    prompts = sorted({r["prompt"] for r in rows})
    if len(prompts) < 2:
        raise ValueError("at least two clusters required")
    grouped = {p: [r["b"] - r["a"] for r in rows if r["prompt"] == p] for p in prompts}
    rng = np.random.default_rng(seed)
    draws = [np.mean([v for p in rng.choice(prompts, len(prompts)) for v in grouped[p]]) for _ in range(repeats)]
    return np.quantile(draws, [.025, .975]).tolist()


def outcome_counts(rows):
    """Keep unsupported, invalid and execution failures in the denominator."""
    counts = Counter(row["status"] for row in rows)
    return {"attempted": len(rows), "counts": dict(counts),
            "end_to_end_detection": counts["detected"] / len(rows) if rows else None}


def run_hash(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def self_check():
    plain = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0L1 1"/></svg>'
    clipped = plain.replace(b'<path ', b'<path clip-path="url(#empty)" ')
    assert appearance_issues(plain) == []
    assert appearance_issues(clipped) == ["clip-path"]
    d = [("v", 1, 2, 3)]
    assert gamma_evidence(d, b"x" * 32) == gamma_evidence(d * 10, b"x" * 32)
    assert combine_tests([.01, .2]) == .02
    a = equal_arclength([[0, 0], [2, 0], [2, 2]])
    b = equal_arclength([[0, 0], [.7, 0], [2, 0], [2, .3], [2, 2]])
    assert np.allclose(a, b)
    assert boundary_distances(a, b)["max"] < 1e-12
    rgba = np.zeros((2, 2, 4)); rgba[..., 3] = 1
    transparent = np.zeros_like(rgba)
    assert appearance_metrics(rgba, transparent)["white_mse"] == 1
    solution, accepted = constrained_proposal([0.0], lambda x: (x[0] - 1) ** 2,
                                             lambda a, b: float(np.linalg.norm(a - b)), .2, lambda _: False)
    assert not accepted and solution[0] == 0
    attack = budgeted_attack([1.], lambda x: [x - .05 * i for i in range(1, 30)],
                             lambda a, b: abs(a[0] - b[0]) <= .2, lambda x: float(x[0]), queries=3)
    assert attack["queries"] == 3 and attack["score"] < 1
    assert outcome_counts([{"status": "detected"}, {"status": "unsupported"}])["end_to_end_detection"] == .5
    assert run_hash({"a": 1, "b": 2}) == run_hash({"b": 2, "a": 1})
    assert fixed_chernoff_log_bound(1, [1, 1]) <= 0
    assert cluster_bootstrap_difference([{"prompt": p, "a": 0, "b": 1} for p in ("a", "b")]) == [1., 1.]
    return {"status": "passed", "checks": ["unsupported effects", "deduplication", "multiple tests",
            "polyline subdivision", "symmetric boundary distances", "alpha compositing",
            "constraint rollback", "attack query budget", "failure denominator", "run identity",
            "Chernoff form", "paired cluster bootstrap"]}


if __name__ == "__main__":
    print(json.dumps(self_check(), indent=2))
