"""A small point-token "generator" for CPU experiments on inference watermarks.

It is not a model of any real generator.  It drafts noisy polygons whose
edges are lines or cubics, emitting IconShop-style tokens, and at every point
token offers a Gaussian distribution over nearby grid points.  ``sigma``
controls how much choice each step has, which lets sampler variants be
compared at a per-step entropy matched to a real decoder (IconShop at its
default top-p measures about 1 nat per keyed step; at top-p 0.9 about 1.9).
"""

from __future__ import annotations

import math

import numpy as np

from .geosample import GeoWatermark
from .point_token_models import DecodeState, IconShopGrammar, watermarked_step

GRAMMAR = IconShopGrammar()
_VOCAB = GRAMMAR.offset + GRAMMAR.grid * GRAMMAR.grid
_Q = np.arange(GRAMMAR.grid * GRAMMAR.grid)
_X, _Y = _Q % GRAMMAR.grid, _Q // GRAMMAR.grid


def point_logits(target: complex, sigma: float, window: float | None = None) -> np.ndarray:
    """Gaussian logits over grid points near ``target``; masked elsewhere."""
    window = 3.5 * sigma if window is None else window
    target = complex(min(max(target.real, window + 1), GRAMMAR.grid - window - 2), min(max(target.imag, window + 1), GRAMMAR.grid - window - 2))
    logits = np.full(_VOCAB, -np.inf)
    distance2 = (_X - target.real) ** 2 + (_Y - target.imag) ** 2
    logits[GRAMMAR.offset:] = np.where(distance2 <= window ** 2, -distance2 / (2 * sigma ** 2), -np.inf)
    return logits


def toy_drawing(watermark: GeoWatermark | None, seed: int, shapes: int = 5, vertices: int = 14, sigma: float = 3.0,
                curve_fraction: float = 0.5, top_p: float = 1.0) -> tuple[DecodeState, dict]:
    """Generate one drawing; returns the decode state and per-step statistics."""
    rng = np.random.default_rng(seed)
    state = DecodeState(GRAMMAR)
    stats = {"keyed_steps": 0, "entropy": [], "kl": []}

    def draw(target: complex) -> None:
        token, keyed = watermarked_step(state, point_logits(target, sigma), watermark, rng, top_p=top_p)
        if watermark is not None and keyed:
            stats["keyed_steps"] += 1
            stats["entropy"].append(watermark.last_entropy)
            if watermark.mode == "bias":
                stats["kl"].append(watermark.last_kl)
        state.feed(token)

    for _ in range(shapes):
        cx, cy = 40 + 120 * rng.random(), 40 + 120 * rng.random()
        radius = 12 + 18 * rng.random()
        points = [complex(cx + radius * (1 + 0.3 * rng.standard_normal()) * math.cos(2 * math.pi * k / vertices),
                          cy + radius * (1 + 0.3 * rng.standard_normal()) * math.sin(2 * math.pi * k / vertices)) for k in range(vertices)]
        state.feed(3)
        draw(points[0])
        draw(points[0])
        for k in range(1, vertices + 1):
            end, start = points[k % vertices], points[k - 1]
            if rng.random() >= curve_fraction:
                state.feed(4)
            else:
                state.feed(5)
                draw((2 * start + end) / 3 + 3j)
                draw((start + 2 * end) / 3 - 3j)
            draw(end)
    return state, stats
