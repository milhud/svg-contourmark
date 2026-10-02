"""Adapters connecting point-token SVG generators to the geosample watermark.

Point-token models (IconShop, OmniSVG, and their relatives) emit a command
token followed by one token per point on a fixed grid.  An adapter only has to
describe that grammar; the watermark itself (``geosample.GeoWatermark``) and
the detector are shared.

* ``IconShopGrammar``: the public IconShop checkpoint (200x200 grid; Move
  carries two points, Line one, Curve three: c1, c2, end).
* ``OmniSVGGrammar``: OmniSVG 1.1 token layout (one token per point on a
  200x200 grid; M/L one point, C three points, Z none).  Token ids come from
  the model's ``config.yaml``.
* ``GeoWatermarkLogitsProcessor``: a Hugging Face ``LogitsProcessor`` that
  makes the keyed choice inside ``model.generate`` for any grammar here.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .geosample import GeoWatermark, PathTracker
from .geometry import SVG_NS


class PointTokenGrammar:
    """Describe a point-token SVG vocabulary."""

    grid: int = 200
    #: command name -> number of point tokens that follow it
    arity: dict[str, int] = {}

    def command(self, token: int) -> str | None:  # pragma: no cover - interface
        raise NotImplementedError

    def point(self, token: int) -> complex | None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_end(self, token: int) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def point_token_range(self) -> range:  # pragma: no cover - interface
        raise NotImplementedError


class IconShopGrammar(PointTokenGrammar):
    """Token values *after* subtracting the text vocabulary size."""

    grid = 200
    arity = {"M": 2, "L": 1, "C": 3}
    _commands = {3: "M", 4: "L", 5: "C"}
    offset = 6  # 3 END tokens + 1 SVG_END + 2 mask tokens precede the grid

    def command(self, token: int) -> str | None:
        return self._commands.get(int(token))

    def point(self, token: int) -> complex | None:
        token = int(token)
        if token < self.offset or token >= self.offset + self.grid * self.grid:
            return None
        q = token - self.offset
        return complex(q % self.grid, q // self.grid)

    def is_end(self, token: int) -> bool:
        return int(token) == 0

    def point_token_range(self) -> range:
        return range(self.offset, self.offset + self.grid * self.grid)


@dataclass
class OmniSVGGrammar(PointTokenGrammar):
    """OmniSVG 1.1 layout; pass the ids from the released ``config.yaml``.

    ``coordinate_start`` is the first coordinate token (151944 for 4B, 152072
    for 8B in the public configuration); coordinates are row-major on a
    ``grid`` x ``grid`` lattice.  Command token ids must be given explicitly.
    """

    move: int = -1
    line: int = -1
    curve: int = -1
    close: int = -1
    end: int = -1
    coordinate_start: int = 151_944
    grid: int = 200
    arity: dict[str, int] = field(default_factory=lambda: {"M": 1, "L": 1, "C": 3, "Z": 0})

    def command(self, token: int) -> str | None:
        return {self.move: "M", self.line: "L", self.curve: "C", self.close: "Z"}.get(int(token))

    def point(self, token: int) -> complex | None:
        q = int(token) - self.coordinate_start
        if not 0 <= q < self.grid * self.grid:
            return None
        return complex(q % self.grid, q // self.grid)

    def is_end(self, token: int) -> bool:
        return int(token) == self.end

    def point_token_range(self) -> range:
        return range(self.coordinate_start, self.coordinate_start + self.grid * self.grid)


@dataclass
class DecodeState:
    """Incremental parse of a point-token stream into path geometry."""

    grammar: PointTokenGrammar
    tracker: PathTracker = field(default_factory=PathTracker)
    pending: str | None = None
    points: list[complex] = field(default_factory=list)
    subpaths: list[list[tuple[str, list[complex]]]] = field(default_factory=list)
    start: complex | None = None

    def feed(self, token: int) -> None:
        name = self.grammar.command(token)
        if name is not None:
            self.pending, self.points = name, []
            if name == "Z":
                self._commit()
            return
        point = self.grammar.point(token)
        if point is None or self.pending is None:
            return
        self.points.append(point)
        if len(self.points) == self.grammar.arity[self.pending]:
            self._commit()

    def _commit(self) -> None:
        name, points = self.pending, self.points
        if name == "M":
            target = points[-1]
            self.tracker.move_to(target)
            self.subpaths.append([("M", [target])])
            self.start = target
        elif name == "L" and self.subpaths:
            self.tracker.line_to(points[0])
            self.subpaths[-1].append(("L", points))
        elif name == "C" and self.subpaths:
            self.tracker.cubic_to(points[0], points[1], points[2])
            self.subpaths[-1].append(("C", points))
        elif name == "Z" and self.subpaths:
            if self.start is not None and self.tracker.current is not None and self.tracker.current != self.start:
                self.tracker.line_to(self.start)
            self.subpaths[-1].append(("Z", []))
        # A command token repeats until another command (implicit repetition).
        self.points = []

    def handle_slot(self) -> bool:
        """Whether the next point token is the first control point of a curve."""
        return self.pending == "C" and not self.points and bool(self.subpaths) and bool(self.tracker.segments)

    def endpoint_slot(self) -> tuple[bool, list[complex]]:
        """Whether the next point token is a segment endpoint, plus known controls."""
        if self.pending in ("L", "C") and self.subpaths:
            needed = self.grammar.arity[self.pending]
            if len(self.points) == needed - 1:
                return True, list(self.points)
        return False, []

    def svg(self, stroke: bool = False) -> bytes:
        root = ET.Element(f"{{{SVG_NS}}}svg", {"viewBox": f"0 0 {self.grammar.grid} {self.grammar.grid}"})
        for subpath in self.subpaths:
            parts = []
            for name, points in subpath:
                if name == "Z":
                    parts.append("Z")
                else:
                    parts.append(name + " ".join(f"{p.real:g} {p.imag:g}" for p in points))
            if len(parts) > 1:
                attributes = {"d": "".join(parts)}
                attributes.update({"fill": "none", "stroke": "black"} if stroke else {"fill": "black"})
                ET.SubElement(root, f"{{{SVG_NS}}}path", attributes)
        ET.register_namespace("", SVG_NS)
        return ET.tostring(root, encoding="unicode").encode()


def truncate(logits: np.ndarray, top_p: float = 1.0, top_k: int = 0, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (token ids, probabilities) after the decoder's own truncation."""
    scaled = logits.astype(np.float64) / temperature
    order = np.argsort(-scaled)
    if top_k > 0:
        order = order[:top_k]
    values = scaled[order]
    values = values - values.max()
    probs = np.exp(values)
    probs /= probs.sum()
    if 0 < top_p < 1:
        keep = int(np.searchsorted(np.cumsum(probs), top_p) + 1)
        order, probs = order[:keep], probs[:keep] / probs[:keep].sum()
    # Tokens with zero probability (masked logits, underflow) cannot be drawn.
    positive = probs > 0
    return order[positive], probs[positive] / probs[positive].sum()


def watermarked_step(state: DecodeState, logits: np.ndarray, watermark: GeoWatermark | None, rng: np.random.Generator,
                     top_p: float = 1.0, top_k: int = 0, temperature: float = 1.0, token_offset: int = 0) -> tuple[int, bool]:
    """Choose the next token; keyed at segment endpoints and first curve handles.

    ``logits`` covers the model vocabulary; ``token_offset`` converts a logit
    index to the grammar's token value (IconShop subtracts the text vocabulary).
    Returns (logit index, whether the keyed choice was applied).
    """
    ids, probs = truncate(logits, top_p, top_k, temperature)
    if watermark is not None:
        state.tracker.params = watermark.params  # descriptors must use the sampler's quantization
    if watermark is not None and watermark.params.scheme == "polygon":
        # Every point of a line or curve is a keyed decision.
        if state.pending in ("L", "C") and state.subpaths and len(ids) > 1 and len(state.points) < state.grammar.arity[state.pending]:
            points = [state.grammar.point(int(i) - token_offset) for i in ids]
            if any(point is not None for point in points):
                choice = watermark.choose_point(state.tracker, state.points, points, probs, line=state.pending == "L")
                return int(ids[choice.index]), choice.keyed
        return int(rng.choice(ids, p=probs)), False
    endpoint, controls = state.endpoint_slot()
    if watermark is not None and len(ids) > 1 and state.handle_slot():
        handles = [state.grammar.point(int(i) - token_offset) for i in ids]
        if any(handle is not None for handle in handles):
            choice = watermark.choose_handle(state.tracker, handles, probs)
            return int(ids[choice.index]), choice.keyed
    if watermark is not None and endpoint and len(ids) > 1:
        # Non-point candidates (e.g. a command or end token) are vertex-free
        # and compete with unkeyed uniforms inside ``choose``.
        ends = [state.grammar.point(int(i) - token_offset) for i in ids]
        if any(end is not None for end in ends):
            choice = watermark.choose(state.tracker, ends, probs, [controls] * len(ids) if controls else None)
            return int(ids[choice.index]), choice.keyed
    return int(rng.choice(ids, p=probs)), False


class GeoWatermarkLogitsProcessor:
    """Hugging Face ``LogitsProcessor`` for point-token models using ``generate``.

    It replays the generated tokens through the grammar, and at endpoint slots
    replaces the distribution by a one-hot on the keyed choice.  Use it with
    ``do_sample=True`` and pass the same ``top_p``/``top_k``/``temperature``
    here (and none to ``generate``) so the keyed choice sees the decoder's
    real truncated distribution.

    Each batch row is one drawing and owns its sampler state (the set of
    descriptors already looked at) and its random streams, so results do not
    depend on batch composition.  Rows must keep their identity across steps:
    use sampling, not beam search.
    """

    def __init__(self, grammar: PointTokenGrammar, watermark: GeoWatermark, prompt_length: int,
                 top_p: float = 1.0, top_k: int = 0, temperature: float = 1.0, seed: int = 0):
        self.grammar = grammar
        self.watermark = watermark
        self.prompt_length = prompt_length
        self.top_p, self.top_k, self.temperature = top_p, top_k, temperature
        self.seed = seed
        self.rows: dict[int, tuple[GeoWatermark, np.random.Generator]] = {}

    def _row(self, row: int, fresh: bool) -> tuple[GeoWatermark, np.random.Generator]:
        if fresh or row not in self.rows:
            template = self.watermark
            self.rows[row] = (
                GeoWatermark(template.key, template.params, seed=self.seed * 1_000_003 + row, reuse=template.reuse),
                np.random.default_rng([self.seed, row]),
            )
        return self.rows[row]

    def __call__(self, input_ids, scores):  # torch tensors
        import torch

        out = torch.full_like(scores, -math.inf)
        for row in range(scores.shape[0]):
            state = DecodeState(self.grammar)
            history = input_ids[row, self.prompt_length:].tolist()
            for token in history:
                state.feed(token)
            watermark, rng = self._row(row, fresh=not history)
            chosen, _ = watermarked_step(state, scores[row].float().cpu().numpy(), watermark, rng, self.top_p, self.top_k, self.temperature)
            out[row, chosen] = 0.0
        return out


def decode_tokens(grammar: PointTokenGrammar, tokens: Iterable[int]) -> DecodeState:
    state = DecodeState(grammar)
    for token in tokens:
        if grammar.is_end(token):
            break
        state.feed(token)
    return state
