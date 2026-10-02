"""Inference-time, distribution-preserving watermarking for SVG generators.

Text watermarks key the *choice of the next token*.  For SVG that is not
enough: optimizers rewrite every token while keeping the drawing.  This
module keys the choice of the next *point* by a geometric quantity of the
drawing that survives rewriting.

When a generator is about to emit the endpoint of a line or curve, every
candidate endpoint completes a *vertex*: the corner at the segment's start,
formed by the previous arm and the new arm.  Its descriptor

    (interior angle, |log arm-length ratio|, summed bulge of the two
     incident segments)

is invariant to translation, rotation, uniform scale, mirroring, traversal
direction, command syntax and numeric precision.  Candidates are grouped by
quantized descriptor; each group gets a keyed uniform ``u_b = PRF_K(b)``; the
group is chosen by Gumbel-max on group probability mass, and the token inside
the group by ordinary sampling.

Distribution preservation.  A keyed uniform is used only the *first* time its
descriptor is looked at within one drawing; every later look at the same
descriptor uses a fresh unkeyed uniform.  A uniform that has never been
looked at is independent of the generated prefix, so by induction over steps
the *joint* law of the output equals the model's own (random-PRF
idealization).  Without this rule the one-step identity still holds but the
sequence law does not: reused scores correlate later choices with earlier
ones (see ``reuse="allow"`` and the regression tests).

Detection needs only the final SVG and the key: re-extract vertex
descriptors from the drawn geometry, look up ``u_b`` for each *distinct*
descriptor, and test ``S = sum -log(1 - u_b)`` against Gamma(n, 1), its exact
null distribution for any SVG chosen independently of the key.

The module is model-agnostic.  A model adapter only has to say, for the next
token, which candidates are endpoints and where they land (see
``point_token_models.py`` for IconShop and OmniSVG).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom

from .geometry import Document, GeometryError, apply_affine, closes, dense_points, evidence_status, load_document


@dataclass(frozen=True)
class GeoParameters:
    angle_step: float = 6.0  # degrees per bin of the interior angle
    ratio_step: float = 0.12  # bin width of |log arm ratio|
    bulge_step: float = 0.05  # bin width of summed relative bulge
    straight_tolerance: float = 1.0  # degrees from 180 treated as no vertex
    max_log_ratio: float = 2.5  # skip vertices whose arms differ by > e^2.5
    # Fraction of a bin by which every quantizer is shifted.  Generators that
    # draw on an integer grid produce values exactly on round numbers (90
    # degrees, equal arms, zero bulge); with no shift those sit on bin edges
    # and a rotation's floating-point error flips them.  0.37 keeps 0, 30,
    # 45, 60 and 90 degrees away from every edge.  Runs made before this
    # parameter existed used 0.0.
    bin_offset: float = 0.37
    # "vertex": one keyed decision per segment endpoint (chord corner) and one
    # per curve's first handle.  "polygon": every point of the control polygon
    # (both handles and the endpoint) is a keyed decision, keyed together with
    # its neighbour's descriptor.  Polygon mode has about three decisions per
    # cubic and a far larger key space, but depends on the control polygon and
    # so is not invariant to degree elevation.
    scheme: str = "vertex"
    threshold: float = 1e-6

    def validate(self) -> None:
        if not 0 < self.angle_step <= 45 or not 0 < self.ratio_step <= 1 or not 0 < self.bulge_step <= 1:
            raise GeometryError("invalid geosample quantization")
        if self.scheme not in ("vertex", "polygon"):
            raise GeometryError("scheme must be 'vertex' or 'polygon'")


# ---------------------------------------------------------------------------
# Geometry shared by the sampler and the detector


@dataclass
class Segment2:
    """A drawn segment in model or document coordinates."""

    start: complex
    end: complex
    bulge: float  # max distance of the curve from its chord / chord length (0 for lines)
    straight: bool
    # End derivatives of the segment's polynomial parameterization: degree x
    # (control offset).  Using derivatives rather than raw handles makes a
    # quadratic and its degree-elevated cubic agree.
    out_handle: complex = 0j  # derivative at the start (chord for lines)
    in_handle: complex = 0j  # derivative at the end (chord for lines)
    controls: tuple = ()  # interior control points, for the polygon scheme

    def __post_init__(self) -> None:
        if self.straight or self.out_handle == 0:
            self.out_handle = self.out_handle or (self.end - self.start)
        if self.straight or self.in_handle == 0:
            self.in_handle = self.in_handle or (self.end - self.start)


def bulge_of(points: np.ndarray) -> float:
    """Relative deviation of a dense polyline from its chord."""
    start, end = points[0], points[-1]
    chord = end - start
    length = abs(chord)
    if length <= 1e-12:
        return 0.0
    deviation = np.abs(np.imag(np.conj(chord) * (points - start))) / length
    return float(deviation.max() / length)


def cubic_points(p0: complex, p1: complex, p2: complex, p3: complex, count: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def quad_points(p0: complex, p1: complex, p2: complex, count: int = 32) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _merge_straight(segments: list[Segment2], params: GeoParameters) -> list[Segment2]:
    """Merge collinear consecutive lines so optimizer merging is a no-op."""
    out: list[Segment2] = []
    for segment in segments:
        if abs(segment.end - segment.start) <= 1e-12:
            continue
        if out and out[-1].straight and segment.straight:
            previous = out[-1]
            a, b = previous.end - previous.start, segment.end - segment.start
            if abs(math.degrees(abs(np.angle(b / a)))) <= params.straight_tolerance:
                out[-1] = Segment2(previous.start, segment.end, 0.0, True, previous.out_handle, segment.in_handle)
                continue
        out.append(segment)
    return out


def vertex_descriptor(before: Segment2, after: Segment2, params: GeoParameters) -> tuple | None:
    """Quantized similarity-, reversal- and mirror-invariant corner descriptor."""
    arm_a = before.start - before.end  # from the vertex back along the previous chord
    arm_b = after.end - after.start
    la, lb = abs(arm_a), abs(arm_b)
    if la <= 1e-12 or lb <= 1e-12:
        return None
    angle = math.degrees(abs(np.angle(arm_b / arm_a)))  # interior angle in [0, 180]
    if angle >= 180 - params.straight_tolerance:
        return None
    log_ratio = abs(math.log(la / lb))
    if log_ratio > params.max_log_ratio:
        return None
    bulge = before.bulge + after.bulge
    o = params.bin_offset
    return ("v", math.floor(angle / params.angle_step + o), math.floor(log_ratio / params.ratio_step + o), math.floor(bulge / params.bulge_step + o))


def tangent_descriptor(before: Segment2, after: Segment2, params: GeoParameters) -> tuple | None:
    """Handle geometry at a vertex with at least one incident curve.

    Angle between the arriving handle and the departing handle, and the
    |log| ratio of their lengths when both segments are curves.  Reversal
    swaps and negates the two handles, mirroring flips the angle's sign;
    both leave the descriptor unchanged.
    """
    if before.straight and after.straight:
        return None
    arriving, departing = before.in_handle, after.out_handle
    if abs(arriving) <= 1e-12 or abs(departing) <= 1e-12:
        return None
    angle = math.degrees(abs(np.angle(departing / arriving)))
    o = params.bin_offset
    ratio = -1 if (before.straight or after.straight) else math.floor(abs(math.log(abs(departing) / abs(arriving))) / params.ratio_step + o)
    return ("t", math.floor(angle / params.angle_step + o), ratio)


def contour_descriptors(segments: list[Segment2], closed: bool, params: GeoParameters) -> list[tuple]:
    merged = _merge_straight(segments, params)
    if closed and len(merged) > 1 and merged[0].straight and merged[-1].straight:
        a, b = merged[-1].end - merged[-1].start, merged[0].end - merged[0].start
        if abs(math.degrees(abs(np.angle(b / a)))) <= params.straight_tolerance:
            merged = [Segment2(merged[-1].start, merged[0].end, 0.0, True, merged[-1].out_handle, merged[0].in_handle)] + merged[1:-1]
    pairs = list(zip(merged[:-1], merged[1:]))
    if closed and len(merged) > 2:
        pairs.append((merged[-1], merged[0]))
    out = []
    for before, after in pairs:
        for descriptor in (vertex_descriptor(before, after, params), tangent_descriptor(before, after, params)):
            if descriptor is not None:
                out.append(descriptor)
    return out


def knot_descriptor(before: complex, knot: complex, after: complex, on_curve: bool, params: GeoParameters) -> tuple | None:
    """Descriptor of one control-polygon point from its two neighbours.

    Interior angle at the point and |log| ratio of the two adjacent polygon
    edges.  Reversing the traversal swaps the neighbours, which changes
    neither; mirroring and similarity transforms do not either.
    """
    a, b = before - knot, after - knot
    la, lb = abs(a), abs(b)
    if la <= 1e-12 or lb <= 1e-12:
        return None
    angle = math.degrees(abs(np.angle(b / a)))
    if angle >= 180 - params.straight_tolerance:
        return None
    log_ratio = abs(math.log(la / lb))
    if log_ratio > params.max_log_ratio:
        return None
    o = params.bin_offset
    return ("k", int(on_curve), math.floor(angle / params.angle_step + o), math.floor(log_ratio / params.ratio_step + o))


def pair_key(previous: tuple | None, current: tuple) -> tuple:
    """Key for a descriptor in the context of its neighbour (unordered)."""
    if previous is None:
        return ("s", current)
    first, second = sorted((previous, current), key=str)
    return ("p", first, second)


def polygon_points(segments: list[Segment2]) -> tuple[list[complex], list[bool]]:
    """Control polygon of merged segments: points and on-curve flags."""
    if not segments:
        return [], []
    points, flags = [segments[0].start], [True]
    for segment in segments:
        for control in segment.controls:
            points.append(control)
            flags.append(False)
        points.append(segment.end)
        flags.append(True)
    return points, flags


def polygon_keys(segments: list[Segment2], params: GeoParameters) -> list[tuple]:
    """Context-paired knot keys of one contour (open traversal)."""
    points, flags = polygon_points(_merge_straight(segments, params))
    knots: list[tuple | None] = [None] * len(points)
    for i in range(1, len(points) - 1):
        knots[i] = knot_descriptor(points[i - 1], points[i], points[i + 1], flags[i], params)
    return [pair_key(knots[i - 1], knots[i]) for i in range(1, len(points) - 1) if knots[i] is not None]


def keyed_uniform(key: bytes, descriptor: tuple[int, ...]) -> float:
    digest = hmac.digest(key, ("geosample-v2|" + ",".join(map(str, descriptor))).encode(), "sha256")
    integer = int.from_bytes(digest[:8], "big") >> 11
    return (integer + 0.5) / float(1 << 53)


# ---------------------------------------------------------------------------
# Sampling


@dataclass
class PathTracker:
    """Model-agnostic state of the drawing being generated.

    Adapters call ``move_to``, ``line_to``, ``quad_to`` and ``cubic_to`` as
    tokens are committed; the watermark asks it what vertex a candidate
    endpoint would create.
    """

    params: GeoParameters = field(default_factory=GeoParameters)
    segments: list[Segment2] = field(default_factory=list)
    current: complex | None = None

    def move_to(self, point: complex) -> None:
        self.segments = []
        self.current = point

    def line_to(self, point: complex) -> None:
        if self.current is not None:
            self.segments.append(Segment2(self.current, point, 0.0, True))
        self.current = point

    def cubic_to(self, c1: complex, c2: complex, point: complex) -> None:
        if self.current is not None:
            self.segments.append(Segment2(self.current, point, bulge_of(cubic_points(self.current, c1, c2, point)), False, 3 * (c1 - self.current), 3 * (point - c2), (c1, c2)))
        self.current = point

    def quad_to(self, c: complex, point: complex) -> None:
        if self.current is not None:
            self.segments.append(Segment2(self.current, point, bulge_of(quad_points(self.current, c, point)), False, 2 * (c - self.current), 2 * (point - c), (c,)))
        self.current = point

    def candidate_descriptor(self, end: complex | None, controls: Sequence[complex] = (), before: list[Segment2] | None = None) -> tuple[int, int, int] | None:
        """Descriptor of the vertex at the current point if the next segment ends at ``end``.

        ``before`` may pass a precomputed ``merged_history()`` to avoid
        recomputing it for every candidate.
        """
        if end is None or self.current is None or not self.segments:
            return None
        if len(controls) == 2:
            after = Segment2(self.current, end, bulge_of(cubic_points(self.current, controls[0], controls[1], end)), False, 3 * (controls[0] - self.current), 3 * (end - controls[1]))
        elif len(controls) == 1:
            after = Segment2(self.current, end, bulge_of(quad_points(self.current, controls[0], end)), False, 2 * (controls[0] - self.current), 2 * (end - controls[0]))
        else:
            after = Segment2(self.current, end, 0.0, True)
        if before is None:
            before = self.merged_history()
        if not before:
            return None
        if after.straight and before[-1].straight:
            a, b = before[-1].end - before[-1].start, after.end - after.start
            if abs(b) > 1e-12 and abs(math.degrees(abs(np.angle(b / a)))) <= self.params.straight_tolerance:
                return None  # continues a straight run: no vertex here
        return vertex_descriptor(before[-1], after, self.params)

    def merged_history(self) -> list[Segment2]:
        return _merge_straight(self.segments, self.params)

    def candidate_knot_key(self, point: complex | None, pending: Sequence[complex], line: bool, before: list[Segment2] | None = None) -> tuple | None:
        """Polygon scheme: key fixed by choosing the next control-polygon point.

        The new point determines the descriptor of the *previous* polygon
        point (the last committed point or the last pending control), which
        is keyed together with the descriptor before it.
        """
        if point is None or self.current is None:
            return None
        before = self.merged_history() if before is None else before
        points, flags = polygon_points(before) if before else ([self.current], [True])
        points = points + list(pending)
        flags = flags + [False] * len(pending)
        if len(points) < 2:
            return None
        if line and not pending and before and before[-1].straight:
            a, b = before[-1].end - before[-1].start, point - points[-1]
            if abs(b) > 1e-12 and abs(math.degrees(abs(np.angle(b / a)))) <= self.params.straight_tolerance:
                return None  # continues a straight run; the run is merged
        knot = knot_descriptor(points[-2], points[-1], point, flags[-1], self.params)
        if knot is None:
            return None
        previous = knot_descriptor(points[-3], points[-2], points[-1], flags[-2], self.params) if len(points) >= 3 else None
        return pair_key(previous, knot)

    def candidate_tangent(self, handle: complex | None, before: list[Segment2] | None = None) -> tuple | None:
        """Descriptor fixed by choosing the first control point of a curve."""
        if handle is None or self.current is None or not self.segments:
            return None
        before = self.merged_history() if before is None else before
        if not before:
            return None
        departing = 3 * (handle - self.current)  # cubic start derivative
        if abs(departing) <= 1e-12:
            return None
        probe = Segment2(self.current, self.current + departing, 1.0, False, departing, departing)
        return tangent_descriptor(before[-1], probe, self.params)


@dataclass
class Choice:
    index: int  # position in the candidate list
    descriptor: tuple | None
    groups: int  # distinct descriptor groups among candidates
    keyed: bool


class GeoWatermark:
    """Keyed, distribution-preserving choice among candidate endpoints.

    Use one instance per generated drawing (or call ``reset``): the set of
    descriptors already looked at is part of the sampler's state.
    ``reuse="allow"`` restores the naive behaviour (keyed score reused at
    every step); it has more detection power but does not preserve the joint
    output distribution and exists for ablation only.
    """

    def __init__(self, key: bytes, params: GeoParameters | None = None, seed: int | None = None, reuse: str = "mask",
                 mode: str = "gumbel", delta: float = 2.0, gamma: float = 0.5):
        if len(key) < 16:
            raise GeometryError("key must contain at least 16 bytes")
        if reuse not in ("mask", "allow"):
            raise GeometryError("reuse must be 'mask' or 'allow'")
        if mode not in ("gumbel", "bias"):
            raise GeometryError("mode must be 'gumbel' or 'bias'")
        if not 0 < gamma < 1 or delta < 0:
            raise GeometryError("gamma must be in (0, 1) and delta >= 0")
        self.key = key
        self.params = params or GeoParameters()
        self.params.validate()
        self.rng = random.Random(seed)
        self.reuse = reuse
        # ``bias`` is a green-list sampler (after Kirchenbauer et al.): groups
        # with keyed uniform above 1 - gamma get +delta on their log mass.  It
        # deliberately changes the distribution (see ``last_kl``) in exchange
        # for evidence at low-entropy steps; it is an explicit trade-off, not
        # a distribution-preserving mode.
        self.mode, self.delta, self.gamma = mode, delta, gamma
        self.queried: set[tuple] = set()
        self.last_kl = 0.0  # KL(biased || model) of the most recent bias step, in nats
        self.last_entropy = 0.0  # entropy of the group masses at the most recent step, in nats

    def reset(self) -> None:
        """Start a new drawing: forget which descriptors were looked at."""
        self.queried.clear()

    def choose(self, tracker: PathTracker, ends: Sequence[complex | None], probabilities: Sequence[float], controls: Sequence[Sequence[complex]] | None = None) -> Choice:
        """Pick one candidate segment endpoint.

        ``ends[i]`` is where candidate token i would place the segment's end,
        ``probabilities`` its (already truncated, temperature-scaled) model
        probability.  Candidates that create no vertex (including ``None``
        ends, i.e. non-point tokens) compete with fresh unkeyed uniforms, so
        the overall distribution is still unchanged.
        """
        before = tracker.merged_history()
        descriptors = [tracker.candidate_descriptor(end, controls[i] if controls else (), before) for i, end in enumerate(ends)]
        return self._gumbel(descriptors, probabilities)

    def choose_handle(self, tracker: PathTracker, handles: Sequence[complex | None], probabilities: Sequence[float]) -> Choice:
        """Pick the first control point of a curve, keyed by the vertex tangent."""
        before = tracker.merged_history()
        descriptors = [tracker.candidate_tangent(handle, before) for handle in handles]
        return self._gumbel(descriptors, probabilities)

    def choose_point(self, tracker: PathTracker, pending: Sequence[complex], points: Sequence[complex | None], probabilities: Sequence[float], line: bool) -> Choice:
        """Polygon scheme: pick the next control-polygon point (handle or endpoint)."""
        before = tracker.merged_history()
        descriptors = [tracker.candidate_knot_key(point, pending, line, before) for point in points]
        return self._gumbel(descriptors, probabilities)

    def _gumbel(self, descriptors: list[tuple | None], probabilities: Sequence[float]) -> Choice:
        """Gumbel-max over descriptor groups (keyed) and free candidates (unkeyed)."""
        probabilities = np.asarray(probabilities, dtype=float)
        total = probabilities.sum()
        if total <= 0:
            raise GeometryError("candidate probabilities must have positive mass")
        probabilities = probabilities / total
        groups: dict[object, list[int]] = {}
        for index, descriptor in enumerate(descriptors):
            groups.setdefault(descriptor if descriptor is not None else ("free", index), []).append(index)
        masses = np.array([float(probabilities[members].sum()) for members in groups.values()])
        positive = masses[masses > 0]
        self.last_entropy = float(-(positive * np.log(positive)).sum())
        if self.mode == "bias":
            return self._bias(descriptors, probabilities, groups, masses)
        best_score, best_group = -math.inf, None
        keyed_groups = 0
        for label, members in groups.items():
            mass = float(probabilities[members].sum())
            if mass <= 0:
                continue
            if label[0] == "free":  # type: ignore[index]
                u = self.rng.random()
            elif self.reuse == "mask" and label in self.queried:
                # Its keyed score already influenced the prefix; a fresh
                # uniform keeps this step independent of the history.
                u = self.rng.random()
            else:
                u = keyed_uniform(self.key, label)  # type: ignore[arg-type]
                self.queried.add(label)  # type: ignore[arg-type]
                keyed_groups += 1
            score = math.log(mass) - math.log(-math.log(max(u, 1e-300)))
            if score > best_score:
                best_score, best_group = score, members
        assert best_group is not None
        weights = probabilities[best_group]
        index = best_group[self.rng.choices(range(len(best_group)), weights=weights)[0]] if len(best_group) > 1 else best_group[0]
        return Choice(index, descriptors[index], keyed_groups, descriptors[index] is not None)


    def _bias(self, descriptors: list[tuple | None], probabilities: np.ndarray, groups: dict, masses: np.ndarray) -> Choice:
        """Green-list choice: boost keyed-green groups by ``delta`` nats."""
        labels = list(groups)
        green = np.array([label[0] != "free" and keyed_uniform(self.key, label) > 1 - self.gamma for label in labels])
        tilted = masses * np.exp(self.delta * green)
        tilted = tilted / tilted.sum()
        mask = (tilted > 0) & (masses > 0)
        self.last_kl = float((tilted[mask] * np.log(tilted[mask] / masses[mask])).sum())
        chosen = self.rng.choices(range(len(labels)), weights=tilted)[0]
        members = groups[labels[chosen]]
        index = members[self.rng.choices(range(len(members)), weights=probabilities[members])[0]] if len(members) > 1 else members[0]
        return Choice(index, descriptors[index], int(sum(label[0] != "free" for label in labels)), descriptors[index] is not None)


# ---------------------------------------------------------------------------
# Blind detection from the final SVG


def document_descriptors(source: bytes | Document, params: GeoParameters) -> list[list[tuple]]:
    """Vertex descriptors per scorable contour, from drawn geometry only."""
    document = source if isinstance(source, Document) else load_document(source)
    out = []
    for contour in document.contours:
        if not contour.scorable:
            continue
        segments: list[Segment2] = []
        for segment in contour.subpath.segments:
            points = apply_affine(contour.ctm, segment.points)
            if segment.kind == "L":
                segments.append(Segment2(complex(points[0]), complex(points[-1]), 0.0, True))
            elif segment.kind == "C":
                # Same 32-sample bulge estimate as the sampler, so sampler and
                # detector cannot disagree near a bin boundary.
                curve = cubic_points(*[complex(p) for p in points])
                segments.append(Segment2(complex(points[0]), complex(points[3]), bulge_of(curve), False, 3 * complex(points[1] - points[0]), 3 * complex(points[3] - points[2]), (complex(points[1]), complex(points[2]))))
            elif segment.kind == "Q":
                curve = quad_points(*[complex(p) for p in points])
                segments.append(Segment2(complex(points[0]), complex(points[2]), bulge_of(curve), False, 2 * complex(points[1] - points[0]), 2 * complex(points[2] - points[1]), (complex(points[1]),)))
            else:  # arcs: derivatives from dense samples over the unit parameter
                dense = apply_affine(contour.ctm, dense_points(type(contour.subpath)([segment], False)))
                scale = len(dense) - 1
                out_d, in_d = complex(dense[1] - dense[0]) * scale, complex(dense[-1] - dense[-2]) * scale
                # Arcs have no control polygon; use the cubic-equivalent handles.
                segments.append(Segment2(complex(dense[0]), complex(dense[-1]), bulge_of(dense), False, out_d, in_d, (complex(dense[0]) + out_d / 3, complex(dense[-1]) - in_d / 3)))
        if params.scheme == "polygon":
            out.append(polygon_keys(segments, params))
        else:
            out.append(contour_descriptors(segments, contour.closed or closes(contour.subpath), params))
    return out


def detect(source: bytes, key: bytes, params: GeoParameters | None = None, visibility: str = "strict",
           statistic: str = "gamma", gamma: float = 0.5) -> dict:
    """Blind detection; see ``spectral.detect`` for the ``visibility`` modes.

    ``statistic="gamma"`` (default) sums -log(1 - u) over distinct descriptors
    and suits the Gumbel sampler.  ``statistic="green"`` counts descriptors
    with u > 1 - gamma (exact binomial null) and suits the bias sampler.  Choose
    one before looking at the document; reporting the better of the two needs
    a multiple-testing correction.
    """
    if statistic not in ("gamma", "green"):
        raise GeometryError("statistic must be 'gamma' or 'green'")
    params = params or GeoParameters()
    params.validate()
    if visibility == "render":
        from .visibility import resolve_by_rendering

        document = resolve_by_rendering(source)
    else:
        document = load_document(source)
    per_contour = document_descriptors(document, params)
    distinct = sorted({d for contour in per_contour for d in contour}, key=str)
    if not distinct:
        return {"p_value": 1.0, "log10_p_value": 0.0, "distinct_vertices": 0, **evidence_status(1.0, params.threshold, document)}
    uniforms = {d: keyed_uniform(key, d) for d in distinct}
    if statistic == "green":
        # Distinct descriptors are green independently with probability gamma.
        scores = {d: float(u > 1 - gamma) for d, u in uniforms.items()}
        tail = lambda n, total: float(binom.sf(total - 1, n, gamma))  # noqa: E731
    else:
        # Distinct descriptors have independent uniform PRF values under H0,
        # so the sum of -log(1 - u) is exactly Gamma(n, 1).
        scores = {d: -math.log(1 - u) for d, u in uniforms.items()}
        tail = lambda n, total: float(gammaincc(n, total))  # noqa: E731
    statistic_value = sum(scores.values())
    global_p = tail(len(distinct), statistic_value)
    contour_ps = []
    for contour in per_contour:
        unique = set(contour)
        if unique:
            contour_ps.append(tail(len(unique), sum(scores[d] for d in unique)))
    minimum = min(1.0, len(contour_ps) * min(contour_ps)) if contour_ps else 1.0
    p_value = min(1.0, 2 * min(global_p, minimum))
    return {
        **evidence_status(p_value, params.threshold, document),
        "p_value": p_value,
        "log10_p_value": math.log10(max(p_value, 1e-300)),
        "global_log10_p": math.log10(max(global_p, 1e-300)),
        "distinct_vertices": len(distinct),
        "vertices": sum(len(c) for c in per_contour),
        "statistic": statistic_value,
        "statistic_kind": statistic,
        "null": "Exact Gamma(n,1) null over the key's PRF values for any SVG fixed independently of the key.",
    }
