"""Blind, representation-invariant geometric watermarking of SVG contours.

Each visible contour is resampled uniformly by arc length.  Its normalized
radial function

    rho(s) = |p(s) - centroid| / mean_s |p(s) - centroid|

is invariant to translation, rotation, uniform scaling, and serialization.
Closed contours use Fourier magnitudes of rho (also invariant to start point
and traversal direction); open contours use DCT-II magnitudes (invariant to
direction).  A keyed dither-modulation quantizer (Chen and Wornell's QIM with
a secret dither) moves a band of these magnitudes onto a keyed lattice.

Dithers are seeded by a coarse descriptor of each contour's shape, so distinct
contours use independent pseudorandom lattices and an attacker cannot pool
unrelated contours to estimate one lattice.

Detection needs only the SVG and the key.  For a document chosen independently
of the key, each lattice offset is uniform, so each group score is
``|A| cos(U)`` with ``U`` uniform.  The detector reports a Chernoff upper bound
on the tail of that exact null distribution, conditional on the observed
document.  No threshold is fitted to a corpus.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import dct
from scipy.optimize import brentq
from scipy.special import i0e, i1e

from .geometry import (
    SVG_NS,
    Contour,
    Document,
    GeometryError,
    Segment,
    Subpath,
    apply_affine,
    apply_linear_inverse,
    dense_points,
    load_document,
    parse_path_data,
    arc_to_cubics,
    resample,
    segment_counts,
    serialize_compact,
    split_segment,
    shape_to_d,
    tag,
)


@dataclass(frozen=True)
class SpectralParameters:
    samples: int = 256
    k_min: int = 2
    k_count: int = 8
    delta: float = 0.004  # lattice step in normalized radial-magnitude units
    seed_bin: float = 0.35  # descriptor quantization width (log units)
    min_relative_length: float = 0.02  # of the document geometry diagonal
    max_straightness: float = 0.995  # open contours straighter than this are skipped
    iterations: int = 8
    min_gain: float = 0.4  # orthogonal Jacobian gain for a coefficient to be carried
    embed_margin: float = 0.625  # embedder marks coefficients with gain >= margin * min_gain
    max_densify: int = 4  # rounds of curved-segment splitting allowed while embedding
    rounding_fraction: float = 0.05  # serialization step as a fraction of delta * contour radius
    realize_gain: float = 0.25  # handle-model gain needed before a coefficient is embedded
    threshold: float = 1e-6  # detection p-value threshold

    def validate(self) -> None:
        if not 64 <= self.samples <= 4096:
            raise GeometryError("samples must be in [64, 4096]")
        if not 1 <= self.k_min or not 1 <= self.k_count <= 64 or self.k_min + self.k_count > self.samples // 4:
            raise GeometryError("invalid frequency band")
        if not 0 < self.delta <= 0.1:
            raise GeometryError("delta must be in (0, 0.1]")
        if not 0 < self.threshold < 1:
            raise GeometryError("threshold must be in (0, 1)")

    @property
    def band(self) -> np.ndarray:
        return np.arange(self.k_min, self.k_min + self.k_count)


# ---------------------------------------------------------------------------
# Features


def radial_signal(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean()
    radius = np.abs(points - center)
    mean = float(radius.mean())
    if mean <= 0:
        raise GeometryError("degenerate contour")
    return radius / mean, mean


def coefficients(points: np.ndarray, closed: bool, band: np.ndarray) -> np.ndarray:
    """Band of radial-function Fourier (closed) or DCT-II (open) coefficients.

    Their magnitudes are invariant to similarity transforms, start point and
    traversal direction.  Phases are not, and the watermark ignores them.
    """
    rho, _ = radial_signal(points)
    n = len(rho)
    if closed:
        return 2 * np.fft.rfft(rho)[band] / n
    return (dct(rho, type=2)[band] / n).astype(complex)


def magnitudes(points: np.ndarray, closed: bool, band: np.ndarray) -> np.ndarray:
    return np.abs(coefficients(points, closed, band))


def descriptor(points: np.ndarray, closed: bool, bin_width: float) -> str:
    """Coarse similarity-invariant shape class used to seed the dither."""
    centered = points - points.mean()
    covariance = np.cov(np.vstack([centered.real, centered.imag]))
    eigen = np.linalg.eigvalsh(covariance)
    elongation = 0.5 * math.log(max(eigen[0], 1e-18) / max(eigen[1], 1e-18))
    segment_lengths = np.abs(np.diff(np.append(points, points[0]) if closed else points))
    length = float(segment_lengths.sum())
    if closed:
        x, y = points.real, points.imag
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        compact = math.log(max(4 * math.pi * area / (length * length), 1e-9))
    else:
        compact = math.log(max(abs(points[-1] - points[0]) / length, 1e-9))
    return f"{'c' if closed else 'o'}:{round(compact / bin_width)}:{round(elongation / bin_width)}"


def dithers(key: bytes, seed: str, count: int) -> np.ndarray:
    digest = b""
    block = 0
    while len(digest) < 8 * count:
        digest += hmac.digest(key, f"contourmark-spectral-v1|{seed}|{block}".encode(), "sha256")
        block += 1
    integers = np.frombuffer(digest[: 8 * count], dtype=">u8") >> np.uint64(11)
    return (integers.astype(np.float64) + 0.5) / float(1 << 53)


def _straightness(points: np.ndarray) -> float:
    length = float(np.abs(np.diff(points)).sum())
    return abs(points[-1] - points[0]) / length if length > 0 else 1.0


@dataclass
class Observation:
    index: int
    seed: str
    closed: bool
    magnitudes: np.ndarray
    length: float
    selected: np.ndarray  # boolean mask over the band: coefficients this geometry can carry


def contour_samples(contour: Contour, count: int) -> tuple[np.ndarray, float]:
    return resample(apply_affine(contour.ctm, dense_points(contour.subpath)), count, contour.closed)


def observe(document: Document, params: SpectralParameters) -> list[Observation]:
    scale = document.scale()
    observations = []
    for index, contour in enumerate(document.contours):
        if not contour.visible:
            continue
        samples, length = contour_samples(contour, params.samples)
        if scale <= 0 or length < params.min_relative_length * scale:
            continue
        if not contour.closed and _straightness(samples) > params.max_straightness:
            continue
        try:
            model = _GeometricModel(samples, contour.closed, params)
            selected = model.selection()
        except (GeometryError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if not selected.any():
            continue
        values = np.abs(coefficients(samples, contour.closed, params.band))
        observations.append(Observation(index, descriptor(samples, contour.closed, params.seed_bin), contour.closed, values, length, selected))
    return observations


# ---------------------------------------------------------------------------
# Null distribution


def _log_i0(x: np.ndarray) -> np.ndarray:
    return np.log(i0e(x)) + np.abs(x)


def _chernoff(statistic: float, weights: np.ndarray) -> tuple[float, float]:
    """(log Chernoff bound, optimizing theta) for P[sum w_i cos(2 pi U_i) >= t]."""
    total = float(weights.sum())
    statistic = min(statistic, total * (1 - 1e-9))

    def slope(theta: float) -> float:
        x = theta * weights
        return float(np.sum(weights * i1e(x) / i0e(x))) - statistic

    upper = 1.0
    while slope(upper) < 0 and upper < 1e12:
        upper *= 2
    theta = brentq(slope, 0.0, upper, xtol=1e-12, rtol=1e-12)
    exponent = theta * statistic - float(np.sum(_log_i0(theta * weights)))
    return -max(exponent, 0.0), theta


def _tilted_tail(statistic: float, weights: np.ndarray, theta: float) -> float:
    """Log of an exact upper bound via exponentially tilted convolution.

    Each term w cos(2 pi U) is rounded *up* to a grid of step h, so the
    discretized sum dominates the true sum and its tail is a valid bound.
    Tilting by exp(theta x) centers the convolution at the statistic, where
    floating-point FFT arithmetic is accurate even for tiny tail masses.
    """
    n = len(weights)
    step = min(0.01, 0.07 / math.sqrt(n))
    values, multiplicity = np.unique(np.round(weights, 12), return_counts=True)
    log_normalizer = 0.0
    pmfs = []
    offset = 0
    length = 1
    for weight, count in zip(values, multiplicity):
        top = int(math.ceil(weight / step))
        bottom = int(math.ceil(-weight / step))
        grid = np.arange(bottom, top + 1) * step
        # P[w cos(2 pi U) <= x] = 1 - arccos(x / w) / pi on [-w, w].
        cdf = 1 - np.arccos(np.clip(grid / weight, -1, 1)) / np.pi
        mass = np.diff(np.concatenate([[0.0], cdf]))
        mass = np.clip(mass, 0, None)
        log_tilted = np.log(np.maximum(mass, 1e-300)) + theta * grid
        peak = float(log_tilted.max())
        tilted = np.exp(log_tilted - peak) * (mass > 0)
        norm = float(tilted.sum())
        log_normalizer += count * (peak + math.log(norm))
        pmfs.append((tilted / norm, int(count)))
        offset += count * bottom
        length += count * (top - bottom)
    size = 1 << int(math.ceil(math.log2(length + 1)))
    spectrum = np.ones(size // 2 + 1, dtype=complex)
    for pmf, count in pmfs:
        spectrum *= np.fft.rfft(pmf, size) ** count
    distribution = np.fft.irfft(spectrum, size)[:length]
    grid = (offset + np.arange(length)) * step
    keep = (grid >= statistic - 1e-12) & (distribution > 0)
    if not keep.any():
        return 0.0
    terms = np.log(distribution[keep]) - theta * grid[keep]
    peak = float(terms.max())
    return min(0.0, log_normalizer + peak + math.log(float(np.exp(terms - peak).sum())))


def chernoff_log_p(statistic: float, weights: np.ndarray, exact: bool = True) -> float:
    """Natural log of a valid upper bound on P[sum w_i cos(2 pi U_i) >= t].

    Returns the smaller of the Chernoff bound and, when affordable, the
    tilted discretized-convolution bound; both are valid, so their minimum is.
    """
    weights = weights[weights > 0]
    total = float(weights.sum())
    if statistic <= 0 or total <= 0:
        return 0.0
    bound, theta = _chernoff(statistic, weights)
    if exact and len(weights) <= 4096 and total / min(0.01, 0.07 / math.sqrt(len(weights))) < 2e6:
        try:
            bound = min(bound, _tilted_tail(statistic, weights, theta))
        except (ValueError, FloatingPointError, OverflowError):
            pass
    return bound


def score(observations: list[Observation], key: bytes, params: SpectralParameters) -> dict:
    """Keyed statistic and a valid p-value bound for one document.

    Contours sharing a seed share dithers, so they are pooled into one group
    whose term for coefficient k is ``Re(A_k exp(-2 pi i d_k))`` with
    ``A_k = sum_j exp(2 pi i m_jk / delta)``.  Distinct seeds have independent
    dithers.  Under H0 each term is ``|A_k| cos(uniform angle)``.  Two tests
    are combined with a Bonferroni factor of two: the sum over all groups
    (dense marks) and the minimum group p-value (sparse marks, e.g. a marked
    icon composed with unmarked artwork).
    """
    groups: dict[str, np.ndarray] = {}
    for observation in observations:
        phasors = np.exp(2j * np.pi * observation.magnitudes / params.delta) * observation.selected
        groups[observation.seed] = groups.get(observation.seed, 0) + phasors
    statistic = 0.0
    weights = []
    for seed, sums in groups.items():
        offsets = np.exp(-2j * np.pi * dithers(key, seed, params.k_count))
        group_statistic = float(np.real(sums * offsets).sum())
        statistic += group_statistic
        weights.append(np.abs(sums))
    per_contour = []
    contour_log_p = []
    for observation in observations:
        offsets = np.exp(-2j * np.pi * dithers(key, observation.seed, params.k_count))
        terms = np.real(np.exp(2j * np.pi * observation.magnitudes / params.delta) * offsets)[observation.selected]
        log_p = chernoff_log_p(float(terms.sum()), np.ones(len(terms)))
        contour_log_p.append(log_p)
        per_contour.append({"contour": observation.index, "seed": observation.seed, "terms": len(terms), "mean_alignment": float(terms.mean()), "log10_p": log_p / math.log(10)})
    weight_array = np.concatenate(weights) if weights else np.zeros(0)
    global_log_p = chernoff_log_p(statistic, weight_array)
    # Union bound over contours: valid even when contours share a seed.
    minimum_log_p = min(0.0, math.log(len(contour_log_p)) + min(contour_log_p)) if contour_log_p else 0.0
    combined = min(0.0, math.log(2) + min(global_log_p, minimum_log_p))
    return {
        "statistic": statistic,
        "max_statistic": float(weight_array.sum()),
        "log10_p_value": combined / math.log(10),
        "p_value": math.exp(combined),
        "log10_p_global": global_log_p / math.log(10),
        "log10_p_min_contour": minimum_log_p / math.log(10),
        "groups": len(groups),
        "terms": int(np.count_nonzero(weight_array)),
        "per_contour": per_contour,
    }


def detect(source: bytes, key: bytes, params: SpectralParameters | None = None) -> dict:
    params = params or SpectralParameters()
    params.validate()
    if len(key) < 16:
        raise GeometryError("key must contain at least 16 bytes")
    document = load_document(source)
    observations = observe(document, params)
    result = score(observations, key, params)
    result.update({
        "detected": result["p_value"] <= params.threshold,
        "threshold": params.threshold,
        "usable_contours": len(observations),
        "distinct_seeds": len({o.seed for o in observations}),
        "null": "For an SVG fixed independently of the key, the p-value bound holds exactly over the key's pseudorandom dithers; no corpus-fitted threshold is used.",
    })
    return result


# ---------------------------------------------------------------------------
# Deformation models


def _basis(positions: np.ndarray, closed: bool, params: SpectralParameters) -> np.ndarray:
    """Low-frequency scalar fields along the contour (columns = parameters)."""
    top = params.k_min + params.k_count + 2
    if closed:
        columns = [np.cos(2 * np.pi * m * positions) for m in range(1, top)] + [np.sin(2 * np.pi * m * positions) for m in range(1, top)]
    else:
        columns = [np.cos(np.pi * m * positions) for m in range(0, 2 * top)]
    return np.stack(columns, axis=1)


def _normals(samples: np.ndarray, closed: bool) -> np.ndarray:
    tangent = np.roll(samples, -1) - np.roll(samples, 1) if closed else np.gradient(samples)
    return 1j * tangent / np.maximum(np.abs(tangent), 1e-15)


class _Model:
    """Shared Jacobian and coefficient-selection logic."""

    params: SpectralParameters
    parameters: int

    def features(self, theta: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, theta: np.ndarray, current: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        columns = []
        for j in range(len(theta)):
            probe = theta.copy()
            probe[j] += eps
            columns.append((self.features(probe) - current) / eps)
        return np.stack(columns, axis=1)

    def rows(self, current: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """Real Jacobian of each coefficient magnitude.

        Where a magnitude is (nearly) zero, e.g. every band coefficient of a
        circle, the magnitude is not differentiable; use the direction in the
        complex plane with the largest gain instead.
        """
        out = np.empty(jacobian.shape, dtype=float)
        for k, (value, gradient) in enumerate(zip(current, jacobian)):
            if abs(value) > 0.05 * self.params.delta:
                out[k] = np.real(np.conj(value / abs(value)) * gradient)
                continue
            a, b = gradient.real, gradient.imag
            gram = np.array([[a @ a, a @ b], [a @ b, b @ b]])
            _, vectors = np.linalg.eigh(gram)
            c, d = vectors[:, -1]
            out[k] = c * a + d * b
        return out

    def select(self, rows: np.ndarray, gain: float | None = None) -> np.ndarray:
        """Coefficients whose gain, orthogonal to all lower frequencies, is >= gain.

        Sets are nested in the threshold, so an embedder using a lower
        threshold marks a superset of what the verifier later selects.
        """
        gain = self.params.min_gain if gain is None else gain
        mask = np.zeros(rows.shape[0], dtype=bool)
        basis: list[np.ndarray] = []
        for k, row in enumerate(rows):
            residual = row.copy()
            for vector in basis:
                residual -= (residual @ vector) * vector
            norm = float(np.linalg.norm(residual))
            mask[k] = norm >= gain
            if norm > 1e-9:
                basis.append(residual / norm)
        return mask

    def selection(self, gain: float | None = None) -> np.ndarray:
        zero = np.zeros(self.parameters)
        current = self.features(zero)
        return self.select(self.rows(current, self.jacobian(zero, current)), gain)


class _GeometricModel(_Model):
    """Representation-free deformation model used to decide capacity.

    Curved stretches of the contour may move along their normals under a
    low-frequency field; exactly straight stretches stay straight, moving only
    through their endpoints.  The model depends on the drawn geometry, not on
    how many path commands encode it, so the verifier's selection survives
    subdivision, command rewriting, and optimizer output.
    """

    STRAIGHT = 1e-7  # radians of turning between consecutive samples

    def __init__(self, samples: np.ndarray, closed: bool, params: SpectralParameters):
        self.params = params
        self.samples = samples
        self.closed = closed
        count = len(samples)
        _, self.mean_radius = radial_signal(samples)
        positions = np.arange(count) / count if closed else np.linspace(0.0, 1.0, count)
        self.basis = _basis(positions, closed, params)
        self.parameters = self.basis.shape[1]
        self.normals = _normals(samples, closed)
        before = samples - np.roll(samples, 1)
        after = np.roll(samples, -1) - samples
        with np.errstate(divide="ignore", invalid="ignore"):
            turning = np.abs(np.angle(after / before))
        turning = np.where((np.abs(before) > 0) & (np.abs(after) > 0), turning, np.inf)
        if not closed:
            turning[0] = turning[-1] = np.inf
        straight = turning < self.STRAIGHT
        # Each straight sample interpolates linearly between the anchors that
        # bound its run, so lines remain lines under the modeled deformation.
        left = np.arange(count)
        right = np.arange(count)
        weight = np.zeros(count)
        if straight.any() and not straight.all():
            order = np.arange(count)
            if closed:
                shift = int(np.argmin(straight))  # start at an anchor
                order = np.roll(order, -shift)
            last_anchor = order[0]
            run: list[int] = []
            for index in list(order[1:]) + ([order[0]] if closed else []):
                if straight[index]:
                    run.append(index)
                    continue
                for position, member in enumerate(run, start=1):
                    left[member], right[member] = last_anchor, index
                    weight[member] = position / (len(run) + 1)
                run = []
                last_anchor = index
        self.left, self.right, self.weight = left, right, weight

    def features(self, theta: np.ndarray) -> np.ndarray:
        field = self.normals * (self.basis @ theta) * self.mean_radius
        displacement = (1 - self.weight) * field[self.left] + self.weight * field[self.right]
        moved, _ = resample(self.samples + displacement, len(self.samples), self.closed)
        return coefficients(moved, self.closed, self.params.band)


class _HandleModel(_Model):
    """Realizes low-frequency normal fields on a path's native handles."""

    def __init__(self, contour: Contour, params: SpectralParameters):
        self.contour = contour
        self.params = params
        self.segments = contour.subpath.segments
        self.closed = contour.closed
        self.counts = segment_counts(contour.subpath)
        # Unique handles: on-curve points shared between segments, plus
        # interior control points.
        handle_of: dict[tuple[int, int], int] = {}
        positions: list[complex] = []
        for s, segment in enumerate(self.segments):
            for p in range(len(segment.points)):
                if p == 0 and s > 0:
                    handle_of[(s, 0)] = handle_of[(s - 1, len(self.segments[s - 1].points) - 1)]
                    continue
                handle_of[(s, p)] = len(positions)
                positions.append(complex(segment.points[p]))
        last = (len(self.segments) - 1, len(self.segments[-1].points) - 1)
        if self.closed and abs(positions[handle_of[last]] - positions[0]) <= 1e-9 * (1 + abs(positions[0])):
            orphan = handle_of[last]
            handle_of[last] = 0
            if orphan == len(positions) - 1:
                positions.pop()
        self.local = np.array(positions, dtype=complex)
        dense = apply_affine(contour.ctm, dense_points(contour.subpath, self.counts))
        samples, _ = resample(dense, params.samples, self.closed)
        _, self.mean_radius = radial_signal(samples)
        steps = np.abs(np.diff(np.append(dense, dense[0]) if self.closed else dense))
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        total = cumulative[-1]
        s_of = np.zeros(len(positions))
        offset = 0
        for s, (segment, count) in enumerate(zip(self.segments, self.counts)):
            span = cumulative[offset: offset + count]
            for p in range(len(segment.points)):
                fraction = p / (len(segment.points) - 1)
                s_of[handle_of[(s, p)]] = span[min(int(round(fraction * (count - 1))), count - 1)] / total
            offset += count
        normals = _normals(samples, self.closed)
        indices = np.round(s_of * (params.samples if self.closed else params.samples - 1)).astype(int)
        self.normals = normals[np.clip(indices, 0, params.samples - 1) % params.samples]
        self.basis = _basis(s_of, self.closed, params)
        self.parameters = self.basis.shape[1]
        self.segment_handles = [np.array([handle_of[(s, p)] for p in range(len(segment.points))]) for s, segment in enumerate(self.segments)]

    def displaced(self, theta: np.ndarray) -> Subpath:
        displacement = self.normals * (self.basis @ theta) * self.mean_radius
        local = self.local + apply_linear_inverse(self.contour.ctm, displacement)
        segments = []
        for segment, handles in zip(self.segments, self.segment_handles):
            points = local[handles]
            arc = segment.arc
            if arc is not None:
                # Scale radii with the chord so a moved arc keeps its shape
                # instead of hitting SVG's radius-correction discontinuity.
                before = abs(segment.points[-1] - segment.points[0])
                ratio = abs(points[-1] - points[0]) / before if before > 0 else 1.0
                arc = (arc[0] * ratio, arc[1] * ratio, arc[2], arc[3], arc[4])
            segments.append(Segment(segment.kind, points, arc))
        return Subpath(segments, self.contour.subpath.closed)

    def features(self, theta: np.ndarray) -> np.ndarray:
        dense = apply_affine(self.contour.ctm, dense_points(self.displaced(theta), self.counts))
        samples, _ = resample(dense, self.params.samples, self.closed)
        return coefficients(samples, self.closed, self.params.band)


# ---------------------------------------------------------------------------
# Embedding


def _targets(values: np.ndarray, offsets: np.ndarray, delta: float) -> np.ndarray:
    lattice = delta * (np.round(values / delta - offsets) + offsets)
    return np.where(lattice < 0, lattice + delta, lattice)


def _solve(model: _HandleModel, theta: np.ndarray, targets: np.ndarray, mask: np.ndarray, params: SpectralParameters) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-norm Gauss-Newton on the selected coefficient magnitudes."""
    current = model.features(theta)
    error = (targets - np.abs(current)) * mask
    for _ in range(params.iterations):
        if np.max(np.abs(error)) < 0.02 * params.delta:
            break
        rows = model.rows(current, model.jacobian(theta, current))[mask]
        gram = rows @ rows.T
        damping = 1e-9 * (np.trace(gram) / len(gram) + 1e-12)
        step = rows.T @ np.linalg.solve(gram + damping * np.eye(len(gram)), error[mask])
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = theta + scale * step
            values = model.features(candidate)
            candidate_error = (targets - np.abs(values)) * mask
            if np.linalg.norm(candidate_error) < np.linalg.norm(error):
                theta, current, error = candidate, values, candidate_error
                improved = True
                break
        if not improved:
            break
    return theta, error


def _densified(contour: Contour, rounds: int) -> Contour:
    """Split long curved segments so handles can realize the modeled field."""
    subpath = contour.subpath
    for _ in range(rounds):
        lengths = [float(np.abs(np.diff(s.points)).sum()) for s in subpath.segments]
        curved = [length for length, s in zip(lengths, subpath.segments) if s.kind != "L"]
        if not curved:
            break
        limit = float(np.median(curved))
        segments: list[Segment] = []
        for length, segment in zip(lengths, subpath.segments):
            if segment.kind == "A":
                # Arcs moved by their endpoints stay rigid; cubic pieces of at
                # most 45 degrees (error < 5e-6 radius) can bend.
                segments.extend(piece for half in split_segment(segment) for piece in arc_to_cubics(half))
            elif segment.kind != "L" and length >= limit:
                segments.extend(split_segment(segment))
            else:
                segments.append(segment)
        subpath = Subpath(segments, subpath.closed)
    return Contour(contour.element, contour.subpath_index, subpath, contour.ctm, contour.visible, contour.editable)


def _curve_distance(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest reference polyline vertex."""
    return np.min(np.abs(points[:, None] - reference[None, :]), axis=1)


def _embed_contour(contour: Contour, key: bytes, params: SpectralParameters) -> tuple[Subpath, dict]:
    samples, _ = contour_samples(contour, params.samples)
    seed = descriptor(samples, contour.closed, params.seed_bin)
    offsets = dithers(key, seed, params.k_count)
    geometric = _GeometricModel(samples, contour.closed, params)
    wanted = geometric.selection(params.min_gain * params.embed_margin)
    model = _HandleModel(contour, params)
    realizable = model.selection(params.realize_gain)
    rounds = 0
    while (wanted & ~realizable).any() and rounds < params.max_densify:
        rounds += 1
        model = _HandleModel(_densified(contour, rounds), params)
        realizable = model.selection(params.realize_gain)
    mask = wanted & realizable
    theta = np.zeros(model.parameters)
    original = model.features(theta)
    targets = _targets(np.abs(original), offsets, params.delta)
    error = np.zeros(len(original))
    if mask.any():
        theta, error = _solve(model, theta, targets, mask, params)
    subpath = model.displaced(theta)
    marked = Contour(contour.element, contour.subpath_index, subpath, contour.ctm, True, True)
    marked_samples, _ = contour_samples(marked, params.samples)
    verifier_mask = _GeometricModel(marked_samples, contour.closed, params).selection()
    marked_values = np.abs(coefficients(marked_samples, contour.closed, params.band))
    alignment = np.cos(2 * np.pi * (marked_values / params.delta - offsets))
    displacement = np.abs(model.normals * (model.basis @ theta)) * model.mean_radius
    curve_shift = _curve_distance(marked_samples, contour_samples(contour, 4 * params.samples)[0])
    return subpath, {
        "seed": seed,
        "seed_stable": descriptor(marked_samples, contour.closed, params.seed_bin) == seed,
        "embedded_coefficients": int(mask.sum()),
        "unrealized_coefficients": int((wanted & ~realizable).sum()),
        "densify_rounds": rounds,
        "verifier_coefficients": int(verifier_mask.sum()),
        "verifier_unmarked": int((verifier_mask & ~mask).sum()),
        "verifier_alignment": float(alignment[verifier_mask].mean()) if verifier_mask.any() else None,
        "max_handle_displacement": float(displacement.max()) if len(displacement) else 0.0,
        "curve_rms_over_radius": float(np.sqrt(np.mean(curve_shift ** 2)) / model.mean_radius),
        "curve_max_displacement": float(curve_shift.max()),
        "mean_radius": model.mean_radius,
    }


def _decimals(quantum: float) -> int:
    """Fewest decimals whose rounding step is at most ``quantum``."""
    return int(min(8, max(1, math.ceil(-math.log10(max(quantum, 1e-12)) - 1e-9))))


def embed(source: bytes, key: bytes, params: SpectralParameters | None = None, convert_shapes: bool = True) -> tuple[bytes, dict]:
    """Embed a blind geometric watermark; returns (svg, public report).

    The report contains no secret material and is not needed for detection.
    """
    params = params or SpectralParameters()
    params.validate()
    if len(key) < 16:
        raise GeometryError("key must contain at least 16 bytes")
    if b"<!ENTITY" in source.upper() or b"<!DOCTYPE" in source.upper():
        raise GeometryError("DTD/entity declarations are not accepted")
    try:
        root = ET.fromstring(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True)))
    except ET.ParseError as exc:
        raise GeometryError(f"invalid SVG XML: {exc}") from exc
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    if convert_shapes:
        _shapes_to_paths(root)
    document = load_document(root)
    scale = document.scale()
    if scale <= 0:
        raise GeometryError("no drawable geometry")
    eligible = {observation.index for observation in observe(document, params)}
    replaced: dict[ET.Element, dict[int, Subpath]] = {}
    quanta: dict[ET.Element, float] = {}
    report_contours = []
    for index, contour in enumerate(document.contours):
        if index not in eligible or not contour.editable or contour.element is None:
            continue
        try:
            subpath, info = _embed_contour(contour, key, params)
        except (GeometryError, np.linalg.LinAlgError, ValueError):
            continue
        replaced.setdefault(contour.element, {})[contour.subpath_index] = subpath
        # Serialization rounding must stay far below the lattice step.
        local_scale = math.sqrt(abs(float(np.linalg.det(contour.ctm[:, :2])))) or 1.0
        quantum = params.rounding_fraction * params.delta * info["mean_radius"] / local_scale
        quanta[contour.element] = min(quanta.get(contour.element, math.inf), quantum)
        report_contours.append({"contour": index, **info})
    decimals_used = {}
    for element, changes in replaced.items():
        original = parse_path_data(element.get("d", ""))
        decimals = _decimals(quanta[element])
        decimals_used[element.get("id") or str(len(decimals_used))] = decimals
        element.set("d", serialize_compact([changes.get(i, sub) for i, sub in enumerate(original)], decimals))
    output = ET.tostring(root, encoding="unicode").encode()
    if source.lstrip().startswith(b"<?xml"):
        output = b'<?xml version="1.0" encoding="UTF-8"?>\n' + output
    report = {
        "schema": "contourmark-spectral-v1",
        "parameters": asdict(params),
        "marked_contours": len(report_contours),
        "eligible_contours": len(eligible),
        "total_contours": len(document.contours),
        "decimals": decimals_used,
        "contours": report_contours,
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }
    return output, report


def _shapes_to_paths(root: ET.Element) -> None:
    """Rewrite basic shapes as equivalent paths so their outlines can carry a mark."""
    geometric = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}
    shape_attributes = {"x", "y", "width", "height", "rx", "ry", "r", "cx", "cy", "x1", "y1", "x2", "y2", "points"}
    skip = {"defs", "clipPath", "mask", "symbol", "pattern", "marker"}

    def visit(element: ET.Element) -> None:
        for child in element:
            name = tag(child)
            if name in skip:
                continue
            if name in geometric:
                d = shape_to_d(child)
                if d:
                    namespace = child.tag[: child.tag.index("}") + 1] if child.tag.startswith("{") else ""
                    for attribute in shape_attributes & set(child.attrib):
                        del child.attrib[attribute]
                    child.tag = namespace + "path"
                    child.set("d", d)
            visit(child)

    visit(root)
