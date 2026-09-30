"""Provider-neutral keyed sampling for autoregressive geometry tokens.

This module contains no model-framework dependency.  Adapters supply the
candidate token IDs and logits after applying the same grammar, temperature,
and truncation policy as the unwatermarked decoder.  The keyed Gumbel draw then
replaces the decoder's random categorical draw without changing that
categorical distribution in the random-PRF idealization.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Hashable, Sequence

from .core import WatermarkError


@dataclass(frozen=True)
class TokenSelection:
    token_id: int
    probability: float
    candidate_count: int
    context_digest: str


def _uniform(key: bytes, asset_id: str, step: int, context_digest: str, stable_id: Hashable) -> float:
    message = f"contourmark-token-v1|{asset_id}|{step}|{context_digest}|{stable_id!s}".encode()
    integer = int.from_bytes(hmac.digest(key, message, "sha256")[:8], "big") >> 11
    return min(max((integer + 0.5) / (1 << 53), 2**-53), 1 - 2**-53)


def categorical_probabilities(
    logits: Sequence[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
) -> list[float]:
    """Apply temperature and top-k/top-p truncation, then normalize."""
    if not logits:
        raise WatermarkError("categorical sampler requires at least one logit")
    if not math.isfinite(temperature) or temperature <= 0:
        raise WatermarkError("temperature must be positive and finite")
    if top_k is not None and not 1 <= top_k <= len(logits):
        raise WatermarkError("top_k must be between 1 and the number of logits")
    if not 0 < top_p <= 1:
        raise WatermarkError("top_p must be in (0, 1]")
    scaled = [float(value) / temperature for value in logits]
    if not all(math.isfinite(value) for value in scaled):
        raise WatermarkError("logits must be finite")
    order = sorted(range(len(scaled)), key=scaled.__getitem__, reverse=True)
    if top_k is not None:
        order = order[:top_k]
    peak = scaled[order[0]]
    masses = {index: math.exp(scaled[index] - peak) for index in order}
    total = sum(masses.values())
    provisional = {index: mass / total for index, mass in masses.items()}
    if top_p < 1:
        kept: list[int] = []
        cumulative = 0.0
        for index in order:
            kept.append(index)
            cumulative += provisional[index]
            if cumulative >= top_p:
                break
        masses = {index: masses[index] for index in kept}
        total = sum(masses.values())
    probabilities = [0.0] * len(scaled)
    for index, mass in masses.items():
        probabilities[index] = mass / total
    return probabilities


class KeyedCategoricalSampler:
    """Deterministically realize categorical draws using a secret PRF."""

    def __init__(self, key: bytes, asset_id: str):
        if len(key) < 16:
            raise WatermarkError("key must contain at least 16 bytes")
        if not 0 < len(asset_id) <= 256:
            raise WatermarkError("asset ID must contain 1 to 256 characters")
        self.key = key
        self.asset_id = asset_id

    def choose(
        self,
        token_ids: Sequence[int],
        logits: Sequence[float],
        *,
        context: bytes,
        step: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
    ) -> TokenSelection:
        if len(token_ids) != len(logits) or len(set(token_ids)) != len(token_ids):
            raise WatermarkError("token IDs and logits must have equal length and unique IDs")
        if step < 0:
            raise WatermarkError("step must be nonnegative")
        probabilities = categorical_probabilities(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        context_digest = hashlib.sha256(context).hexdigest()
        eligible = [index for index, probability in enumerate(probabilities) if probability > 0]
        scores = []
        for index in eligible:
            uniform = _uniform(self.key, self.asset_id, step, context_digest, token_ids[index])
            scores.append(math.log(probabilities[index]) - math.log(-math.log(uniform)))
        winner = eligible[max(range(len(scores)), key=scores.__getitem__)]
        return TokenSelection(
            token_id=int(token_ids[winner]),
            probability=probabilities[winner],
            candidate_count=len(eligible),
            context_digest=context_digest,
        )


@dataclass(frozen=True)
class OmniSVGTokenPolicy:
    """Published OmniSVG 1.1 token ranges for a conservative adapter.

    Only coordinate tokens are marked.  Command and color tokens are excluded
    because changing them can alter grammar or paint semantics.  An integration
    must additionally check that the decoder is currently expecting a
    coordinate and that its complete truncated candidate set lies in this
    range; token-range membership alone is not a grammar proof.
    """

    model_size: str = "4B"

    def __post_init__(self) -> None:
        if self.model_size not in {"4B", "8B"}:
            raise WatermarkError("OmniSVG model_size must be 4B or 8B")

    @property
    def coordinate_range(self) -> range:
        # Derived from OmniSVG config.yaml and SVGTokenizer._load_config.
        start = 151_944 if self.model_size == "4B" else 152_072
        end_exclusive = 191_947 if self.model_size == "4B" else 192_076
        return range(start, end_exclusive)

    def is_coordinate(self, token_id: int) -> bool:
        return token_id in self.coordinate_range

    def eligible_step(self, truncated_token_ids: Sequence[int]) -> bool:
        return len(truncated_token_ids) >= 2 and all(self.is_coordinate(int(token)) for token in truncated_token_ids)
