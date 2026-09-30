import math

import pytest

from contourmark.core import WatermarkError
from contourmark.token_sampling import KeyedCategoricalSampler, OmniSVGTokenPolicy, categorical_probabilities


def test_probability_truncation():
    probabilities = categorical_probabilities([0.0, 1.0, 2.0, 3.0], top_k=3, top_p=0.8)
    assert probabilities[0] == 0
    assert probabilities[1] == 0
    assert sum(probabilities) == pytest.approx(1)
    assert probabilities[3] > probabilities[2]


def test_keyed_sampler_is_deterministic_for_context():
    sampler = KeyedCategoricalSampler(bytes(range(32)), "asset")
    first = sampler.choose([10, 11, 12], [-1.0, 0.0, 1.0], context=b"prefix", step=7)
    second = sampler.choose([10, 11, 12], [-1.0, 0.0, 1.0], context=b"prefix", step=7)
    assert first == second
    assert first.candidate_count == 3
    assert first.probability in categorical_probabilities([-1.0, 0.0, 1.0])


def test_gumbel_draw_preserves_categorical_distribution_over_keys():
    selected = {0: 0, 1: 0, 2: 0}
    logits = [math.log(0.1), math.log(0.3), math.log(0.6)]
    for value in range(4000):
        sampler = KeyedCategoricalSampler(value.to_bytes(32, "big"), "asset")
        selected[sampler.choose([0, 1, 2], logits, context=b"same", step=0).token_id] += 1
    assert 330 < selected[0] < 470
    assert 1080 < selected[1] < 1320
    assert 2280 < selected[2] < 2520


def test_omnisvg_ranges_and_conservative_gate():
    policy = OmniSVGTokenPolicy("4B")
    assert policy.is_coordinate(151_944)
    assert policy.is_coordinate(191_946)
    assert not policy.is_coordinate(151_943)
    assert not policy.is_coordinate(191_947)
    assert policy.eligible_step([151_944, 160_000])
    assert not policy.eligible_step([151_944, 151_943])
    assert not policy.eligible_step([151_944])
    with pytest.raises(WatermarkError):
        OmniSVGTokenPolicy("3B")
