import pytest

from contourmark.core import WatermarkError
from experiments.ollama_generate import target_for, validate_proposals


def test_proposal_validation_rejects_off_target_geometry():
    target = target_for("flower", 3, 4)
    start = (128, 128)
    assert target == (128, 48)
    valid = "M128,128 C110,110 110,60 128,48"
    validate_proposals([valid, "M128,128 C140,110 140,60 128,48"], start, target)
    with pytest.raises(WatermarkError, match="must run"):
        validate_proposals([valid, "M128,128 C168,102 208,115 208,128"], start, target)
    with pytest.raises(WatermarkError, match="one cubic"):
        validate_proposals([valid, "M128,128 L128,48"], start, target)
