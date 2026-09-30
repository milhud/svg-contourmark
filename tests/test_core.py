import json
import subprocess
from pathlib import Path

import pytest

from contourmark.core import Parameters, WatermarkError, embed, verify


EXAMPLE = (Path(__file__).parents[1] / "examples" / "flower.svg").read_bytes()
KEY = bytes(range(32))


@pytest.fixture(scope="module")
def marked():
    return embed(EXAMPLE, KEY, "test-asset", Parameters(samples=128))


def test_round_trip(marked):
    candidate, manifest = marked
    result = verify(EXAMPLE, candidate, manifest, KEY, null_trials=199)
    assert result["detected"], result
    assert result["normalized_strength"] > 0.35


def test_original_has_no_watermark(marked):
    _, manifest = marked
    assert not verify(EXAMPLE, EXAMPLE, manifest, KEY, null_trials=99)["detected"]


def test_wrong_key_rejected(marked):
    candidate, manifest = marked
    with pytest.raises(WatermarkError, match="authentication"):
        verify(EXAMPLE, candidate, manifest, b"different key that is long enough")


def test_tampered_manifest_rejected(marked):
    candidate, manifest = marked
    tampered = json.loads(json.dumps(manifest))
    tampered["asset_id"] = "other"
    with pytest.raises(WatermarkError, match="authentication"):
        verify(EXAMPLE, candidate, tampered, KEY)


def test_real_scour_minification(marked, tmp_path):
    candidate, manifest = marked
    input_svg = tmp_path / "marked.svg"
    output_svg = tmp_path / "minified.svg"
    input_svg.write_bytes(candidate)
    subprocess.run([
        str(Path(__file__).parents[1] / ".venv" / "bin" / "scour"),
        "-i", str(input_svg), "-o", str(output_svg), "--enable-viewboxing",
        "--enable-id-stripping", "--shorten-ids", "--indent=none",
    ], check=True, capture_output=True)
    result = verify(EXAMPLE, output_svg.read_bytes(), manifest, KEY, null_trials=199)
    assert result["detected"], result


def test_unsupported_svg_errors():
    with pytest.raises(WatermarkError, match="no supported"):
        embed(b'<svg xmlns="http://www.w3.org/2000/svg"><text>hello</text></svg>', KEY)
