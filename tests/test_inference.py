import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from contourmark.core import WatermarkError
from contourmark.inference import Candidate, GenerationSession, _winner, verify_generation


KEY = bytes(range(32))


def _proposals():
    for i in range(9):
        y = 15 + 19 * i
        # Alternative control points change the visible contour, not XML syntax.
        a = f"M 10 {y} C 60 {y-9} 140 {y+9} 190 {y}"
        b = f"M 10 {y} C 60 {y-7} 140 {y+11} 190 {y}"
        yield [Candidate(a), Candidate(b)]


@pytest.fixture(scope="module")
def generation():
    session = GenerationSession(KEY, "0 0 200 200", "sample-asset")
    for choices in _proposals():
        session.add_step(choices, {"fill": "none", "stroke": "#333"})
    return session.finish()


def test_generation_watermark(generation):
    svg, manifest = generation
    result = verify_generation(svg, manifest, KEY)
    assert result["detected"], result
    assert result["conditional_p_value"] == pytest.approx(2 ** -9)


def test_generation_minification(generation, tmp_path):
    svg, manifest = generation
    before = tmp_path / "before.svg"
    after = tmp_path / "after.svg"
    before.write_bytes(svg)
    subprocess.run([
        str(Path(__file__).parents[1] / ".venv" / "bin" / "scour"),
        "-i", str(before), "-o", str(after), "--indent=none",
    ], check=True, capture_output=True)
    assert verify_generation(after.read_bytes(), manifest, KEY)["detected"]


def test_svgo_minification(generation, tmp_path):
    binary = Path(__file__).parents[1] / "node_modules" / ".bin" / "svgo"
    if not binary.exists():
        pytest.skip("SVGO development dependency is not installed")
    svg, manifest = generation
    before = tmp_path / "before.svg"
    after = tmp_path / "after.svg"
    before.write_bytes(svg)
    subprocess.run([str(binary), str(before), "-o", str(after)], check=True, capture_output=True)
    assert verify_generation(after.read_bytes(), manifest, KEY)["detected"]


def test_reordered_and_merged_paths(generation):
    svg, manifest = generation
    root = ET.fromstring(svg)
    paths = list(root)
    root[:] = paths[::-1]
    assert verify_generation(ET.tostring(root), manifest, KEY)["detected"]
    root[:] = paths
    merged_d = " ".join(path.get("d") for path in paths)
    paths[0].set("d", merged_d)
    root[:] = [paths[0]]
    assert verify_generation(ET.tostring(root), manifest, KEY)["detected"]


def test_different_choice_does_not_verify(generation):
    svg, manifest = generation
    result = verify_generation(svg, manifest, KEY)
    index = result["details"][0]["expected_index"]
    alternate = manifest["steps"][0]["candidates"][1 - index]["d"]
    chosen = manifest["steps"][0]["candidates"][index]["d"]
    changed = svg.replace(chosen.encode(), alternate.encode(), 1)
    assert not verify_generation(changed, manifest, KEY)["detected"]


def test_gumbel_sampling_preserves_categorical_distribution():
    candidates = [{"fingerprint": "a", "probability": 0.2}, {"fingerprint": "b", "probability": 0.8}]
    count = sum(_winner(i.to_bytes(32, "big"), "asset", 0, candidates) == 0 for i in range(2000))
    assert 350 < count < 450


def test_requires_distinct_geometry():
    session = GenerationSession(KEY, "0 0 200 200")
    with pytest.raises(WatermarkError, match="distinct geometry"):
        session.add_step([Candidate("M0 0L100 0"), Candidate("M0 0 L100 0")])


def test_cli_end_to_end(tmp_path):
    root = Path(__file__).parents[1]
    key = tmp_path / "owner.key"
    proposals = tmp_path / "proposals.json"
    svg = tmp_path / "drawing.svg"
    manifest = tmp_path / "drawing.wm.json"
    proposals.write_bytes(subprocess.check_output([sys.executable, str(root / "examples" / "make_proposals.py")]))
    cli = [sys.executable, "-m", "contourmark.cli"]
    subprocess.run(cli + ["keygen", str(key)], check=True, capture_output=True)
    subprocess.run(cli + ["sample", str(proposals), str(svg), "--key", str(key), "--manifest", str(manifest)], check=True, capture_output=True)
    result = subprocess.run(cli + ["verify-sample", str(svg), "--key", str(key), "--manifest", str(manifest)], check=True, capture_output=True)
    assert b'"detected": true' in result.stdout
