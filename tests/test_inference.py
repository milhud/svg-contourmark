import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from svgpathtools import parse_path

from contourmark.core import WatermarkError
from contourmark.inference import Candidate, GenerationSession, _acceptance_radius, _path_points, _shape_distance, _winner, verify_generation


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


def test_path_translation_preserves_shape_evidence(generation):
    svg, manifest = generation
    root = ET.fromstring(svg)
    for index, path in enumerate(root):
        shift = 1 if index % 2 else -1
        path.set("d", parse_path(path.get("d")).translated(shift + 0j).d())
    assert verify_generation(ET.tostring(root), manifest, KEY)["detected"]


def test_different_choice_does_not_verify(generation):
    svg, manifest = generation
    result = verify_generation(svg, manifest, KEY)
    index = result["details"][0]["expected_index"]
    alternate = manifest["steps"][0]["candidates"][1 - index]["d"]
    chosen = manifest["steps"][0]["candidates"][index]["d"]
    changed = svg.replace(chosen.encode(), alternate.encode(), 1)
    assert not verify_generation(changed, manifest, KEY)["detected"]


def test_partial_path_removal():
    session = GenerationSession(KEY, "0 0 200 200", "redundant-asset")
    for i in range(32):
        y = 5 + 6 * i
        session.add_step([
            Candidate(f"M 10 {y} C 60 {y-9} 140 {y+9} 190 {y}"),
            Candidate(f"M 10 {y} C 60 {y-7} 140 {y+11} 190 {y}"),
        ], {"fill": "none", "stroke": "#333"})
    svg, manifest = session.finish()
    root = ET.fromstring(svg)
    root[:] = list(root)[8:]
    result = verify_generation(ET.tostring(root), manifest, KEY)
    assert result["detected"], result
    assert result["matched_steps"] == 24
    assert result["recognized_steps"] == 24
    assert result["conditional_p_value"] == pytest.approx(2 ** -24)


def test_hidden_contours_do_not_count(generation):
    svg, manifest = generation
    root = ET.fromstring(svg)
    list(root)[0].set("style", "display:none")
    result = verify_generation(ET.tostring(root), manifest, KEY)
    assert result["detected"]
    assert result["recognized_steps"] == 8
    assert result["conditional_p_value"] == pytest.approx(2 ** -8)


def test_too_few_recognized_choices_do_not_detect(generation):
    svg, manifest = generation
    root = ET.fromstring(svg)
    root[:] = list(root)[:6]
    result = verify_generation(ET.tostring(root), manifest, KEY)
    assert not result["detected"]
    assert result["recognized_steps"] == 6
    assert result["conditional_p_value"] == pytest.approx(2 ** -6)


def test_rendering_indirection_cannot_preserve_a_false_positive(generation):
    svg, manifest = generation
    root = ET.fromstring(svg)
    root.set("transform", "translate(1000 1000)")
    with pytest.raises(WatermarkError, match="unsupported rendering attribute"):
        verify_generation(ET.tostring(root), manifest, KEY)

    root = ET.fromstring(svg)
    group = ET.Element("{http://www.w3.org/2000/svg}g", {"opacity": "0"})
    group.extend(list(root))
    root[:] = [group]
    with pytest.raises(WatermarkError, match="no visible contours"):
        verify_generation(ET.tostring(root), manifest, KEY)


def test_gumbel_sampling_preserves_categorical_distribution():
    candidates = [{"fingerprint": "a", "probability": 0.2}, {"fingerprint": "b", "probability": 0.8}]
    count = sum(_winner(i.to_bytes(32, "big"), "asset", 0, candidates) == 0 for i in range(2000))
    assert 350 < count < 450


def test_requires_distinct_geometry():
    session = GenerationSession(KEY, "0 0 200 200")
    with pytest.raises(WatermarkError, match="distinct geometry"):
        session.add_step([Candidate("M0 0L100 0"), Candidate("M0 0 L100 0")])


def test_adaptive_acceptance_regions_remain_disjoint():
    candidates = [
        _path_points("M0 0 C20 -15 80 15 100 0"),
        _path_points("M0 0 C20 -5 80 25 100 0"),
    ]
    scale = 200 * 2**0.5
    separation = _shape_distance(candidates[0], candidates[1], scale)
    radius = _acceptance_radius(candidates, scale, 0.0005)
    assert radius < separation / 2
    assert separation - 2 * radius == pytest.approx(0.1 * separation)


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
