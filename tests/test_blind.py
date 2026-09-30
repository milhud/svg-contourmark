import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from contourmark.blind import BlindGenerationSession, detect_blind
from contourmark.core import WatermarkError


KEY = bytes(range(32))


@pytest.fixture(scope="module")
def blind_svg():
    session = BlindGenerationSession(KEY, "0 0 200 200", "blind-asset")
    for i in range(32):
        y = 5 + 6 * i
        session.add_step([
            f"M 10 {y} C 60 {y-9} 140 {y+9} 190 {y}",
            f"M 10 {y} C 60 {y-5} 140 {y+13} 190 {y}",
        ], {"fill": "none", "stroke": "#333"})
    return session.finish()


def test_blind_detection(blind_svg):
    result = detect_blind(blind_svg, KEY, "blind-asset")
    assert result["detected"], result
    assert result["usable_contours"] == 32


def test_blind_scour(blind_svg, tmp_path):
    before = tmp_path / "before.svg"
    after = tmp_path / "after.svg"
    before.write_bytes(blind_svg)
    binary = Path(__file__).parents[1] / ".venv" / "bin" / "scour"
    subprocess.run([str(binary), "-i", str(before), "-o", str(after), "--indent=none"], check=True, capture_output=True)
    assert detect_blind(after.read_bytes(), KEY, "blind-asset")["detected"]


def test_blind_svgo(blind_svg, tmp_path):
    binary = Path(__file__).parents[1] / "node_modules" / ".bin" / "svgo"
    if not binary.exists():
        pytest.skip("SVGO development dependency is not installed")
    before = tmp_path / "before.svg"
    after = tmp_path / "after.svg"
    before.write_bytes(blind_svg)
    subprocess.run([str(binary), str(before), "-o", str(after)], check=True, capture_output=True)
    assert detect_blind(after.read_bytes(), KEY, "blind-asset")["detected"]


def test_blind_path_deletion(blind_svg):
    root = ET.fromstring(blind_svg)
    root[:] = list(root)[8:]
    assert detect_blind(ET.tostring(root), KEY, "blind-asset")["detected"]


def test_blind_rejects_unsupported_rendering(blind_svg):
    root = ET.fromstring(blind_svg)
    ET.SubElement(root, "{http://www.w3.org/2000/svg}rect", {"x": "0", "y": "0", "width": "200", "height": "200"})
    with pytest.raises(WatermarkError, match="unsupported rendering element"):
        detect_blind(ET.tostring(root), KEY, "blind-asset")
    root = ET.fromstring(blind_svg)
    list(root)[0].set("style", "stroke-width:0;fill:none")
    assert detect_blind(ET.tostring(root), KEY, "blind-asset")["usable_contours"] == 31


def test_wrong_keys_are_rare(blind_svg):
    detections = 0
    for i in range(200):
        wrong = (i + 100).to_bytes(32, "big")
        detections += detect_blind(blind_svg, wrong, "blind-asset")["detected"]
    assert detections <= 2


def test_blind_cli(tmp_path):
    root = Path(__file__).parents[1]
    key = tmp_path / "owner.key"
    proposals = tmp_path / "proposals.json"
    svg = tmp_path / "blind.svg"
    proposals.write_bytes(subprocess.check_output([sys.executable, str(root / "examples" / "make_blind_proposals.py")]))
    cli = [sys.executable, "-m", "contourmark.cli"]
    # Detection is statistical; a fresh random key can legitimately miss the
    # fixed threshold. Pin the key so this checks the CLI contract, not power.
    key.write_text(KEY.hex() + "\n")
    subprocess.run(cli + ["sample-blind", str(proposals), str(svg), "--key", str(key), "--asset-id", "cli-asset"], check=True, capture_output=True)
    result = subprocess.run(cli + ["detect-blind", str(svg), "--key", str(key), "--asset-id", "cli-asset"], check=True, capture_output=True)
    assert b'"detected": true' in result.stdout
