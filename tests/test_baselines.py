import subprocess
from pathlib import Path

from contourmark.baselines import metadata_detect, metadata_embed, numeric_lsb_detect, numeric_lsb_embed


KEY = bytes(range(32))
SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
       b'<path fill="none" stroke="black" d="M10 10 C20 11 30 12 40 13 C50 14 60 15 70 16 '
       b'C80 17 90 18 100 19 C110 20 120 21 130 22 C140 23 150 24 160 25 '
       b'C170 26 180 27 190 28 C180 29 170 30 160 31 C150 32 140 33 130 34 '
       b'C120 35 110 36 100 37 C90 38 80 39 70 40 C60 41 50 42 40 43 Z"/></svg>')


def test_metadata_baseline_optimizer_behavior(tmp_path):
    marked = metadata_embed(SVG, "owner")
    assert metadata_detect(marked, "owner")
    before, after = tmp_path / "before.svg", tmp_path / "after.svg"
    before.write_bytes(marked)
    scour = Path(__file__).parents[1] / ".venv/bin/scour"
    subprocess.run([str(scour), "-i", str(before), "-o", str(after), "--indent=none"], check=True, capture_output=True)
    assert metadata_detect(after.read_bytes(), "owner")
    svgo = Path(__file__).parents[1] / "node_modules/.bin/svgo"
    subprocess.run([str(svgo), str(before), "-o", str(after)], check=True, capture_output=True)
    assert not metadata_detect(after.read_bytes(), "owner")


def test_numeric_lsb_roundtrip_and_rounding_removal():
    marked = numeric_lsb_embed(SVG, KEY, "fixture", chips=64)
    result = numeric_lsb_detect(marked, KEY, "fixture", chips=64)
    assert result["detected"] and result["matches"] == 64
    # Two-decimal formatting is a visually tiny keyless edit at this scale.
    import re
    rounded = re.sub(
        rb"[-+]?(?:\d*\.\d+|\d+\.\d*)(?:[Ee][-+]?\d+)?",
        lambda match: f"{float(match.group()):.2f}".encode(),
        marked,
    )
    assert not numeric_lsb_detect(rounded, KEY, "fixture", chips=64)["detected"]
