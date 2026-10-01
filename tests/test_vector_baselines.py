from contourmark import vector_baselines as vb
from contourmark.attacks import round_absolute

KEY = bytes(range(32))
SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="'
       + b"".join(f"M{10 + i} {10 + 3 * i}C{20 + i} 0 {40 + i} 0 {50 + i} {10 + 3 * i}C{60 + i} 30 {80 + i} 30 {90 - i} {12 + 3 * i}".encode() for i in range(12))
       + b'"/></svg>')


def test_numeric_lsb_survives_identity_not_rounding():
    marked = vb.numeric_lsb_embed(SVG, KEY)
    assert vb.numeric_lsb_detect(marked, KEY)["detected"]
    assert not vb.numeric_lsb_detect(SVG, KEY)["detected"]
    assert not vb.numeric_lsb_detect(round_absolute(2)(marked), KEY)["detected"]


def test_bezier_split_reads_back_keyed_ratios():
    marked = vb.bezier_split_embed(SVG, KEY)
    result = vb.bezier_split_detect(marked, KEY)
    assert result["splits"] == 24 and result["matches"] == 24
    assert vb.bezier_split_detect(SVG, KEY)["splits"] == 0


def test_metadata_element_round_trip():
    marked = vb.metadata_element_embed(SVG, KEY)
    assert vb.metadata_element_detect(marked, KEY)["detected"]
    assert not vb.metadata_element_detect(SVG, KEY)["detected"]
