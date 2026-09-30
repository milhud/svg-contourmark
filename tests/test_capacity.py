import json

from contourmark.capacity import assess_svg
from contourmark.cli import main


def test_path_only_document_is_eligible():
    result = assess_svg(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path d="M0 0L10 10"/></svg>')
    assert result["whole_document_supported"]
    assert result["supported_contours"] == 1
    assert result["choice_capacity_bits"] is None
    assert not result["universal_watermark_guarantee"]


def test_omnisvg_filling_metadata_is_inert():
    source = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><path filling="0" fill="#000" d="M0 0L10 10"/></svg>'
    result = assess_svg(source)
    assert result["whole_document_supported"]
    assert result["status"] == "eligible_path_subset"


def test_mixed_content_fails_closed():
    source = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">
      <path d="M0 0L10 10"/><text x="1" y="5">A</text>
      <image href="data:image/png;base64,AA==" width="1" height="1"/>
      <filter id="f"><feGaussianBlur stdDeviation="1"/></filter><rect width="2" height="2"/>
    </svg>'''
    result = assess_svg(source)
    assert not result["whole_document_supported"]
    assert result["unsupported_feature_counts"]["text"] == 1
    assert result["unsupported_feature_counts"]["embedded_raster"] == 1
    assert result["unsupported_feature_counts"]["filter"] == 2
    assert result["unsupported_feature_counts"]["rigid_primitive"] == 1


def test_assess_cli(tmp_path, capsys):
    svg = tmp_path / "mixed.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><circle cx="1" cy="1" r="1"/></svg>')
    assert main(["assess", str(svg)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "no_supported_geometric_carrier"
