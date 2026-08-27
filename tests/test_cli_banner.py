from mini_llm._version import __version__
from mini_llm.cli.banner import render_banner


def test_banner_has_three_lines_matching_the_logo_height():
    banner = render_banner("test model info")
    lines = banner.split("\n")
    assert len(lines) == 3


def test_banner_includes_version_and_model_info():
    banner = render_banner("26.4M params · step 20,000")
    assert __version__ in banner
    assert "26.4M params · step 20,000" in banner


def test_banner_coconut_icon_has_three_eyes():
    banner = render_banner("info")
    assert banner.count("●") == 3
