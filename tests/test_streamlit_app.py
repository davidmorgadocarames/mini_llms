"""Headless regression test for the Streamlit demo, using Streamlit's own
AppTest harness (no browser needed). Downloads the checkpoint from the
HuggingFace Hub the first time it runs, then uses the local HF cache."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.mark.slow
def test_app_runs_without_exception_and_renders_banner():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    markdown_html = "\n".join(m.value for m in at.markdown)
    assert "coconut-banner" in markdown_html  # the large ASCII logo
    assert "coconut-banner-compact" in markdown_html  # the mobile fallback
    assert "Coconut" in markdown_html  # the caption line
    assert "tokens generados" in markdown_html  # the live token counter


@pytest.mark.slow
def test_example_prompt_button_fills_the_text_area():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert at.text_area[0].value == at.button[0].label
