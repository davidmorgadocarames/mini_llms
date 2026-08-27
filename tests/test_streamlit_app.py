"""Headless regression test for the Streamlit demo, using Streamlit's own
AppTest harness (no browser needed). Downloads the checkpoint from the
HuggingFace Hub the first time it runs, then uses the local HF cache."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "streamlit_app.py"


@pytest.mark.slow
def test_app_runs_without_exception_and_renders_title():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value == "🥥 Coconut"


@pytest.mark.slow
def test_example_prompt_button_fills_the_text_area():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert at.text_area[0].value == at.button[0].label
