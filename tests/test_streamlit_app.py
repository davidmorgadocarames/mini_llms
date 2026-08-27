"""Headless regression test for the Streamlit demo, using Streamlit's own
AppTest harness (no browser needed). Downloads the checkpoint from the
HuggingFace Hub the first time it runs, then uses the local HF cache."""

import html
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
    assert "coconut-banner-img" in markdown_html  # the pre-rendered logo PNG
    assert "data:image/png;base64," in markdown_html
    assert "Coconut" in markdown_html  # the caption line
    assert "tokens generados" in markdown_html  # the live token counter
    assert "terminal-head-bar" in markdown_html  # window-title bar above the fixed-height panel


@pytest.mark.slow
def test_example_prompt_button_fills_the_prompt_input():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert at.text_input[0].value == at.button[0].label


@pytest.mark.slow
def test_generated_output_persists_in_the_scroll_panel_after_a_rerun():
    """The output must survive a script rerun triggered by something else
    (e.g. clicking a chip) -- otherwise the fixed-height panel would go
    blank the moment you interact with anything else, defeating the point
    of a persistent scrollback."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()
    gen_btn = [b for b in at.button if b.label == "Generar"][0]
    gen_btn.click().run()

    markdown_html = "\n".join(m.value for m in at.markdown)
    assert "coconut-output" in markdown_html
    assert at.session_state["last_output"]

    # unrelated rerun (clicking a different example chip)
    at.button[1].click().run()
    markdown_html_after = "\n".join(m.value for m in at.markdown)
    assert html.escape(at.session_state["last_output"]) in markdown_html_after
