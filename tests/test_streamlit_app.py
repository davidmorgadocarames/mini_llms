"""Headless regression test for the Streamlit demo, using Streamlit's own
AppTest harness (no browser needed). Downloads the checkpoint from the
HuggingFace Hub the first time it runs, then uses the local HF cache."""

import html
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mini_llm.tokenizer import clean_for_display

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
def test_no_separate_generate_button():
    """Generation is triggered by submitting the prompt (Enter) or clicking
    a chip, not a dedicated button -- matching docs/index.html."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert "Generar" not in [b.label for b in at.button]


@pytest.mark.slow
def test_chip_click_immediately_triggers_real_generation():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["last_output_prompt"] == at.button[0].label
    assert at.session_state["last_output_body"].strip() != ""
    assert at.session_state["total_tokens"] > 0

    markdown_html = "\n".join(m.value for m in at.markdown)
    assert "prompt-echo" in markdown_html
    assert html.escape(at.button[0].label) in markdown_html


@pytest.mark.slow
def test_submitting_prompt_input_triggers_generation_exactly_once():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.text_input[0].set_value("The war began when").run()
    assert not at.exception
    assert at.session_state["last_processed"] == "The war began when"

    tokens_after_first = at.session_state["total_tokens"]
    at.run()  # an unrelated rerun with the same value must not regenerate
    assert at.session_state["total_tokens"] == tokens_after_first


@pytest.mark.slow
def test_generated_output_persists_in_the_scroll_panel_after_a_rerun():
    """The output must survive a script rerun triggered by something else
    (e.g. clicking a chip) -- otherwise the fixed-height panel would go
    blank the moment you interact with anything else."""
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    at.button[0].click().run()

    markdown_html = "\n".join(m.value for m in at.markdown)
    assert "coconut-output" not in markdown_html  # old id, should be gone
    assert "output-frame" in markdown_html
    body = at.session_state["last_output_body"]
    assert body

    # unrelated rerun (clicking a different example chip)
    at.button[1].click().run()
    markdown_html_after = "\n".join(m.value for m in at.markdown)
    # rendering applies clean_for_display() (see test_tokenizer_display.py),
    # so compare against the same transformation, not the raw stored text
    assert html.escape(clean_for_display(at.session_state["last_output_body"])) in markdown_html_after
