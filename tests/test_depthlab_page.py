"""Headless regression test for the Fase B Streamlit page, using Streamlit's
own AppTest harness (no browser needed). Downloads the four Fase B
checkpoints from the HuggingFace Hub the first time it runs, then uses the
local HF cache -- same pattern as tests/test_streamlit_app.py."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "pages" / "1_Fase_B_Depth_Lab.py"


@pytest.mark.slow
def test_page_runs_without_exception_and_renders_title():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    assert not at.exception
    assert "Depth Lab" in at.title[0].value


@pytest.mark.slow
def test_random_expression_button_populates_the_text_input():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120)
    at.run()
    generate_button = next(b for b in at.sidebar.button if "aleatoria" in b.label)
    generate_button.click().run()
    assert not at.exception
    assert at.text_input[0].value.strip() != ""


@pytest.mark.slow
def test_evaluate_button_produces_a_correct_verdict_for_a_known_expression():
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)
    at.run()

    # depth-2 expression with a known, hand-verified value
    at.text_input[0].set_value("(True and (False or True))").run()
    evaluate_button = next(b for b in at.button if "Evaluar" in b.label)
    evaluate_button.click().run(timeout=180)

    assert not at.exception
    markdown_html = "\n".join(m.value for m in at.markdown)
    assert "Valor real:" in markdown_html
    assert "True" in markdown_html
    # all three architecture headers rendered
    assert "Decoder-only" in "\n".join(h.value for h in at.subheader)
    assert "Encoder-decoder" in "\n".join(h.value for h in at.subheader)
    assert "Looped Locate-and-Replace" in "\n".join(h.value for h in at.subheader)
