"""Fase D Streamlit app, driven headless with AppTest (same approach as the
Fase A/B page tests). Marked slow: it loads the real ~80M checkpoint. Deployed
standalone (streamlit_app_fase_d.py at the repo root, not under pages/) -- see
that file's docstring for why."""

from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "streamlit_app_fase_d.py"
CKPT = (Path(__file__).resolve().parent.parent / "compare_lab" / "checkpoints"
        / "cracked" / "finetune_final.pt")

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="Cracked-D not trained yet")


@pytest.mark.slow
def test_page_loads_and_shows_the_banner_and_warning():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=300).run()
    assert not at.exception, at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Cracked-D" in body
    # the page must be honest about the scale; the plan requires this warning
    assert "80M" in body
    assert any("Reiniciar" in b.label for b in at.button)


@pytest.mark.slow
def test_page_answers_a_message_and_keeps_history():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGE), default_timeout=600).run()
    assert not at.exception, at.exception

    at.text_input(key="fased_prompt").set_value("What is the capital of France?").run()
    assert not at.exception, at.exception

    history = at.session_state["fased_history"]
    assert len(history) == 2, f"expected user+assistant, got {history}"
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"].strip(), "the assistant replied with nothing"
