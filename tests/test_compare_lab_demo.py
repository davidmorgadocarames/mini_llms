"""Demo load/generate logic (Streamlit-free): a locally-saved Cracked-D
checkpoint loads and generates a reply that stops at <eos>."""

import dataclasses

import torch

from mini_llm.model.config import GPTConfig
from mini_llm.model.transformer import GPT
from compare_lab.demo import load_from_local, chat


def test_load_from_local_and_chat(tiny_tokenizer, tiny_tokenizer_dir, tmp_path):
    cfg = GPTConfig(vocab_size=tiny_tokenizer.vocab_size, block_size=128,
                    n_layer=2, n_embd=64, n_head=4, n_kv_head=2)
    model = GPT(cfg)
    ckpt = tmp_path / "cracked_d_final.pt"
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "step": 0}, ckpt)

    loaded, tok = load_from_local(ckpt, tiny_tokenizer_dir, device="cpu")
    assert loaded.num_parameters() == model.num_parameters()

    reply = chat(loaded, tok, [{"role": "user", "content": "hello there"}],
                 device="cpu", max_new_tokens=16)
    assert isinstance(reply, str)  # untrained model, but must return and stop within budget


def test_chat_takes_its_sampling_from_the_frozen_config(monkeypatch):
    """No second copy of the sampling settings.

    All three architectures must generate under identical settings or the
    comparison is not fair, so config.GENERATION has to be the only place these
    numbers live. They were duplicated as literal defaults in chat(), and the
    Fase D page separately passed max_new_tokens=200 against a frozen 256 -- the
    same drift that already shipped a fine-tuning run at batch 8 / 2000 steps
    while the frozen config said 16 / 20000.
    """
    import compare_lab.demo as demo
    from compare_lab import config

    seen = {}

    def fake_generate(model, tok, messages, **kw):
        seen.update(kw)
        return "ok"

    monkeypatch.setattr(demo, "generate_response", fake_generate)
    demo.chat(None, None, [{"role": "user", "content": "hi"}])

    assert seen["temperature"] == config.GENERATION["temperature"]
    assert seen["top_k"] == config.GENERATION["top_k"]
    assert seen["max_new_tokens"] == config.GENERATION["max_new_tokens"]


def test_explicit_arguments_still_win_over_the_frozen_defaults(monkeypatch):
    """Tests and experiments need to shorten generation; only the *defaults*
    come from the config."""
    import compare_lab.demo as demo

    seen = {}
    monkeypatch.setattr(demo, "generate_response",
                        lambda model, tok, messages, **kw: seen.update(kw) or "ok")
    demo.chat(None, None, [{"role": "user", "content": "hi"}], max_new_tokens=7, temperature=0.1)

    assert seen["max_new_tokens"] == 7
    assert seen["temperature"] == 0.1
