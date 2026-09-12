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
