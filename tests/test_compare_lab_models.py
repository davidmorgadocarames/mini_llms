"""Models: parameter counts matched within 5%, forward shapes, overfit sanity."""

import torch

from compare_lab import config
from compare_lab.models.cracked_d import build_model as build_cracked
from compare_lab.models.sliced_d import SlicedD, SlicedDConfig


def test_param_counts_match_frozen_values():
    cracked = build_cracked(config.cracked_config())
    sliced = SlicedD(config.sliced_config())
    assert cracked.num_parameters() == config.PARAMS["cracked"]
    assert sliced.num_parameters() == config.PARAMS["sliced"]


def test_sliced_within_tolerance_of_cracked():
    n_cracked = build_cracked(config.cracked_config()).num_parameters()
    n_sliced = SlicedD(config.sliced_config()).num_parameters()
    rel = abs(n_sliced - n_cracked) / n_cracked
    assert rel <= config.PARAM_TOLERANCE, f"{rel:.3%} exceeds {config.PARAM_TOLERANCE:.0%}"


def test_forward_shapes_with_padding():
    cracked = build_cracked(config.cracked_config())
    idx = torch.randint(6, 8000, (2, 50))
    logits, _ = cracked(idx)
    assert logits.shape == (2, 50, config.VOCAB_SIZE)

    sliced = SlicedD(SlicedDConfig(n_enc_layer=3, n_dec_layer=4, max_src_len=64, max_tgt_len=64))
    src = torch.randint(6, 8000, (2, 40))
    tgt = torch.randint(6, 8000, (2, 20))
    src_pad = torch.zeros(2, 40, dtype=torch.bool)
    src_pad[:, 30:] = True
    logits, _ = sliced(src, tgt, src_key_padding_mask=src_pad)
    assert logits.shape == (2, 20, config.VOCAB_SIZE)


def test_sliced_overfits_tiny_batch():
    """A masking/cross-attention bug would stop the encoder-decoder from
    overfitting a handful of padded examples."""
    torch.manual_seed(0)
    cfg = SlicedDConfig(vocab_size=64, d_model=64, n_head=4, n_kv_head=2,
                        n_enc_layer=2, n_dec_layer=2, max_src_len=32, max_tgt_len=32)
    m = SlicedD(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    src = torch.randint(6, 64, (4, 16))
    src_pad = torch.zeros(4, 16, dtype=torch.bool)
    src_pad[:, 12:] = True
    tgt = torch.randint(6, 64, (4, 10))
    import torch.nn.functional as F
    for _ in range(200):
        logits, _ = m(src, tgt[:, :-1], src_key_padding_mask=src_pad)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1


def test_both_architectures_stop_generation_at_eos(tiny_tokenizer):
    """Both generate_response helpers must cut at <eos>. The generators
    themselves deliberately never stop (they just yield), so a missing caller-side
    break would make the model ramble to the token budget every time -- the exact
    bug Fase C carried, and what the plan forbids ("la generacion se detiene en
    <eos>")."""
    import torch
    from compare_lab.models.cracked_d import build_model as build_cracked, build_config
    from compare_lab.models.sliced_d import SlicedD, SlicedDConfig, generate_response as sliced_gen
    from compare_lab.models.cracked_d import generate_response as cracked_gen

    tok = tiny_tokenizer
    msgs = [{"role": "user", "content": "hello"}]

    class _AlwaysEos(torch.nn.Module):
        """Forces <eos> as the very first sampled token; a correct caller then
        returns an EMPTY string instead of max_new_tokens of noise."""
        def __init__(self, inner, eos_id):
            super().__init__()
            self.inner, self.eos_id = inner, eos_id
            self.config = inner.config

        def eval(self):
            self.inner.eval(); return self

        def generate_stream(self, *a, **kw):
            base = a[0] if a else kw["src_idx"]
            row = torch.tensor([[self.eos_id]] * base.size(0))
            for _ in range(5):
                yield torch.cat([base[:, :1], row], dim=1)

    cracked = build_cracked(build_config(vocab_size=tok.vocab_size, block_size=128))
    assert cracked_gen(_AlwaysEos(cracked, tok.eos_id), tok, msgs,
                       max_new_tokens=16, device="cpu") == ""

    sliced = SlicedD(SlicedDConfig(vocab_size=tok.vocab_size, max_src_len=128, max_tgt_len=128))
    assert sliced_gen(_AlwaysEos(sliced, tok.eos_id), tok, msgs,
                      max_new_tokens=16, device="cpu") == ""


def test_sliced_generate_response_runs_end_to_end(tiny_tokenizer):
    from compare_lab.models.sliced_d import SlicedD, SlicedDConfig, generate_response

    tok = tiny_tokenizer
    m = SlicedD(SlicedDConfig(vocab_size=tok.vocab_size, d_model=64, n_head=4, n_kv_head=2,
                              n_enc_layer=2, n_dec_layer=2, max_src_len=64, max_tgt_len=64))
    out = generate_response(m, tok, [{"role": "user", "content": "hi there"}],
                            max_new_tokens=8, device="cpu")
    assert isinstance(out, str)
