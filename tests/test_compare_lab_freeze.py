"""The freeze rule, enforced against the checkpoints that were actually trained.

config.py claims to be the single source of truth for Fase D. That claim is only
worth anything if what ran matches it -- and it once did not: finetune's CLI
defaults said batch 8 / 2000 steps while the frozen config said 16 / 20000, which
produced a "finished" checkpoint with zero steps of training.

So this compares the frozen values against the hyperparameters and shapes stored
inside each trained checkpoint. In Etapa 2 it extends to Sliced-D and
Cracked-D-full for free: if a config changes, Cracked-D has to be retrained
before them, and this test is what makes that visible instead of silent.

Skipped when an architecture has not been trained yet.
"""

from pathlib import Path

import pytest
import torch

from compare_lab import config as C

CKPT_DIR = Path(__file__).resolve().parent.parent / "compare_lab" / "checkpoints"

ARCHS = ["cracked", "cracked_full", "sliced"]


def _load(arch: str, stage: str):
    path = CKPT_DIR / arch / f"{stage}_final.pt"
    if not path.exists():
        pytest.skip(f"{arch} {stage} not trained yet")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.slow
@pytest.mark.parametrize("arch", ARCHS)
def test_pretrain_ran_with_the_frozen_hyperparameters(arch):
    ckpt = _load(arch, "pretrain")
    hp = ckpt["train_hparams"]

    # a single epoch over the frozen token budget, at the frozen batch geometry
    expected_steps = C.PRETRAIN_TOKENS // (C.MICRO_BATCH * C.GRAD_ACCUM * C.BLOCK_SIZE)
    assert ckpt["step"] == expected_steps, (
        f"{arch} pretrained for {ckpt['step']} steps, frozen config implies {expected_steps}")

    assert hp["grad_accum"] == C.GRAD_ACCUM
    assert hp["lr"] == C.PRETRAIN["lr"]
    assert hp["min_lr"] == C.PRETRAIN["min_lr"]
    assert hp["warmup_steps"] == C.PRETRAIN["warmup_steps"]


@pytest.mark.slow
@pytest.mark.parametrize("arch", ARCHS)
def test_finetune_reached_the_frozen_step_target(arch):
    ckpt = _load(arch, "finetune")
    assert ckpt["step"] == C.FINETUNE_MAX_STEPS, (
        f"{arch} fine-tuned for {ckpt['step']} steps, frozen target is {C.FINETUNE_MAX_STEPS}")


@pytest.mark.slow
@pytest.mark.parametrize("arch", ARCHS)
def test_trained_architecture_matches_the_frozen_one(arch):
    ckpt = _load(arch, "pretrain")
    got = ckpt["config"]

    assert got["vocab_size"] == C.VOCAB_SIZE
    if arch == "sliced":
        frozen = C.sliced_config()
        assert got["d_model"] == frozen.d_model
        assert got["n_enc_layer"] == frozen.n_enc_layer
        assert got["n_dec_layer"] == frozen.n_dec_layer
        assert got["max_src_len"] == frozen.max_src_len
    else:
        frozen = C.cracked_config()
        assert got["block_size"] == frozen.block_size
        assert got["n_layer"] == frozen.n_layer
        assert got["n_embd"] == frozen.n_embd
        assert got["n_head"] == frozen.n_head
        assert got["n_kv_head"] == frozen.n_kv_head


@pytest.mark.slow
@pytest.mark.parametrize("arch", ARCHS)
def test_published_weights_load_into_the_frozen_model_at_the_frozen_size(arch):
    """The comparison's whole premise is a matched parameter budget, so the
    weights actually shipped must load into the frozen architecture and carry the
    count the tables report.

    Counted from the built model, not from the state_dict's keys: tok_emb and
    lm_head are tied, so they appear as two entries sharing one tensor and naive
    key summing over-counts by vocab_size*d_model (4,194,304 here, a 15.9% error).
    """
    from compare_lab.train.pretrain import build_model

    ckpt = _load(arch, "finetune")
    model = build_model(arch, C.VOCAB_SIZE, C.BLOCK_SIZE, "cpu")
    # strict=True: a checkpoint that does not fit the frozen architecture exactly
    # is a broken freeze, which is the whole point of this module
    model.load_state_dict(ckpt["model"])

    n = sum(p.numel() for p in model.parameters())
    expected = C.PARAMS[arch]
    drift = abs(n - expected) / expected
    assert drift <= C.PARAM_TOLERANCE, (
        f"{arch} checkpoint has {n:,} params, frozen table says {expected:,} ({drift:.1%} off)")
