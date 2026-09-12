"""FROZEN Fase D configuration (end of Etapa 1) -- the single source of truth for
the tokenizer, the data sample and its order, the prefix-LM cuts, the fine-tuning
set, the hyperparameters and the three architectures.

Freeze rule (plan): if any of these change before/during Etapa 2, Cracked-D must
be retrained with the new config before Sliced-D and Cracked-D-full. The pretrain
scripts guard this via the per-batch hash registry.

Etapa 1 numbers (26.35M/26.21M params, 1200M tokens) were fixed after measuring
real throughput on the RTX 4060 at block_size 1024, bf16: ~50-55k tok/s for BOTH
Cracked-D and Sliced-D (Sliced is not slower because prefix-LM splits the work
between encoder and decoder). Those checkpoints were discarded after an
interactive test showed the 26M/1200M combination too undertrained for basic
chat coherence (see PARAMS/PRETRAIN_TOKENS below for the resized values), per
two literature-backed bottlenecks: knowledge capacity (~2 bits/parameter, Allen-Zhu
& Li, "Physics of Language Models 3.3", ICLR 2025) and tokens/parameter far below
comparable small chat models (TinyLlama-1.1B/3T tokens, SmolLM2-135M/2T,
SmolLM2-360M/4T -- Muennighoff et al., "Scaling Data-Constrained LM", NeurIPS 2023).

PRETRAIN_TOKENS=1.6B against the new ~80M-parameter architectures is the
Chinchilla floor (~20 tokens/parameter, Hoffmann et al. 2022) for BOTH Cracked-D
(80,628,480 params -> 19.84x) and Sliced-D (80,335,872 params -> 19.92x), chosen
deliberately at the floor rather than deeper into "overtrained small model"
territory (as TinyLlama/LLaMA do) to keep the wall-clock budget on a single
RTX 4060 in the few-day range instead of a week+.

VRAM headroom vs. compute utilization at the new ~80M size WAS measured with a
real short rehearsal (nvidia-smi polled every ~2.5s during real optimizer steps,
same method as Etapa 1) before committing to the full multi-day run, and it did
NOT scale linearly with parameters as naive FLOP-based estimates assumed:

  - At the OLD MICRO_BATCH=16/GRAD_ACCUM=16: Cracked-D peaked at 7930/8188 MB
    (97%, ~260 MB free) and throughput COLLAPSED to ~3k tok/s (vs. a ~17k tok/s
    linear-scaling estimate) -- the near-OOM pressure was causing allocator
    thrashing, not just tight headroom. Too risky for an unattended multi-hour run.
  - At MICRO_BATCH=8/GRAD_ACCUM=32 (same effective batch): Cracked-D peaked at
    83.7% (22k tok/s measured) but Sliced-D peaked at 96.0% (21k tok/s) -- still
    too tight for Sliced-D's cross-attention overhead specifically.
  - At MICRO_BATCH=4/GRAD_ACCUM=64 (same effective batch, this file's frozen
    values): Cracked-D peaked at 57% (4.67/8.19 GB, ~24k tok/s measured) and
    Sliced-D peaked at 59% (4.83/8.19 GB, ~26k tok/s measured) -- comparable
    headroom to Etapa 1's 60%/74%, and FASTER than the tight configurations
    above, not slower: avoiding allocator pressure more than compensates for the
    extra micro-step overhead. This is the frozen setting.
  - compare_lab/train/finetune.py has no grad-accum mechanism at all, so the
    same MICRO_BATCH=16-equivalent risk applies directly to FINETUNE_BATCH_SIZE:
    measured 96.8% VRAM at batch_size=16 vs. 57% at batch_size=4 (both real
    short rehearsals against a real pretrained-then-discarded checkpoint).
    FINETUNE_BATCH_SIZE is frozen at 4 for the same reason.

Real measured pretrain throughput (~24-26k tok/s) at 1.6B tokens implies
Cracked-D ~18.5h and Sliced-D ~17.1h of pretraining alone -- both notably FASTER
than the ~26h/model naive linear-scaling estimate, precisely because the smaller
micro-batch avoids the thrashing regime above.
"""

from dataclasses import dataclass, asdict

from mini_llm.model.config import GPTConfig
from compare_lab.models.sliced_d import SlicedDConfig
from compare_lab.data.tokenizer import SPECIAL_TOKENS

SEED = 1337
VOCAB_SIZE = 8192
BLOCK_SIZE = 1024

# --- pretraining sample (SmolLM-Corpus: cosmopedia-v2 + fineweb-edu-dedup 50/50) ---
PRETRAIN_TOKENS = 1_600_000_000     # ~800M + ~800M; Chinchilla floor (~20 tok/param) at ~80M params
PRETRAIN_VAL_TOKENS = 5_000_000
MICRO_BATCH = 4                    # fragments per forward (also the plan's batch id unit)
GRAD_ACCUM = 64                    # effective batch = 4*64*1024 ~= 262k tokens/step (VRAM-safe at ~80M, see docstring)
TOKENIZER_DOCS = 20_000            # streamed docs used to train the shared BPE

# --- fine-tuning (HuggingFaceTB/smol-smoltalk) ---
FINETUNE_TEST_FRAC = 0.02
# smol-smoltalk train has 460,341 conversations; this cap keeps fine-tuning
# proportionate to the multi-hour pretrainings. Must match what is executed --
# a "frozen" value that disagrees with the run is exactly what the freeze rule
# exists to prevent.
FINETUNE_MAX_CONVERSATIONS = 150_000
FINETUNE_BATCH_SIZE = 4            # finetune.py has no grad-accum; 16 measured at 96.8% VRAM at ~80M, 4 at 57% (see docstring)
FINETUNE_MAX_STEPS = 20_000        # ~2 epochs over the kept conversations

# --- optimizer / schedule ---
PRETRAIN = dict(lr=6e-4, min_lr=6e-5, warmup_steps=200, weight_decay=0.1, dtype="bfloat16")
FINETUNE = dict(lr=2e-4, min_lr=2e-5, warmup_steps=100, weight_decay=0.1, dtype="bfloat16")

# --- logging / checkpointing cadence ---
PRETRAIN_EVAL_INTERVAL = 500
FINETUNE_EVAL_INTERVAL = 500
LOG_INTERVAL = 20
# A hard failure is now the ONLY way a run stops other than the STOP file, so
# this interval is exactly what an unplanned stop costs. Measured: one
# checkpoint takes ~0.4s (fsync + verify included), so 5 min costs ~30s over
# the whole 6.3h run (0.13%) while capping the loss at 5 min instead of 15.
CKPT_INTERVAL_MIN = 5.0

# --- sampling for generation (identical across models for a fair comparison) ---
# Measured, do not "improve" by eye. Over 6 prompts x 5 seeds, lowering the
# temperature is worse on BOTH axes that matter here:
#   temp   replies ending at <eos>   repeated 4-grams
#   0.8              77%                    3%
#   0.6              67%                    4%
#   0.3              63%                   24%
# which is what Holtzman et al. 2020 ("The Curious Case of Neural Text
# Degeneration") predicts: maximization-based decoding CAUSES the degenerate
# repetition it looks like it should fix. 0.8 stays.
#
# Not adopted, deliberately. A repetition penalty (Keskar et al. 2019, CTRL
# section 4.1) is inference-only and needs no retraining, but measured here it
# moved 4-gram repetition by ~0-5 points in either direction -- repetition is
# not what makes this model's answers bad; missing knowledge at 26M params is.
# The training-time counterpart, unlikelihood training (Welleck et al., ICLR
# 2020), would need a full retraining run. Worth revisiting only if a larger
# model is trained from scratch -- bundle it with that run, not on its own.
GENERATION = dict(temperature=0.8, top_k=50, max_new_tokens=256)


def cracked_config() -> GPTConfig:
    """Cracked-D and Cracked-D-full: identical decoder-only, ~80.63M params."""
    return GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
                     n_layer=12, n_embd=768, n_head=12, n_kv_head=3)


def sliced_config() -> SlicedDConfig:
    """Sliced-D: encoder-decoder, same blocks, 7 enc / 4 dec -> ~80.34M (-0.36%)."""
    return SlicedDConfig(vocab_size=VOCAB_SIZE, d_model=768, n_head=12, n_kv_head=3,
                         n_enc_layer=7, n_dec_layer=4, max_src_len=BLOCK_SIZE, max_tgt_len=BLOCK_SIZE)


# Expected parameter counts (verified by instantiation; see tests). cracked_full
# shares Cracked-D's architecture (only the pretraining loss differs), so it gets
# the same expected count even though its training is deferred past this resize.
PARAMS = {"cracked": 80_628_480, "cracked_full": 80_628_480, "sliced": 80_335_872}
PARAM_TOLERANCE = 0.05


def as_dict() -> dict:
    return {
        "seed": SEED, "vocab_size": VOCAB_SIZE, "block_size": BLOCK_SIZE,
        "special_tokens": SPECIAL_TOKENS,
        "pretrain_tokens": PRETRAIN_TOKENS, "pretrain_val_tokens": PRETRAIN_VAL_TOKENS,
        "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
        "tokenizer_docs": TOKENIZER_DOCS,
        "finetune_test_frac": FINETUNE_TEST_FRAC,
        "finetune_max_conversations": FINETUNE_MAX_CONVERSATIONS,
        "finetune_batch_size": FINETUNE_BATCH_SIZE, "finetune_max_steps": FINETUNE_MAX_STEPS,
        "pretrain_eval_interval": PRETRAIN_EVAL_INTERVAL,
        "finetune_eval_interval": FINETUNE_EVAL_INTERVAL,
        "log_interval": LOG_INTERVAL, "ckpt_interval_min": CKPT_INTERVAL_MIN,
        "pretrain_optim": PRETRAIN, "finetune_optim": FINETUNE, "generation": GENERATION,
        "cracked": asdict(cracked_config()), "sliced": asdict(sliced_config()),
        "params": PARAMS, "param_tolerance": PARAM_TOLERANCE,
    }
