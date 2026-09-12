"""FROZEN Fase D configuration (end of Etapa 1) -- the single source of truth for
the tokenizer, the data sample and its order, the prefix-LM cuts, the fine-tuning
set, the hyperparameters and the three architectures.

Freeze rule (plan): if any of these change before/during Etapa 2, Cracked-D must
be retrained with the new config before Sliced-D and Cracked-D-full. The pretrain
scripts guard this via the per-batch hash registry.

Numbers were fixed after measuring real throughput on the RTX 4060 at block_size
1024, bf16: ~50-55k tok/s for BOTH Cracked-D and Sliced-D (Sliced is not slower
because prefix-LM splits the work between encoder and decoder). So 1200M corpus
tokens x 3 single-epoch pretrainings ~= 20h, under the 24h budget.

VRAM headroom vs. compute utilization, measured at MICRO_BATCH=16/GRAD_ACCUM=16
(this file's frozen values) with nvidia-smi polled every 0.4s during a real
15-step run: Cracked-D peaks at 4.9 GB/8.2 GB (60%, ~3.2 GB free) with GPU
utilization 96% mean / 100% max; Sliced-D peaks at 6.0 GB/8.2 GB (74%, ~2.1 GB
free) with 99% mean / 100% max. Both keep meaningful headroom below the 8 GB
card limit while the GPU's compute is essentially never idle.
"""

from dataclasses import dataclass, asdict

from mini_llm.model.config import GPTConfig
from compare_lab.models.sliced_d import SlicedDConfig
from compare_lab.data.tokenizer import SPECIAL_TOKENS

SEED = 1337
VOCAB_SIZE = 8192
BLOCK_SIZE = 1024

# --- pretraining sample (SmolLM-Corpus: cosmopedia-v2 + fineweb-edu-dedup 50/50) ---
PRETRAIN_TOKENS = 1_200_000_000     # ~600M + ~600M, single epoch per model
PRETRAIN_VAL_TOKENS = 5_000_000
MICRO_BATCH = 16                    # fragments per forward (also the plan's batch id unit)
GRAD_ACCUM = 16                    # effective batch = 16*16*1024 ~= 262k tokens/step
TOKENIZER_DOCS = 20_000            # streamed docs used to train the shared BPE

# --- fine-tuning (HuggingFaceTB/smol-smoltalk) ---
FINETUNE_TEST_FRAC = 0.02
# smol-smoltalk train has 460,341 conversations; this cap keeps fine-tuning
# proportionate to the multi-hour pretrainings. Must match what is executed --
# a "frozen" value that disagrees with the run is exactly what the freeze rule
# exists to prevent.
FINETUNE_MAX_CONVERSATIONS = 150_000
FINETUNE_BATCH_SIZE = 16
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
    """Cracked-D and Cracked-D-full: identical decoder-only, ~26.35M params."""
    return GPTConfig(vocab_size=VOCAB_SIZE, block_size=BLOCK_SIZE,
                     n_layer=8, n_embd=512, n_head=8, n_kv_head=2)


def sliced_config() -> SlicedDConfig:
    """Sliced-D: encoder-decoder, same blocks, 3 enc / 4 dec -> ~26.21M (-0.55%)."""
    return SlicedDConfig(vocab_size=VOCAB_SIZE, d_model=512, n_head=8, n_kv_head=2,
                         n_enc_layer=3, n_dec_layer=4, max_src_len=BLOCK_SIZE, max_tgt_len=BLOCK_SIZE)


# Expected parameter counts (verified by instantiation; see tests).
PARAMS = {"cracked": 26_354_176, "cracked_full": 26_354_176, "sliced": 26_208_256}
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
