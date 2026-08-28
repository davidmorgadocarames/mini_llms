"""Fase C.6 item 3: custom adapter wiring Cracked/Sliced/Pressed into
EleutherAI's lm-evaluation-harness (the de-facto standard eval harness).

None of the three architectures are `transformers`-compatible (they're
hand-built from scratch, per this project's whole point -- see CLAUDE.md),
so the harness's automatic HF loader can't load them. lm-eval-harness does
support arbitrary custom models, though: passing an `lm_eval.api.model.LM`
subclass directly to `simple_evaluate(model=...)` is enough. This module
implements that subclass for our own architectures instead of forcing them
into the HF interface.

`loglikelihood` (context, continuation) -> (logprob, is_greedy) is the
request type both chosen tasks (lambada_openai, piqa) use -- it's exactly
the training loss computation (cross-entropy = -log P(token)), just summed
over the continuation instead of averaged, with no gradient:

- Cracked / Pressed's drafter (decoder-only GPT): concatenate context and
  continuation into one sequence, run a single forward pass, and read off
  log P(continuation token | everything before it) at each continuation
  position -- literally what next-token training already optimizes.
- Sliced (encoder-decoder): context -> encoder input, continuation ->
  teacher-forced decoder target, exactly how it was trained (src=prompt,
  tgt=response in coconut_lab/models/sliced.py). Not the mismatch it might
  look like at first -- the encode/decode split maps directly onto
  context/continuation.
- Pressed as a whole (not just its drafter) has no meaningful role here:
  the locator/replacer only ever change text that's still being generated
  (looking for `<<expr=result>>` spans in a fresh draft) -- there's no
  generation step to correct when the continuation is already fixed by the
  task. So "Pressed" on loglikelihood-based tasks is, honestly and by
  construction, identical to scoring with its drafter alone. Documented
  here rather than silently treating them as different.

`generate_until` and `loglikelihood_rolling` are implemented for ABC
completeness (lm_eval.api.model.LM requires all three) but are not
exercised by lambada_openai/piqa -- neither task issues those request
types.
"""

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from coconut_lab.models import cracked as cracked_mod
from coconut_lab.models import sliced as sliced_mod
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN


class GPTFamilyAdapter(LM):
    """Covers Cracked and Pressed's drafter -- both are plain mini_llm.model.GPT
    instances, so the loglikelihood/generation logic is identical; only the
    checkpoint differs."""

    def __init__(self, model: GPT, tokenizer: BPETokenizer, device: str, label: str):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self._device = device
        self.label = label
        self.eot_id = tokenizer.encode(EOT_TOKEN)[0]
        self.block_size = model.config.block_size

    @torch.no_grad()
    def _loglikelihood_one(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_ids = self.tokenizer.encode(context) if context else [self.eot_id]
        cont_ids = self.tokenizer.encode(continuation)
        if not cont_ids:
            return 0.0, True

        full_ids = ctx_ids + cont_ids
        n_cont = len(cont_ids)
        # Keep the tail if the sequence overflows block_size -- the
        # continuation (and enough context to condition it) always survives;
        # only the earliest context tokens get dropped.
        if len(full_ids) > self.block_size:
            full_ids = full_ids[-self.block_size:]
        cont_start = len(full_ids) - n_cont

        idx = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        logits, _ = self.model(idx)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        total_logprob = 0.0
        is_greedy = True
        for j in range(n_cont):
            pos = cont_start + j  # position of the continuation token in full_ids
            pred_logits_pos = pos - 1  # logits at pos-1 predict the token at pos
            token_id = full_ids[pos]
            total_logprob += log_probs[0, pred_logits_pos, token_id].item()
            if log_probs[0, pred_logits_pos, :].argmax().item() != token_id:
                is_greedy = False
        return total_logprob, is_greedy

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return [self._loglikelihood_one(*req.args) for req in requests]

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tokenizer.encode(text)
            total = 0.0
            for start in range(0, len(ids), self.block_size):
                chunk = ids[start:start + self.block_size]
                if len(chunk) < 2:
                    continue
                idx = torch.tensor([chunk], dtype=torch.long, device=self.device)
                logits, _ = self.model(idx)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                for j in range(1, len(chunk)):
                    total += log_probs[0, j - 1, chunk[j]].item()
            results.append(total)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs = []
        for req in requests:
            context, gen_kwargs = req.args
            gen_kwargs = gen_kwargs or {}
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_gen_toks", 128)
            text = cracked_mod.generate_response(self.model, self.tokenizer, context, self.device,
                                                  max_new_tokens=max_new_tokens, temperature=0.0001, top_k=1)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
            outputs.append(text)
        return outputs


class SlicedAdapter(LM):
    """Sliced (encoder-decoder): context -> encoder, continuation -> teacher-
    forced decoder target, using the exact src/tgt split it was trained on."""

    def __init__(self, model, tokenizer: BPETokenizer, device: str):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self._device = device
        self.bos_id = tokenizer.encode(EOT_TOKEN)[0]
        self.src_block_size = sliced_mod.SRC_BLOCK_SIZE
        self.tgt_block_size = sliced_mod.TGT_BLOCK_SIZE

    @torch.no_grad()
    def _loglikelihood_one(self, context: str, continuation: str) -> tuple[float, bool]:
        cont_ids = self.tokenizer.encode(continuation)
        if not cont_ids:
            return 0.0, True
        # Some harness tasks (e.g. piqa's answer choices) are full sentences,
        # not single words -- Sliced's decoder window is fixed-size, so an
        # over-long continuation has to be truncated rather than crash the
        # whole eval run. Matches Seq2SeqDataset's own tgt_block_size budget.
        if len(cont_ids) > self.tgt_block_size:
            cont_ids = cont_ids[:self.tgt_block_size]

        src_ids = self.tokenizer.encode(context)[:self.src_block_size] if context else [self.bos_id]
        src_ids = src_ids + [self.bos_id] * (self.src_block_size - len(src_ids))
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_pad_mask = src == self.bos_id
        memory = self.model.encode(src, src_pad_mask)

        tgt_in = torch.tensor([[self.bos_id] + cont_ids[:-1]], dtype=torch.long, device=self.device)
        logits = self.model.decode(tgt_in, memory, src_pad_mask)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        total_logprob = 0.0
        is_greedy = True
        for j, token_id in enumerate(cont_ids):
            total_logprob += log_probs[0, j, token_id].item()
            if log_probs[0, j, :].argmax().item() != token_id:
                is_greedy = False
        return total_logprob, is_greedy

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return [self._loglikelihood_one(*req.args) for req in requests]

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        # Sliced has no single-sequence causal reading of a document the way
        # a decoder-only model does -- score it as its own continuation with
        # an empty context, chunked to fit the decoder's block size.
        results = []
        for req in requests:
            (text,) = req.args
            ids = self.tokenizer.encode(text)
            total = 0.0
            for start in range(0, len(ids), sliced_mod.TGT_BLOCK_SIZE):
                chunk = ids[start:start + sliced_mod.TGT_BLOCK_SIZE]
                if not chunk:
                    continue
                logprob, _ = self._loglikelihood_one("", self.tokenizer.decode(chunk))
                total += logprob
            results.append(total)
        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        outputs = []
        for req in requests:
            context, gen_kwargs = req.args
            gen_kwargs = gen_kwargs or {}
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_new_tokens = gen_kwargs.get("max_gen_toks", 128)
            text = sliced_mod.generate_response(self.model, self.tokenizer, context, self.device,
                                                 max_new_tokens=max_new_tokens)
            for stop in until:
                idx = text.find(stop)
                if idx != -1:
                    text = text[:idx]
            outputs.append(text)
        return outputs
