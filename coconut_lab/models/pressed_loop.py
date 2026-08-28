"""Fase C.5: ties drafter + locator + replacer into Pressed's full inference
pipeline. Two stages, kept as separate functions so the locate-and-replace
mechanism can be tested on its own (with a hand-written or pre-generated
draft) independent of drafter quality:

  1. draft: the drafter writes a full candidate reasoning/response.
  2. run_llr_loop: the classic Fase-B-style loop -- find the next
     "<<expr=result>>result" span (locator), recompute its correct value
     (replacer), splice it in, repeat until no such span remains.

For general chat text (no "<<...>>" patterns at all), run_llr_loop finds
nothing to fix and the draft passes through unchanged -- the same behavior
for reasoning and chat, no special-casing needed.
"""

import re
from dataclasses import dataclass, field

import torch

from coconut_lab.models.cracked import generate_response
from depth_lab.models.locator import Locator, extract_span_from_probs
from depth_lab.models.replacer import Replacer
from mini_llm.model import GPT
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN

DEFAULT_MAX_ITERATIONS = 20

# Not anchored at the start: BPE often merges a leading space into the same
# token as "<<" (byte-level BPE convention -- confirmed empirically, the
# locator's predicted span is consistently off by exactly that one leading
# space character), so span_text can legitimately start with a stray
# whitespace character before the real "<<expr=" pattern.
_EXPR_PATTERN = re.compile(r"<<([^=<>]+)=")


@torch.no_grad()
def predict_char_span(model: Locator, tokenizer: BPETokenizer, text: str, device: str) -> tuple[int, int]:
    """BPE-aware counterpart of depth_lab.models.locator.predict_span:
    that function returns a *token*-index span, which only equals a
    character span for depth_lab's char-level tokenizer. With BPE, a single
    token can cover several characters, so the token span returned by the
    model has to be mapped back to characters via encode_with_offsets --
    the exact bug this function exists to avoid repeating."""
    ids, offsets = tokenizer.encode_with_offsets(text)
    model.eval()
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(idx)
    probs = torch.sigmoid(logits[0])
    model.train()

    tok_start, tok_end = extract_span_from_probs(probs)
    char_start = offsets[tok_start][0]
    char_end = offsets[tok_end - 1][1]
    return char_start, char_end


@dataclass(frozen=True)
class PressedStep:
    text: str                # the full text before this step
    span: tuple[int, int]    # span the locator pointed at, in `text`
    span_text: str           # text[span[0]:span[1]]
    predicted_value: str     # replacer's recomputed value


@dataclass(frozen=True)
class PressedResult:
    draft: str
    final_text: str
    steps: list[PressedStep] = field(default_factory=list)


def _extract_expr(span_text: str) -> str | None:
    m = _EXPR_PATTERN.search(span_text)
    return m.group(1) if m else None


@torch.no_grad()
def _replace_value(replacer: Replacer, tokenizer: BPETokenizer, expr: str, device: str,
                    max_new_tokens: int = 12) -> str:
    pad_id = tokenizer.encode(EOT_TOKEN)[0]
    prompt_ids = tokenizer.encode(f"{expr[::-1]} => ")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    replacer.eval()
    out_ids: list[int] = []
    for grown in replacer.generate_stream(idx, max_new_tokens, temperature=1e-6):
        next_id = grown[0, -1].item()
        if next_id == pad_id:
            break
        out_ids.append(next_id)
    replacer.train()
    return tokenizer.decode(out_ids)


def run_llr_loop(locator: Locator, replacer: Replacer, tokenizer: BPETokenizer, draft: str, device: str,
                  max_iterations: int = DEFAULT_MAX_ITERATIONS) -> tuple[str, list[PressedStep]]:
    """The locate-and-replace part alone, operating on an already-written
    `draft` -- kept separate from drafting so it can be tested against a
    hand-written or pre-generated draft, independent of drafter quality."""
    current = draft
    steps: list[PressedStep] = []
    for _ in range(max_iterations):
        if "<<" not in current:
            break
        start, end = predict_char_span(locator, tokenizer, current, device)
        span_text = current[start:end]
        expr = _extract_expr(span_text)
        if expr is None:
            break  # locator pointed at something that isn't a real annotation -- can't safely proceed
        value = _replace_value(replacer, tokenizer, expr, device)
        steps.append(PressedStep(text=current, span=(start, end), span_text=span_text, predicted_value=value))
        current = current[:start] + value + current[end:]
    return current, steps


def reduce_with_pressed(drafter: GPT, locator: Locator, replacer: Replacer, tokenizer: BPETokenizer, prompt: str,
                         device: str, max_new_tokens_draft: int = 200,
                         max_iterations: int = DEFAULT_MAX_ITERATIONS) -> PressedResult:
    draft = generate_response(drafter, tokenizer, prompt, device, max_new_tokens=max_new_tokens_draft)
    final_text, steps = run_llr_loop(locator, replacer, tokenizer, draft, device, max_iterations)
    return PressedResult(draft=draft, final_text=final_text, steps=steps)
