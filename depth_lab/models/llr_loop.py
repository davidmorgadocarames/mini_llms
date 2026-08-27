"""Ties the locator and replacer together into the full Looped
Locate-and-Replace inference procedure: repeatedly (1) ask the locator which
span of the current expression string is the next innermost reducible
sub-expression, (2) ask the replacer what that span's value is, (3) splice
the predicted value back into the string in place of the span, and (4)
repeat until a bare "True"/"False" literal remains (or a step limit /
malformed prediction stops the loop early).

Every step is recorded in the returned LLRResult.steps so a caller (e.g. the
Fase B.7 interactive demo) can animate the reduction, not just report the
final answer.
"""

from dataclasses import dataclass, field

import torch

from depth_lab.models.locator import Locator, predict_span
from depth_lab.models.replacer import Replacer
from depth_lab.tokenizer import CharTokenizer

DEFAULT_MAX_ITERATIONS = 40  # generous: real traces need one step per parenthesis pair


@dataclass(frozen=True)
class LLRStep:
    expr: str                # the full expression string before this step
    span: tuple[int, int]    # the span the locator pointed at, in `expr`
    span_text: str           # expr[span[0]:span[1]]
    predicted_value: str     # raw decoded replacer output ("True"/"False", or malformed text)


@dataclass(frozen=True)
class LLRResult:
    final_expr: str
    steps: list[LLRStep] = field(default_factory=list)
    converged: bool = False  # True iff final_expr is a bare "True"/"False" literal


@torch.no_grad()
def _replace_value(replacer: Replacer, tokenizer: CharTokenizer, span_text: str, device: str,
                    max_new_tokens: int = 6) -> str:
    reversed_span = span_text[::-1]
    prompt_ids = [tokenizer.bos_id, *tokenizer.encode(f"{reversed_span} => ")]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    replacer.eval()
    out_ids: list[int] = []
    for grown in replacer.generate_stream(idx, max_new_tokens, temperature=1e-6):
        next_id = grown[0, -1].item()
        if next_id == tokenizer.eos_id:
            break
        out_ids.append(next_id)
    replacer.train()
    return tokenizer.decode(out_ids)


def reduce_with_llr(locator: Locator, replacer: Replacer, tokenizer: CharTokenizer, expr: str, device: str,
                     max_iterations: int = DEFAULT_MAX_ITERATIONS) -> LLRResult:
    steps: list[LLRStep] = []
    current = expr

    for _ in range(max_iterations):
        if current in ("True", "False"):
            return LLRResult(final_expr=current, steps=steps, converged=True)
        if "(" not in current:
            # no more reducible groups, but not a bare literal -- something upstream went wrong
            return LLRResult(final_expr=current, steps=steps, converged=False)

        start, end = predict_span(locator, tokenizer, current, device)
        span_text = current[start:end]
        predicted_value = _replace_value(replacer, tokenizer, span_text, device)
        steps.append(LLRStep(expr=current, span=(start, end), span_text=span_text, predicted_value=predicted_value))

        if predicted_value not in ("True", "False"):
            # can't safely splice a malformed prediction back into the string
            return LLRResult(final_expr=current, steps=steps, converged=False)

        current = current[:start] + predicted_value + current[end:]

    return LLRResult(final_expr=current, steps=steps, converged=current in ("True", "False"))


def evaluate_exact_match(locator: Locator, replacer: Replacer, tokenizer: CharTokenizer, examples: list[dict],
                          device: str, max_iterations: int = DEFAULT_MAX_ITERATIONS) -> float:
    correct = 0
    for ex in examples:
        result = reduce_with_llr(locator, replacer, tokenizer, ex["expr"], device, max_iterations)
        if result.converged and result.final_expr == str(ex["value"]):
            correct += 1
    return correct / len(examples)
