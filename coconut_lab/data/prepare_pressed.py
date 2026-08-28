"""Fase C.5 ("Pressed"): builds locator/replacer training supervision from
GSM8K's step annotations (coconut_lab.data.prepare_reasoning), matching Fase
B's exact "locate the first remaining reducible span, resolve it, collapse
it to its bare value, repeat" mechanic -- just adapted so it also works on a
*drafter's own generated text*, not only on hand-written ground truth.

Important design correction made while building this: an earlier version of
this module redacted not-yet-resolved steps down to "<<expr=???>>???"
placeholders. That doesn't match reality: at inference time, Pressed's
locator/replacer run on a *drafter's own generated draft*, which always
already has a (possibly wrong) number in every <<expr=result>>result slot --
never a literal "???". Training the locator to look for "???" would make it
useless on real drafts. The fix: only the *already-resolved* steps are
collapsed down to their bare (correct) value, mimicking exactly how Fase B
physically replaced a resolved paren-group with its value and shrank the
string; steps at and after the current one are left completely untouched,
in whatever form they're already in (ground-truth text during training,
the drafter's own guess at inference time) -- so the locator learns "find
the first remaining <<expr=result>>result span", a task equally well-posed
on hand-written or model-generated text.
"""

import re

from coconut_lab.data.prepare_reasoning import extract_steps

_TRAILING_NUMBER = re.compile(r"^-?\d+(?:,\d{3})*(?:\.\d+)?")


def _full_step_span(text: str, step: dict) -> tuple[int, int]:
    """Extends step["span"] (just the <<expr=result>> annotation) to also
    cover the plain-text repetition of the result immediately following it,
    if any (e.g. "<<48/2=24>>24" -- both must move/collapse together)."""
    start, end = step["span"]
    m = _TRAILING_NUMBER.match(text[end:])
    if m:
        end += m.end()
    return start, end


def resolve_up_to(text: str, steps: list[dict], up_to_index: int) -> tuple[str, tuple[int, int] | None]:
    """Simulates the state after an LLR loop has already resolved
    steps[:up_to_index]: each of those steps' full <<expr=result>>result
    annotation is collapsed down to just its bare (correct) result value --
    shrinking the string, exactly like Fase B did. steps[up_to_index:] are
    left untouched. Returns (text_at_this_state, span_of_step[up_to_index]
    in that text) for locator supervision, or span=None if up_to_index is
    past the last step (nothing left to resolve -- the loop's stopping
    state)."""
    text_state = text
    shift = 0
    for j in range(up_to_index):
        step = steps[j]
        start, end = _full_step_span(text, step)
        start, end = start + shift, end + shift
        replacement = step["result"]
        text_state = text_state[:start] + replacement + text_state[end:]
        shift += len(replacement) - (end - start)

    if up_to_index >= len(steps):
        return text_state, None

    start, end = _full_step_span(text, steps[up_to_index])
    start, end = start + shift, end + shift
    return text_state, (start, end)


def build_locator_examples(examples: list[dict]) -> list[dict]:
    """One instance per calculation step: {"text": <state after resolving
    steps 0..i-1>, "span": (start, end) of step i's still-unresolved
    <<expr=result>>result span}."""
    out = []
    for ex in examples:
        for i in range(len(ex["steps"])):
            text_state, span = resolve_up_to(ex["answer_text"], ex["steps"], i)
            if span is not None:
                out.append({"text": text_state, "span": list(span)})
    return out


def build_replacer_examples(examples: list[dict]) -> list[dict]:
    """One instance per calculation step: {"prompt": <reversed expr> " => ",
    "response": result} -- same shape InstructionDataset already expects, so
    it can be reused directly instead of writing a new dataset class (same
    "reverse the span text" trick Fase B's replacer used). Always built from
    the real GSM8K expr/result, independent of resolution order/state --
    this task (isolated expr -> its value) doesn't depend on surrounding
    context at all."""
    out = []
    for ex in examples:
        for step in ex["steps"]:
            out.append({"prompt": f"{step['expr'][::-1]} => ", "response": str(step["result"])})
    return out
