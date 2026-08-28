"""Fase C.6 item 2: custom domain eval set with explicit, programmatically-
checkable success criteria -- the piece the user singled out as most
important for deciding whether Coconut "actually works," beyond any
benchmark number ("ninguno de estos te dirá si el modelo sirve").

Checks are restricted to types a script can verify unambiguously
(contains_any/contains_all/not_contains/regex/numeric_equals, composed with
all_of/any_of): there's no LLM-judge infrastructure in this project and
manually grading ~1000+ generations isn't feasible, so subjective categories
(creative-writing quality, free-form summarization) are out of scope --
noted honestly here rather than faked with a shaky heuristic.

Methodology (per the user's own spec): fixed prompt template and fixed
decoding temperature/top_k, 3 sampling seeds. For each seed, the *entire*
eval set is scored once (same seed applied before every generation), giving
one whole-set accuracy per seed; the reported number is mean +- std across
those 3 seed-level accuracies -- not per-example variance -- so it reads as
a single comparable "42% +- 3%" per architecture, matching how the plan
describes it ("corre 3 semillas para reportar varianza").
"""

import re

import torch

from coconut_lab.models import cracked as cracked_mod
from coconut_lab.models import sliced as sliced_mod
from coconut_lab.models.pressed_loop import reduce_with_pressed

_NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _extract_last_number(text: str) -> str | None:
    numbers = _NUMBER_PATTERN.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def check_response(response: str, check: dict) -> bool:
    kind = check["type"]
    if kind == "all_of":
        return all(check_response(response, c) for c in check["checks"])
    if kind == "any_of":
        return any(check_response(response, c) for c in check["checks"])

    response = response.strip()
    if not response:
        return False

    if kind == "contains_any":
        low = response.lower()
        return any(kw.lower() in low for kw in check["keywords"])
    if kind == "contains_all":
        low = response.lower()
        return all(kw.lower() in low for kw in check["keywords"])
    if kind == "not_contains":
        low = response.lower()
        return not any(kw.lower() in low for kw in check["keywords"])
    if kind == "regex":
        # No IGNORECASE: some patterns deliberately test case (e.g. an
        # "answer in all lowercase" format check), so case-folding here
        # would silently defeat them.
        return re.search(check["pattern"], response) is not None
    if kind == "numeric_equals":
        predicted = _extract_last_number(response)
        if predicted is None:
            return False
        try:
            return abs(float(predicted) - float(check["expected"])) <= check.get("tolerance", 1e-6)
        except ValueError:
            return False
    raise ValueError(f"unknown check type: {kind!r}")


@torch.no_grad()
def _generate_all(models: dict, tokenizer, prompt: str, device: str, max_new_tokens: int,
                   temperature: float, top_k: int) -> dict[str, str]:
    cracked, sliced, (drafter, locator, replacer) = models["cracked"], models["sliced"], models["pressed"]
    return {
        "cracked": cracked_mod.generate_response(cracked, tokenizer, prompt, device,
                                                   max_new_tokens=max_new_tokens,
                                                   temperature=temperature, top_k=top_k),
        "sliced": sliced_mod.generate_response(sliced, tokenizer, prompt, device,
                                                max_new_tokens=max_new_tokens,
                                                temperature=temperature, top_k=top_k),
        "pressed": reduce_with_pressed(drafter, locator, replacer, tokenizer, prompt, device,
                                        max_new_tokens_draft=max_new_tokens).final_text,
    }


def run_domain_eval(models: dict, tokenizer, examples: list[dict], device: str,
                     seeds: tuple[int, ...] = (0, 1, 2), temperature: float = 0.8, top_k: int = 50,
                     max_new_tokens: int = 80) -> dict:
    architectures = ("cracked", "sliced", "pressed")
    seed_accuracy: dict[str, list[float]] = {a: [] for a in architectures}
    per_example_pass: dict[str, dict[str, list[bool]]] = {a: {} for a in architectures}

    for seed in seeds:
        seed_correct = {a: 0 for a in architectures}
        for ex in examples:
            torch.manual_seed(seed)
            responses = _generate_all(models, tokenizer, ex["prompt"], device, max_new_tokens, temperature, top_k)
            for arch in architectures:
                ok = check_response(responses[arch], ex["check"])
                seed_correct[arch] += ok
                per_example_pass[arch].setdefault(ex["id"], []).append(ok)

        for arch in architectures:
            acc = seed_correct[arch] / len(examples)
            seed_accuracy[arch].append(acc)
        print(f"seed {seed}: " + " | ".join(f"{a} {seed_accuracy[a][-1]:.3f}" for a in architectures))

    overall = {}
    for arch in architectures:
        accs = seed_accuracy[arch]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        overall[arch] = {"seed_accuracies": accs, "mean": mean, "std": std}

    by_category: dict[str, dict[str, float]] = {a: {} for a in architectures}
    for arch in architectures:
        cats: dict[str, list[bool]] = {}
        for ex in examples:
            cats.setdefault(ex["category"], []).extend(per_example_pass[arch][ex["id"]])
        by_category[arch] = {cat: sum(v) / len(v) for cat, v in cats.items()}

    return {"overall": overall, "by_category": by_category, "n_examples": len(examples), "seeds": list(seeds)}
