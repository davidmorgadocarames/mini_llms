"""Builds the custom domain eval set (Fase C.6 item 2) consumed by
coconut_lab/eval/domain_eval.py. ~130 examples across six categories, each
with an explicit, automatically-checkable success criterion -- see
domain_eval.py's docstring for why checks are restricted to programmatic
types (no LLM judge in this project).

Arithmetic and unit-conversion answers are computed in Python (not typed by
hand) so they can't contain an arithmetic mistake. The other categories are
hand-curated common-knowledge facts, double-checked for accuracy while
writing them.

Usage:
    python -m coconut_lab.eval.build_domain_eval_set
"""

import json
import random
from pathlib import Path

from coconut_lab.data.prepare_instructions import _PROMPT_NO_INPUT

OUT_PATH = Path(__file__).resolve().parent / "domain_eval_set.jsonl"


def _prompt(instruction: str) -> str:
    return _PROMPT_NO_INPUT.format(instruction=instruction)


def _ex(id_: str, category: str, instruction: str, check: dict) -> dict:
    return {"id": id_, "category": category, "prompt": _prompt(instruction), "check": check}


def _build_arithmetic(rng: random.Random) -> list[dict]:
    examples = []
    templates = [
        ("plus", lambda a, b: a + b),
        ("minus", lambda a, b: a - b),
        ("times", lambda a, b: a * b),
    ]
    for i in range(18):
        a, b = rng.randint(2, 60), rng.randint(2, 60)
        word, fn = templates[i % len(templates)]
        if word == "minus" and b > a:
            a, b = b, a
        answer = fn(a, b)
        examples.append(_ex(f"arith_{i}", "arithmetic", f"What is {a} {word} {b}?",
                             {"type": "numeric_equals", "expected": answer}))

    for i in range(6):
        base = rng.choice([10, 20, 40, 50, 80, 100, 200])
        examples.append(_ex(f"arith_double_{i}", "arithmetic", f"What is double of {base}?",
                             {"type": "numeric_equals", "expected": base * 2}))
        half_base = base if base % 2 == 0 else base + 1
        examples.append(_ex(f"arith_half_{i}", "arithmetic", f"What is half of {half_base}?",
                             {"type": "numeric_equals", "expected": half_base / 2}))
    return examples


def _build_unit_conversion() -> list[dict]:
    conversions = [
        ("How many centimeters are in {v} meters?", "meters", 100, [2, 3, 5, 7]),
        ("How many meters are in {v} kilometers?", "kilometers", 1000, [2, 3, 4, 6]),
        ("How many minutes are in {v} hours?", "hours", 60, [2, 3, 4, 5]),
        ("How many seconds are in {v} minutes?", "minutes", 60, [2, 3, 4, 6]),
        ("How many grams are in {v} kilograms?", "kilograms", 1000, [2, 3, 5, 7]),
        ("How many items are in {v} dozen?", "dozen", 12, [2, 3, 4, 5]),
    ]
    examples = []
    for i, (template, _unit, factor, values) in enumerate(conversions):
        for j, v in enumerate(values):
            examples.append(_ex(f"unit_{i}_{j}", "unit_conversion", template.format(v=v),
                                 {"type": "numeric_equals", "expected": v * factor}))
    return examples


def _build_factual_qa() -> list[dict]:
    qa = [
        ("What is the capital of France?", ["paris"]),
        ("What is the capital of Japan?", ["tokyo"]),
        ("What is the capital of Italy?", ["rome"]),
        ("What is the capital of Spain?", ["madrid"]),
        ("What is the capital of Germany?", ["berlin"]),
        ("What is the capital of Canada?", ["ottawa"]),
        ("What is the capital of Australia?", ["canberra"]),
        ("What is the capital of Egypt?", ["cairo"]),
        ("What is the largest planet in the solar system?", ["jupiter"]),
        ("What is the smallest planet in the solar system?", ["mercury"]),
        ("What is the chemical symbol for gold?", ["au"]),
        ("How many continents are there on Earth?", ["seven", "7"]),
        ("How many days are there in a week?", ["seven", "7"]),
        ("How many months are there in a year?", ["twelve", "12"]),
        ("What color is the sky on a clear day?", ["blue"]),
        ("What is the freezing point of water in Celsius?", ["0", "zero"]),
        ("What is the boiling point of water in Celsius?", ["100", "hundred"]),
        ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is the longest river in the world?", ["nile", "amazon"]),
        ("What gas do humans need to breathe to survive?", ["oxygen"]),
        ("What is the closest star to Earth?", ["sun"]),
        ("What language is primarily spoken in Brazil?", ["portuguese"]),
        ("What is the largest ocean on Earth?", ["pacific"]),
        ("What is the tallest mountain on Earth?", ["everest"]),
    ]
    return [_ex(f"factual_{i}", "factual_qa", q, {"type": "contains_any", "keywords": kws})
            for i, (q, kws) in enumerate(qa)]


def _build_keyword_instruction() -> list[dict]:
    words = ["ocean", "mountain", "guitar", "garden", "bicycle", "library", "thunderstorm",
             "lighthouse", "volcano", "astronaut", "waterfall", "telescope", "bakery", "campfire",
             "harbor", "meadow"]
    examples = [
        _ex(f"keyword_word_{i}", "keyword_instruction", f"Write a sentence that includes the word '{w}'.",
            {"type": "contains_any", "keywords": [w]})
        for i, w in enumerate(words)
    ]

    open_set = [
        ("Name an animal that lives in the ocean.",
         ["fish", "whale", "dolphin", "shark", "octopus", "seal", "crab", "turtle", "jellyfish", "squid"]),
        ("Name a fruit that is yellow.", ["banana", "lemon", "pineapple", "mango", "yellow"]),
        ("Name a primary color.", ["red", "blue", "yellow"]),
        ("Name a country in Europe.",
         ["france", "germany", "spain", "italy", "portugal", "poland", "sweden", "norway", "greece",
          "netherlands", "belgium", "austria", "ireland", "finland", "denmark"]),
        ("Name a planet in our solar system.",
         ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]),
        ("Name a sport played with a ball.",
         ["soccer", "football", "basketball", "baseball", "tennis", "volleyball", "golf", "cricket"]),
        ("Name a musical instrument.",
         ["guitar", "piano", "violin", "drum", "flute", "trumpet", "saxophone", "cello"]),
        ("Name a season of the year.", ["spring", "summer", "autumn", "fall", "winter"]),
    ]
    examples += [_ex(f"keyword_open_{i}", "keyword_instruction", q, {"type": "contains_any", "keywords": kws})
                 for i, (q, kws) in enumerate(open_set)]
    return examples


def _build_classification() -> list[dict]:
    items = [
        ("Is the number 8 even or odd? Answer with one word.", ["even"]),
        ("Is the number 15 even or odd? Answer with one word.", ["odd"]),
        ("Is a whale a mammal or a fish? Answer with one word.", ["mammal"]),
        ("Is a tomato a fruit or a vegetable, botanically speaking? Answer with one word.", ["fruit"]),
        ("Is Mount Everest the tallest mountain on Earth? Answer yes or no.", ["yes"]),
        ("Is the Earth flat? Answer yes or no.", ["no"]),
        ("Is 17 a prime number? Answer yes or no.", ["yes"]),
        ("Is 21 a prime number? Answer yes or no.", ["no"]),
        ("Does ice float on water? Answer yes or no.", ["yes"]),
        ("Is Antarctica located in the Northern Hemisphere? Answer yes or no.", ["no"]),
        ("Is the sun a star or a planet? Answer with one word.", ["star"]),
        ("Is a spider an insect or an arachnid? Answer with one word.", ["arachnid"]),
        ("Is 100 Celsius the boiling point of water at sea level? Answer yes or no.", ["yes"]),
        ("Is the chemical symbol for oxygen 'O'? Answer yes or no.", ["yes"]),
    ]
    return [_ex(f"classify_{i}", "classification", q, {"type": "contains_any", "keywords": kws})
            for i, (q, kws) in enumerate(items)]


def _build_format_following() -> list[dict]:
    single_word = {"type": "regex", "pattern": r"^\s*\S+\s*$"}
    items = [
        ("List three colors, separated by commas.", {"type": "regex", "pattern": r",.*,"}),
        ("List two animals, separated by commas.", {"type": "regex", "pattern": r","}),
        ("Give a numbered list of two fruits.", {"type": "regex", "pattern": r"1[.)][\s\S]*2[.)]"}),
        ("Give a numbered list of three hobbies.", {"type": "regex", "pattern": r"1[.)][\s\S]*2[.)][\s\S]*3[.)]"}),
        ("Is water a liquid or a solid at room temperature? Answer with exactly one word.",
         {"type": "all_of", "checks": [single_word, {"type": "contains_any", "keywords": ["liquid"]}]}),
        ("What is the opposite of 'hot'? Answer with exactly one word.",
         {"type": "all_of", "checks": [single_word, {"type": "contains_any", "keywords": ["cold"]}]}),
        ("What is the opposite of 'up'? Answer with exactly one word.",
         {"type": "all_of", "checks": [single_word, {"type": "contains_any", "keywords": ["down"]}]}),
        ("What is the plural of 'child'? Answer with exactly one word.",
         {"type": "all_of", "checks": [single_word, {"type": "contains_any", "keywords": ["children"]}]}),
        ("Answer only with 'true' or 'false': the Great Wall of China is visible from space with the naked eye.",
         {"type": "contains_any", "keywords": ["false"]}),
        ("Answer only with 'true' or 'false': humans have walked on the Moon.",
         {"type": "contains_any", "keywords": ["true"]}),
        ("List the first three even numbers, separated by commas.",
         {"type": "all_of", "checks": [{"type": "regex", "pattern": ","},
                                        {"type": "contains_all", "keywords": ["2", "4", "6"]}]}),
        ("List the first three odd numbers, separated by commas.",
         {"type": "all_of", "checks": [{"type": "regex", "pattern": ","},
                                        {"type": "contains_all", "keywords": ["1", "3", "5"]}]}),
        ("Give your answer as a single word in all lowercase: what is 5 minus 5?",
         {"type": "all_of", "checks": [{"type": "regex", "pattern": r"^[a-z]+$"},
                                        {"type": "contains_any", "keywords": ["zero"]}]}),
    ]
    return [_ex(f"format_{i}", "format_following", q, check) for i, (q, check) in enumerate(items)]


def _build_topicality() -> list[dict]:
    items = [
        ("Explain in one sentence why the sky appears blue.", ["scatter", "light", "wavelength", "blue"]),
        ("Explain in one sentence what photosynthesis is.",
         ["plant", "sunlight", "light", "energy", "oxygen", "carbon dioxide", "glucose"]),
        ("Briefly explain what gravity does.", ["pull", "attract", "force", "down", "mass", "weight"]),
        ("Briefly explain what the water cycle is.",
         ["evaporat", "rain", "condens", "cloud", "precipitat", "water"]),
        ("Briefly explain why we have seasons.", ["tilt", "axis", "orbit", "sun", "earth"]),
        ("Briefly explain what a computer's CPU does.", ["process", "instruction", "calculat", "compute", "cpu"]),
        ("Briefly explain what recycling is.", ["reuse", "waste", "material", "reduce", "recycl", "environment"]),
        ("Briefly explain what exercise does for the body.",
         ["health", "muscle", "heart", "fitness", "strength", "body"]),
        ("Briefly explain what a democracy is.", ["vote", "elect", "citizen", "government", "people"]),
        ("Briefly explain what the internet is.", ["network", "computer", "connect", "web", "information"]),
        ("Briefly explain what a solar eclipse is.", ["moon", "sun", "shadow", "block", "eclipse"]),
        ("Briefly explain what bacteria are.", ["microorganism", "cell", "organism", "microbe", "germ"]),
    ]
    return [_ex(f"topic_{i}", "topicality", q, {"type": "contains_any", "keywords": kws})
            for i, (q, kws) in enumerate(items)]


def build(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    examples = (
        _build_arithmetic(rng)
        + _build_unit_conversion()
        + _build_factual_qa()
        + _build_keyword_instruction()
        + _build_classification()
        + _build_format_following()
        + _build_topicality()
    )
    return examples


def main() -> None:
    examples = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    by_category: dict[str, int] = {}
    for ex in examples:
        by_category[ex["category"]] = by_category.get(ex["category"], 0) + 1
    print(f"{len(examples)} examples -> {OUT_PATH}")
    for cat, n in sorted(by_category.items()):
        print(f"  {cat:20s} {n}")


if __name__ == "__main__":
    main()
