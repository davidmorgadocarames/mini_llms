"""Synthetic generator of nested, Python-flavored arithmetic/boolean expressions
with a controllable recursion depth — the Fase B counterpart of the abstract
postfix expressions used in Zhiyuan He's "Exploring Depth Generalization..."
(AAAI 2026), themed as recognizable code instead of bare symbols.

Depth is defined exactly as in that paper (their Figure 1): the length of the
longest root-to-leaf path in the expression's parse tree — not the total number
of operators (that's "length", which is left free to vary here on purpose, since
the paper shows depth, not length, is what breaks Transformers).

We never call Python's `eval()` on generated text: the value is computed
directly while the tree is built, so correctness doesn't depend on parsing our
own string output back. (`eval()` is only used in tests, as an independent
oracle to confirm the rendered string matches Python's own semantics.)
"""

import random
from dataclasses import dataclass

BOOL_BINARY_OPS = ("and", "or")
ARITH_BINARY_OPS = ("+", "-", "*")
UNARY_NOT_PROB = 0.2


@dataclass(frozen=True)
class Example:
    expr: str
    value: bool | int
    depth: int


def _gen_bool(depth: int, rng: random.Random) -> tuple[str, bool]:
    if depth == 0:
        v = rng.choice([True, False])
        return str(v), v

    if rng.random() < UNARY_NOT_PROB:
        sub_expr, sub_val = _gen_bool(depth - 1, rng)
        return f"not ({sub_expr})", not sub_val

    op = rng.choice(BOOL_BINARY_OPS)
    other_depth = rng.randint(0, depth - 1)
    deep_expr, deep_val = _gen_bool(depth - 1, rng)
    other_expr, other_val = _gen_bool(other_depth, rng)

    if rng.choice([True, False]):
        left_expr, left_val, right_expr, right_val = deep_expr, deep_val, other_expr, other_val
    else:
        left_expr, left_val, right_expr, right_val = other_expr, other_val, deep_expr, deep_val

    value = (left_val and right_val) if op == "and" else (left_val or right_val)
    return f"({left_expr} {op} {right_expr})", value


def _gen_arith(depth: int, rng: random.Random) -> tuple[str, int]:
    if depth == 0:
        v = rng.randint(0, 999)
        return str(v), v

    op = rng.choice(ARITH_BINARY_OPS)
    other_depth = rng.randint(0, depth - 1)
    deep_expr, deep_val = _gen_arith(depth - 1, rng)
    other_expr, other_val = _gen_arith(other_depth, rng)

    if rng.choice([True, False]):
        left_expr, left_val, right_expr, right_val = deep_expr, deep_val, other_expr, other_val
    else:
        left_expr, left_val, right_expr, right_val = other_expr, other_val, deep_expr, deep_val

    raw = {"+": left_val + right_val, "-": left_val - right_val, "*": left_val * right_val}[op]
    value = raw % 1000  # truncate, same as the paper: keeps the task about depth, not digit growth
    return f"({left_expr} {op} {right_expr})", value


def generate(domain: str, depth: int, rng: random.Random | None = None) -> Example:
    """domain: 'bool' or 'arith'. depth: exact longest-path depth of the tree
    (depth=0 yields a bare literal, no operators)."""
    rng = rng or random.Random()
    if domain == "bool":
        expr, value = _gen_bool(depth, rng)
    elif domain == "arith":
        expr, value = _gen_arith(depth, rng)
    else:
        raise ValueError(f"unknown domain: {domain!r}")
    return Example(expr=expr, value=value, depth=depth)


def generate_dataset(domain: str, depths: range | list[int], n_per_depth: int,
                      seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    return [generate(domain, d, rng) for d in depths for _ in range(n_per_depth)]
