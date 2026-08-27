import random

import pytest

from depth_lab.data.generator import generate, generate_dataset


def max_paren_depth(s: str) -> int:
    """Independent structural check: every recursive call in the generator
    wraps its subexpression in exactly one parenthesis pair, so the maximum
    paren-nesting level in the rendered string must equal the tree depth we
    asked for — regardless of how the generator computed the value."""
    depth = 0
    max_depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


@pytest.mark.parametrize("domain", ["bool", "arith"])
@pytest.mark.parametrize("depth", [0, 1, 2, 5, 10])
def test_rendered_depth_matches_requested_depth(domain, depth):
    rng = random.Random(42)
    for _ in range(20):
        example = generate(domain, depth, rng)
        assert max_paren_depth(example.expr) == depth
        assert example.depth == depth


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5, 8])
def test_bool_value_matches_python_eval(depth):
    rng = random.Random(7)
    for _ in range(30):
        example = generate("bool", depth, rng)
        assert eval(example.expr) == example.value  # noqa: S307 (safe: our own generated text)


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5, 8])
def test_arith_value_matches_python_eval_mod_1000(depth):
    rng = random.Random(11)
    for _ in range(30):
        example = generate("arith", depth, rng)
        assert eval(example.expr) % 1000 == example.value  # noqa: S307


def test_depth_zero_is_a_bare_literal():
    rng = random.Random(0)
    bool_example = generate("bool", 0, rng)
    assert bool_example.expr in ("True", "False")

    arith_example = generate("arith", 0, rng)
    assert arith_example.expr.isdigit()


def test_generate_dataset_covers_requested_depths_and_count():
    dataset = generate_dataset("bool", depths=range(0, 4), n_per_depth=5, seed=3)
    assert len(dataset) == 4 * 5
    depths_seen = sorted({ex.depth for ex in dataset})
    assert depths_seen == [0, 1, 2, 3]


def test_generate_dataset_is_deterministic_given_seed():
    a = generate_dataset("arith", depths=range(0, 5), n_per_depth=3, seed=123)
    b = generate_dataset("arith", depths=range(0, 5), n_per_depth=3, seed=123)
    assert [ex.expr for ex in a] == [ex.expr for ex in b]
    assert [ex.value for ex in a] == [ex.value for ex in b]
