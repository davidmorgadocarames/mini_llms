import random

import pytest

from depth_lab.data.generator import generate
from depth_lab.data.reduce import evaluate, find_innermost_span, reduce_step, reduction_trace


@pytest.mark.parametrize("depth", [0, 1, 2, 3, 5, 8])
def test_evaluate_matches_python_eval(depth):
    rng = random.Random(21)
    for _ in range(30):
        example = generate("bool", depth, rng)
        assert evaluate(example.expr) == eval(example.expr)  # noqa: S307 (safe: our own generated text)


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
def test_every_intermediate_step_stays_consistent_with_python_eval(depth):
    """Not just the final value -- every intermediate new_expr along the way
    must still evaluate (via Python's eval, as an independent oracle) to the
    same value as the original expression."""
    rng = random.Random(22)
    for _ in range(20):
        example = generate("bool", depth, rng)
        original_value = eval(example.expr)  # noqa: S307
        for step in reduction_trace(example.expr):
            assert eval(step.new_expr) == original_value  # noqa: S307


def test_reduction_trace_is_empty_for_a_bare_literal():
    assert reduction_trace("True") == []
    assert reduction_trace("False") == []
    assert evaluate("True") is True
    assert evaluate("False") is False


def test_find_innermost_span_picks_the_first_closing_group():
    expr = "((True and False) or (True and True))"
    start, end = find_innermost_span(expr)
    assert expr[start:end] == "(True and False)"


def test_find_innermost_span_extends_to_include_a_governing_not():
    expr = "not (True)"
    start, end = find_innermost_span(expr)
    assert expr[start:end] == "not (True)"


def test_reduce_step_handles_double_negation():
    step1 = reduce_step("not (not (True))")
    assert step1.span_text == "not (True)"
    assert step1.value is False
    assert step1.new_expr == "not (False)"

    step2 = reduce_step(step1.new_expr)
    assert step2.value is True
    assert step2.new_expr == "True"


def test_reduce_step_handles_not_wrapping_a_binary_group():
    step1 = reduce_step("not ((True and False))")
    assert step1.span_text == "(True and False)"
    assert step1.value is False
    assert step1.new_expr == "not (False)"

    step2 = reduce_step(step1.new_expr)
    assert step2.span_text == "not (False)"
    assert step2.value is True
    assert step2.new_expr == "True"


@pytest.mark.parametrize("depth", [1, 2, 3, 5, 8])
def test_reduction_trace_terminates_in_a_bare_literal(depth):
    rng = random.Random(23)
    for _ in range(20):
        example = generate("bool", depth, rng)
        trace = reduction_trace(example.expr)
        final = trace[-1].new_expr if trace else example.expr
        assert final in ("True", "False")
