"""Step-by-step reduction of boolean-domain expressions produced by
depth_lab.data.generator -- the ground truth that Fase B.5's Looped
Locate-and-Replace pipeline is trying to learn to imitate: at each step,
locate the single innermost parenthesized sub-expression, replace it with its
value, and repeat until nothing but a literal remains.

Deliberately hand-written instead of calling Python's eval() (unlike the test
suite, which uses eval() purely as an independent oracle) -- the grammar is
tiny (True/False/and/or/not/parens) and this module's whole purpose *is* the
evaluation logic, so writing it by hand keeps it in the same "reinvent the
core logic, don't hide it behind a library" spirit as the rest of the
project, and sidesteps eval() entirely rather than arguing it's safe here.

Only supports the boolean domain: the arithmetic domain's numeric literals
would need a different atomic evaluator, and Fase B's approved scope is
bool-only (see the Fase B plan).
"""

from dataclasses import dataclass

_BOOL_LITERALS = {"True": True, "False": False}


@dataclass(frozen=True)
class ReduceStep:
    expr: str                  # the expression string before this step
    span: tuple[int, int]      # (start, end) of the sub-expression replaced, in `expr`
    span_text: str             # expr[span[0]:span[1]] -- e.g. "(True and False)" or "not (True)"
    value: bool                # the value that span reduced to
    new_expr: str               # `expr` with `span` replaced by str(value)


def _eval_atomic(content: str) -> bool:
    """content is always either a single literal or "left op right" -- by the
    time reduce_step calls this, the innermost-paren search already
    guarantees content is fully reduced-or-literal (see find_innermost_span)."""
    tokens = content.split()
    if len(tokens) == 1:
        return _BOOL_LITERALS[tokens[0]]
    left, op, right = tokens
    left_v, right_v = _BOOL_LITERALS[left], _BOOL_LITERALS[right]
    return (left_v and right_v) if op == "and" else (left_v or right_v)


def find_innermost_span(expr: str) -> tuple[int, int]:
    """The first ")" encountered scanning left-to-right always closes an
    innermost group (a well-formed string can't close an outer paren before
    all of its nested ones), so this never needs to search for nesting
    depth explicitly. If that group is directly governed by a unary "not"
    (i.e. immediately preceded by "not "), the span is extended to include
    it, since "not (...)" always reduces as one atomic step."""
    stack: list[int] = []
    for i, ch in enumerate(expr):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            start = stack.pop()
            end = i + 1
            if expr[max(0, start - 4):start] == "not ":
                start -= 4
            return start, end
    raise ValueError(f"no parenthesized sub-expression in {expr!r}")


def reduce_step(expr: str) -> ReduceStep:
    start, end = find_innermost_span(expr)
    span_text = expr[start:end]

    if span_text.startswith("not ("):
        content = span_text[len("not ("):-1]
        value = not _eval_atomic(content)
    else:
        content = span_text[1:-1]
        value = _eval_atomic(content)

    new_expr = expr[:start] + str(value) + expr[end:]
    return ReduceStep(expr=expr, span=(start, end), span_text=span_text, value=value, new_expr=new_expr)


def reduction_trace(expr: str) -> list[ReduceStep]:
    """The full sequence of reduce_step calls needed to bring `expr` down to
    a bare literal. Empty for a depth-0 expression (no parens at all)."""
    trace = []
    current = expr
    while "(" in current:
        step = reduce_step(current)
        trace.append(step)
        current = step.new_expr
    return trace


def evaluate(expr: str) -> bool:
    """The final value `expr` reduces to."""
    if "(" not in expr:
        return _BOOL_LITERALS[expr]
    return reduction_trace(expr)[-1].value
