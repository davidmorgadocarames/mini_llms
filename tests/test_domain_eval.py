from coconut_lab.eval.build_domain_eval_set import build
from coconut_lab.eval.domain_eval import check_response


def test_contains_any_matches_case_insensitively():
    check = {"type": "contains_any", "keywords": ["Paris"]}
    assert check_response("The capital is paris.", check)
    assert not check_response("The capital is Rome.", check)


def test_contains_all_requires_every_keyword():
    check = {"type": "contains_all", "keywords": ["2", "4", "6"]}
    assert check_response("The numbers are 2, 4, 6.", check)
    assert not check_response("The numbers are 2, 4.", check)


def test_not_contains_fails_when_keyword_present():
    check = {"type": "not_contains", "keywords": ["sorry"]}
    assert check_response("Here is the answer.", check)
    assert not check_response("Sorry, I cannot help.", check)


def test_regex_check():
    check = {"type": "regex", "pattern": r"^\s*\S+\s*$"}
    assert check_response("liquid", check)
    assert not check_response("it is a liquid", check)


def test_numeric_equals_extracts_last_number_and_tolerates_formatting():
    check = {"type": "numeric_equals", "expected": 42}
    assert check_response("The answer is 42.", check)
    assert check_response("42", check)
    assert not check_response("The answer is 43.", check)


def test_numeric_equals_returns_false_when_no_number_present():
    check = {"type": "numeric_equals", "expected": 42}
    assert not check_response("I don't know.", check)


def test_all_of_requires_every_subcheck():
    check = {"type": "all_of", "checks": [
        {"type": "regex", "pattern": r"^[a-z]+$"},
        {"type": "contains_any", "keywords": ["zero"]},
    ]}
    assert check_response("zero", check)
    assert not check_response("Zero", check)  # fails the all-lowercase regex
    assert not check_response("empty", check)  # fails the keyword


def test_any_of_requires_at_least_one_subcheck():
    check = {"type": "any_of", "checks": [
        {"type": "contains_any", "keywords": ["nile"]},
        {"type": "contains_any", "keywords": ["amazon"]},
    ]}
    assert check_response("The river is the Amazon.", check)
    assert not check_response("The river is the Mississippi.", check)


def test_empty_response_always_fails_leaf_checks():
    for check in [
        {"type": "contains_any", "keywords": ["x"]},
        {"type": "regex", "pattern": r".*"},
        {"type": "numeric_equals", "expected": 0},
    ]:
        assert not check_response("   ", check)


def test_build_produces_a_set_within_the_planned_size_range_with_unique_ids():
    examples = build()
    assert 100 <= len(examples) <= 300
    ids = [ex["id"] for ex in examples]
    assert len(ids) == len(set(ids))


def test_build_arithmetic_answers_are_internally_consistent_with_the_check():
    """Every arithmetic/unit-conversion example's own check must classify a
    response containing its expected value as correct -- an oracle check on
    the generator itself, same spirit as depth_lab's reduce() label tests."""
    examples = build()
    numeric = [ex for ex in examples if ex["category"] in ("arithmetic", "unit_conversion")]
    assert len(numeric) > 0
    for ex in numeric:
        expected = ex["check"]["expected"]
        assert check_response(f"The answer is {expected}.", ex["check"])


def test_build_every_example_has_a_well_formed_prompt_and_check():
    examples = build()
    valid_types = {"contains_any", "contains_all", "not_contains", "regex", "numeric_equals", "all_of", "any_of"}
    for ex in examples:
        assert ex["prompt"].strip()
        assert ex["check"]["type"] in valid_types
