import torch

from depth_lab.data.loader import ExprDataset, format_target_text
from depth_lab.tokenizer import CharTokenizer


def test_format_target_text_renders_python_bool_as_capitalized_word():
    assert format_target_text("(True and False)", True) == "(True and False) => True"
    assert format_target_text("(True and False)", False) == "(True and False) => False"


def test_dataset_item_shapes_match_block_size():
    tok = CharTokenizer()
    examples = [{"expr": "True", "value": True, "depth": 0}]
    ds = ExprDataset(examples, tok, block_size=16)
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.long


def test_dataset_x_y_are_shifted_by_one():
    tok = CharTokenizer()
    examples = [{"expr": "True", "value": False, "depth": 0}]
    ds = ExprDataset(examples, tok, block_size=16)
    x, y = ds[0]
    assert x[1:].tolist() == y[:-1].tolist()


def test_dataset_raises_when_example_exceeds_block_size():
    tok = CharTokenizer()
    examples = [{"expr": "(True and False)", "value": True, "depth": 1}]
    ds = ExprDataset(examples, tok, block_size=4)
    try:
        ds[0]
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for oversized example")


def test_prompt_ids_stop_right_after_the_arrow():
    tok = CharTokenizer()
    examples = [{"expr": "True", "value": False, "depth": 0}]
    ds = ExprDataset(examples, tok, block_size=16)
    prompt = ds.prompt_ids(0)
    assert tok.decode(prompt) == "True => "
