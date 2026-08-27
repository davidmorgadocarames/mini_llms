import pytest

from depth_lab.tokenizer import CharTokenizer


def test_round_trip_encode_decode():
    tok = CharTokenizer()
    text = "(True and (False or not (True)))"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_round_trip_arithmetic_expression():
    tok = CharTokenizer()
    text = "(3 + (2 * (5 - 1)))"
    assert tok.decode(tok.encode(text)) == text


def test_each_character_maps_to_a_distinct_id():
    tok = CharTokenizer()
    ids = tok.encode("True")
    assert len(set(ids)) == len(set("True"))


def test_special_tokens_have_stable_ids_and_are_skipped_on_decode():
    tok = CharTokenizer()
    ids = [tok.bos_id, *tok.encode("True"), tok.eos_id, tok.pad_id]
    assert tok.decode(ids) == "True"
    assert tok.decode(ids, skip_special=False) == "<bos>True<eos><pad>"


def test_pad_bos_eos_are_distinct():
    tok = CharTokenizer()
    assert len({tok.pad_id, tok.bos_id, tok.eos_id}) == 3


def test_unknown_character_raises():
    tok = CharTokenizer()
    with pytest.raises(ValueError):
        tok.encode("日本語")


def test_vocab_size_is_small():
    tok = CharTokenizer()
    assert tok.vocab_size < 40


def test_arrow_separator_round_trips():
    tok = CharTokenizer()
    text = "(True and False) => False"
    assert tok.decode(tok.encode(text)) == text
