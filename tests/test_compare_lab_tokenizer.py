"""Tokenizer: distinct special ids and correct chat masking (the Fase C fixes)."""


def test_special_ids_are_distinct_and_ordered(tiny_tokenizer):
    tok = tiny_tokenizer
    ids = [tok.pad_id, tok.bos_id, tok.eos_id, tok.system_id, tok.user_id, tok.assistant_id]
    assert ids == [0, 1, 2, 3, 4, 5]
    assert len(set(ids)) == 6
    # the exact Fase C bug: eos must NOT equal pad
    assert tok.eos_id != tok.pad_id
    assert tok.bos_id != tok.pad_id


def test_raw_text_never_emits_special_ids(tiny_tokenizer):
    tok = tiny_tokenizer
    # even literal marker strings in raw text must not collapse to the reserved ids
    enc = tok.encode("The <pad> <bos> <eos> <|user|> fox and dog")
    assert all(i >= 6 for i in enc)


def test_build_chat_masks_only_assistant_including_eos(tiny_tokenizer):
    tok = tiny_tokenizer
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ids, mask = tok.build_chat(msgs)
    assert len(ids) == len(mask)
    # <bos> is position 0 and must never be masked in
    assert ids[0] == tok.bos_id and mask[0] == 0
    # last token is the assistant-closing <eos>, and it IS in the loss
    assert ids[-1] == tok.eos_id and mask[-1] == 1
    # exactly the assistant content tokens + its eos are masked in
    assert sum(mask) == len(tok.encode("Hello!")) + 1
    # no user/system token is in the loss
    for i, m in zip(ids, mask):
        if i in (tok.user_id, tok.system_id):
            assert m == 0


def test_generation_prompt_ends_with_assistant_marker(tiny_tokenizer):
    tok = tiny_tokenizer
    msgs = [{"role": "user", "content": "Question?"}]
    ids, mask = tok.build_chat(msgs, add_generation_prompt=True)
    assert ids[-1] == tok.assistant_id
    assert sum(mask) == 0  # nothing to train on in a generation prompt


def test_multi_turn_masks_every_assistant_turn(tiny_tokenizer):
    tok = tiny_tokenizer
    msgs = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second answer"},
    ]
    ids, mask = tok.build_chat(msgs)
    expected = (len(tok.encode("first answer")) + 1) + (len(tok.encode("second answer")) + 1)
    assert sum(mask) == expected
