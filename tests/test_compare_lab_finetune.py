"""Fine-tuning: loss only on assistant tokens (incl <eos>), and the two
architectures get IDENTICAL example sets so the comparison is fair."""

from compare_lab.train.finetune import (build_cracked_examples, build_sliced_examples,
                                        iter_assistant_turns, collate_cracked, collate_sliced)


CONVS = [
    {"messages": [
        {"role": "user", "content": "hello there friend"},
        {"role": "assistant", "content": "general kenobi you are a bold one"},
    ]},
    {"messages": [
        {"role": "user", "content": "count please"},
        {"role": "assistant", "content": "one two three"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "four five six"},
    ]},
]
# 1 assistant turn in conv 0 + 2 in conv 1
N_TURNS = 3


def test_one_example_per_assistant_turn(tiny_tokenizer):
    """Both builders emit one example per assistant turn -- NOT one per
    conversation for the decoder-only, which would give it more gradient over
    assistant tokens than Sliced-D at the same number of steps."""
    tok = tiny_tokenizer
    cracked = build_cracked_examples(CONVS, tok, block_size=256)
    sliced = build_sliced_examples(CONVS, tok, block_size=256)
    assert len(cracked) == len(sliced) == N_TURNS


def test_cracked_example_masks_only_assistant_with_eos(tiny_tokenizer):
    tok = tiny_tokenizer
    ids, mask = build_cracked_examples(CONVS, tok, block_size=256)[0]
    response = tok.encode("general kenobi you are a bold one") + [tok.eos_id]
    assert sum(mask) == len(response)
    # the masked-in tail is exactly the response plus its <eos>
    assert [t for t, m in zip(ids, mask) if m] == response
    assert ids[-1] == tok.eos_id and mask[-1] == 1
    # nothing before the response is in the loss (user/system/history)
    assert sum(mask[:len(ids) - len(response)]) == 0


def test_sliced_example_encoder_ends_at_assistant_marker(tiny_tokenizer):
    tok = tiny_tokenizer
    for src_ids, resp in build_sliced_examples(CONVS, tok, block_size=256):
        assert src_ids[-1] == tok.assistant_id
        assert resp[-1] == tok.eos_id


def test_both_architectures_cover_exactly_the_same_assistant_tokens(tiny_tokenizer):
    """The fairness property: same examples, same order, same loss tokens."""
    tok = tiny_tokenizer
    cracked = build_cracked_examples(CONVS, tok, block_size=256)
    sliced = build_sliced_examples(CONVS, tok, block_size=256)
    for (ids, mask), (src_ids, resp) in zip(cracked, sliced):
        cracked_loss_tokens = [t for t, m in zip(ids, mask) if m]
        assert cracked_loss_tokens == resp          # identical target tokens
        assert ids[:len(src_ids)] == src_ids        # identical history/prefix
    assert sum(sum(m) for _, m in cracked) == sum(len(r) for _, r in sliced)


def test_turns_too_long_are_discarded_not_truncated(tiny_tokenizer):
    tok = tiny_tokenizer
    long_conv = [{"messages": [
        {"role": "user", "content": "word " * 400},
        {"role": "assistant", "content": "reply " * 400},
    ]}]
    assert build_cracked_examples(long_conv, tok, block_size=64) == []
    assert build_sliced_examples(long_conv, tok, block_size=64) == []
    # and the shared source of truth agrees
    assert list(iter_assistant_turns(long_conv, tok, block_size=64)) == []


def test_collate_shapes(tiny_tokenizer):
    tok = tiny_tokenizer
    cracked = build_cracked_examples(CONVS, tok, block_size=256)
    x, y, lm = collate_cracked(cracked, tok.pad_id, device="cpu")
    assert x.shape == y.shape == lm.shape
    assert x.shape[0] == N_TURNS

    sliced = build_sliced_examples(CONVS, tok, block_size=256)
    src, src_pad, tin, tout, lm_s = collate_sliced(sliced, tok.bos_id, tok.pad_id, device="cpu")
    assert src.shape[0] == tin.shape[0] == N_TURNS
    assert tin.shape == tout.shape == lm_s.shape
    # padding mask built from lengths: the real prefix is never masked as padding
    for i, (src_ids, _) in enumerate(sliced):
        assert not src_pad[i, :len(src_ids)].any()
        assert tin[i, 0].item() == tok.bos_id
