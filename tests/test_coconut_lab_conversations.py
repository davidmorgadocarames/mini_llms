import pytest
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import ConversationDataset
from coconut_lab.data.prepare_conversations import _build_trees, build, format_turns, load_jsonl
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def _row(message_id, parent_id, role, text, rank=None):
    return {"message_id": message_id, "parent_id": parent_id, "role": role, "text": text, "rank": rank}


def test_build_trees_follows_the_best_ranked_branch_at_each_split():
    rows = [
        _row("root", None, "prompter", "hi"),
        _row("reply-good", "root", "assistant", "good reply", rank=0),
        _row("reply-bad", "root", "assistant", "worse reply", rank=1),
        _row("followup-good", "reply-good", "prompter", "thanks, more?"),
        _row("final", "followup-good", "assistant", "sure, here's more", rank=0),
    ]
    paths = _build_trees(rows)
    assert len(paths) == 1
    texts = [r["text"] for r in paths[0]]
    assert texts == ["hi", "good reply", "thanks, more?", "sure, here's more"]


def test_build_trees_handles_a_single_message_with_no_replies():
    rows = [_row("root", None, "prompter", "hello?")]
    paths = _build_trees(rows)
    assert len(paths) == 1
    assert [r["text"] for r in paths[0]] == ["hello?"]


def test_format_turns_uses_role_markers_in_order():
    turns = [{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Hello!"}]
    rendered = format_turns(turns)
    assert rendered.index("<|user|>") < rendered.index("Hi") < rendered.index("<|assistant|>") < rendered.index("Hello!")


@pytest.mark.slow
def test_build_produces_conversations_starting_with_user_ending_with_assistant(tmp_path, monkeypatch):
    monkeypatch.setattr("coconut_lab.data.prepare_conversations.ARTIFACTS_DIR", tmp_path)
    paths = build(val_fraction=0.02, seed=0)

    train = load_jsonl(paths["train"])
    assert len(train) > 1000  # sanity: real dataset, not an empty/broken filter
    for ex in train[:200]:
        turns = ex["turns"]
        assert len(turns) >= 2
        assert turns[0]["role"] == "user"
        assert turns[-1]["role"] == "assistant"


@pytest.mark.slow
def test_conversation_dataset_masks_loss_to_every_assistant_turn(tokenizer):
    examples = [{"turns": [
        {"role": "user", "text": "Q1"},
        {"role": "assistant", "text": "A1"},
        {"role": "user", "text": "Q2"},
        {"role": "assistant", "text": "A2"},
    ]}]
    ds = ConversationDataset(examples, tokenizer, block_size=64)
    assert len(ds) == 1
    x, y, y_mask = ds[0]
    assert y_mask.sum().item() > 0

    # both assistant turns must contribute to the loss, not just the last one
    ids, is_response = ds.examples[0], ds.masks[0]
    text = tokenizer.decode(ids)
    assert "A1" in text and "A2" in text
    # there must be at least two separate contiguous runs of 1s in the mask
    runs = 0
    prev = 0
    for m in is_response:
        if m == 1 and prev == 0:
            runs += 1
        prev = m
    assert runs >= 2


@pytest.mark.slow
def test_conversation_dataset_truncates_from_the_start_keeping_recent_turns(tokenizer):
    long_text = "word " * 100
    examples = [{"turns": [
        {"role": "user", "text": f"old question {long_text}"},
        {"role": "assistant", "text": f"old answer {long_text}"},
        {"role": "user", "text": "recent question"},
        {"role": "assistant", "text": "recent answer"},
    ]}]
    ds = ConversationDataset(examples, tokenizer, block_size=40)
    assert len(ds) == 1
    text = tokenizer.decode(ds.examples[0])
    assert "recent answer" in text
    assert "old answer" not in text
