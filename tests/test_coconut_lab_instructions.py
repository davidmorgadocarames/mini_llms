import pytest
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import InstructionDataset
from coconut_lab.data.prepare_instructions import build, format_example, load_jsonl
from mini_llm.tokenizer import BPETokenizer
from mini_llm.tokenizer.bpe import EOT_TOKEN

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def test_format_example_without_input_omits_the_input_section():
    ex = format_example("Name a color.", "", "Blue.")
    assert "### Input:" not in ex["prompt"]
    assert ex["prompt"].endswith("### Response:\n")
    assert ex["response"] == "Blue."


def test_format_example_with_input_includes_it():
    ex = format_example("Translate to French.", "Hello", "Bonjour")
    assert "### Input:\nHello" in ex["prompt"]
    assert ex["response"] == "Bonjour"


@pytest.mark.slow
def test_build_produces_disjoint_train_val_splits(tmp_path, monkeypatch):
    monkeypatch.setattr("coconut_lab.data.prepare_instructions.ARTIFACTS_DIR", tmp_path)
    paths = build(val_fraction=0.02, seed=0)

    train = load_jsonl(paths["train"])
    val = load_jsonl(paths["val"])
    assert len(train) + len(val) == 52002
    assert 900 < len(val) < 1200  # ~2% of 52002

    train_prompts = {ex["prompt"] for ex in train}
    val_prompts = {ex["prompt"] for ex in val}
    assert train_prompts.isdisjoint(val_prompts)


@pytest.mark.slow
def test_build_is_reproducible_given_the_same_seed(tmp_path, monkeypatch):
    monkeypatch.setattr("coconut_lab.data.prepare_instructions.ARTIFACTS_DIR", tmp_path / "a")
    paths_a = build(val_fraction=0.02, seed=7)
    monkeypatch.setattr("coconut_lab.data.prepare_instructions.ARTIFACTS_DIR", tmp_path / "b")
    paths_b = build(val_fraction=0.02, seed=7)

    assert load_jsonl(paths_a["train"])[:5] == load_jsonl(paths_b["train"])[:5]


@pytest.mark.slow
def test_instruction_dataset_masks_loss_to_response_tokens_only(tokenizer):
    examples = [{"prompt": "### Instruction:\nSay hi.\n\n### Response:\n", "response": "Hi!"}]
    ds = InstructionDataset(examples, tokenizer, block_size=64)
    x, y, y_mask = ds[0]

    assert x.shape == (64,)
    assert y.shape == (64,)
    assert y_mask.shape == (64,)

    n_prompt_tokens = len(tokenizer.encode(examples[0]["prompt"]))
    # everything before the response starts must be masked out of the loss
    assert y_mask[: n_prompt_tokens - 1].sum().item() == 0
    # at least the response + EOT must be included
    assert y_mask.sum().item() >= len(tokenizer.encode(examples[0]["response"])) + 1


@pytest.mark.slow
def test_instruction_dataset_drops_examples_whose_prompt_alone_exceeds_block_size(tokenizer):
    huge_prompt = "word " * 200  # comfortably exceeds a block_size=8 budget
    examples = [{"prompt": huge_prompt, "response": "ok"}]
    ds = InstructionDataset(examples, tokenizer, block_size=8)
    assert len(ds) == 0


@pytest.mark.slow
def test_pad_id_is_the_eot_token(tokenizer):
    examples = [{"prompt": "### Response:\n", "response": "ok"}]
    ds = InstructionDataset(examples, tokenizer, block_size=32)
    assert ds.pad_id == tokenizer.encode(EOT_TOKEN)[0]
