import pytest
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import PressedLocatorDataset
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


@pytest.mark.slow
def test_char_span_converts_to_the_matching_token_span(tokenizer):
    text = "Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    char_start = text.index("<<48/2=24>>24")
    char_end = char_start + len("<<48/2=24>>24")
    ds = PressedLocatorDataset([{"text": text, "span": [char_start, char_end]}], tokenizer, block_size=64)
    assert len(ds) == 1

    ids, labels, pad_mask = ds[0]
    ids_list, offsets = tokenizer.encode_with_offsets(text)
    marked_token_indices = [i for i, v in enumerate(labels.tolist()) if v == 1.0]

    # every marked token's character range must fall within the target span
    for i in marked_token_indices:
        s, e = offsets[i]
        assert s < char_end and e > char_start

    # the decoded marked tokens, concatenated, must reconstruct the span text
    marked_ids = [ids_list[i] for i in marked_token_indices]
    assert tokenizer.decode(marked_ids).strip() == "<<48/2=24>>24"


@pytest.mark.slow
def test_examples_longer_than_block_size_are_dropped(tokenizer):
    text = "word " * 500
    ds = PressedLocatorDataset([{"text": text, "span": [0, 4]}], tokenizer, block_size=8)
    assert len(ds) == 0
