import pytest
import torch
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import Seq2SeqDataset
from coconut_lab.models.sliced import (
    SRC_BLOCK_SIZE,
    TGT_BLOCK_SIZE,
    alpaca_to_seq2seq,
    build_config,
    build_optimizer,
    conversations_to_seq2seq,
    generate_response,
    train_steps,
)
from depth_lab.models.encoder_decoder import EncoderDecoderTransformer
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def test_alpaca_to_seq2seq_maps_prompt_and_response():
    examples = [{"prompt": "Say hi.", "response": "Hi!"}]
    out = alpaca_to_seq2seq(examples)
    assert out == [{"src": "Say hi.", "tgt": "Hi!"}]


def test_conversations_to_seq2seq_uses_history_as_src_and_last_turn_as_tgt():
    examples = [{"turns": [
        {"role": "user", "text": "Q1"},
        {"role": "assistant", "text": "A1"},
        {"role": "user", "text": "Q2"},
        {"role": "assistant", "text": "A2"},
    ]}]
    out = conversations_to_seq2seq(examples)
    assert len(out) == 1
    assert out[0]["tgt"] == "A2"
    assert "Q1" in out[0]["src"] and "A1" in out[0]["src"] and "Q2" in out[0]["src"]
    assert "A2" not in out[0]["src"]


def test_conversations_to_seq2seq_drops_conversations_not_ending_in_assistant():
    examples = [{"turns": [{"role": "user", "text": "Q1"}]}]
    assert conversations_to_seq2seq(examples) == []


@pytest.mark.slow
def test_sliced_overfits_a_tiny_batch(tokenizer):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = build_config(tokenizer.vocab_size, n_layer=2, d_model=64, n_head=4, d_ff=128)
    model = EncoderDecoderTransformer(config).to(device)

    examples = [
        {"src": "Name the color of the sky.", "tgt": "The sky is blue."},
        {"src": "Name a common pet.", "tgt": "A dog is a common pet."},
    ]
    train_ds = Seq2SeqDataset(examples, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    train_steps(model, train_ds, optimizer, device, max_steps=300, batch_size=len(examples), pad_id=train_ds.pad_id)

    response = generate_response(model, tokenizer, examples[0]["src"], device, max_new_tokens=20)
    assert "blue" in response.lower()
