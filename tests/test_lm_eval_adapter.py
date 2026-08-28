import pytest
import torch
from huggingface_hub import hf_hub_download
from lm_eval.api.instance import Instance

from coconut_lab.data.loader import InstructionDataset, Seq2SeqDataset
from coconut_lab.eval.lm_eval_adapter import GPTFamilyAdapter, SlicedAdapter
from coconut_lab.models.cracked import build_optimizer as build_gpt_optimizer
from coconut_lab.models.cracked import train_steps as gpt_train_steps
from coconut_lab.models.sliced import SRC_BLOCK_SIZE, TGT_BLOCK_SIZE, build_config
from coconut_lab.models.sliced import build_optimizer as build_sliced_optimizer
from coconut_lab.models.sliced import train_steps as sliced_train_steps
from depth_lab.models.encoder_decoder import EncoderDecoderTransformer
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def _loglikelihood_instance(context: str, continuation: str) -> Instance:
    return Instance(request_type="loglikelihood", doc={}, arguments=(context, continuation), idx=0)


def _tiny_gpt(tokenizer: BPETokenizer) -> GPT:
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2, n_embd=64, n_head=4, n_kv_head=2)
    return GPT(config)


# --- fast interface/plumbing tests (untrained models, just check shapes/types) ---

def test_gpt_family_adapter_loglikelihood_dispatches_instance_args(tokenizer):
    device = "cpu"
    model = _tiny_gpt(tokenizer).to(device)
    adapter = GPTFamilyAdapter(model, tokenizer, device, "test")

    results = adapter.loglikelihood([_loglikelihood_instance("hello", " world"),
                                      _loglikelihood_instance("", "x")])
    assert len(results) == 2
    for logprob, is_greedy in results:
        assert isinstance(logprob, float)
        assert isinstance(is_greedy, bool)


def test_gpt_family_adapter_loglikelihood_of_empty_continuation_is_zero(tokenizer):
    device = "cpu"
    model = _tiny_gpt(tokenizer).to(device)
    adapter = GPTFamilyAdapter(model, tokenizer, device, "test")
    logprob, is_greedy = adapter._loglikelihood_one("some context", "")
    assert logprob == 0.0
    assert is_greedy is True


def test_gpt_family_adapter_generate_until_truncates_at_stop_sequence(tokenizer):
    device = "cpu"
    model = _tiny_gpt(tokenizer).to(device)
    adapter = GPTFamilyAdapter(model, tokenizer, device, "test")
    req = Instance(request_type="generate_until", doc={}, arguments=("hello", {"until": ["\n"], "max_gen_toks": 10}),
                   idx=0)
    out = adapter.generate_until([req])
    assert len(out) == 1
    assert isinstance(out[0], str)


def test_sliced_adapter_loglikelihood_dispatches_instance_args(tokenizer):
    device = "cpu"
    config = build_config(tokenizer.vocab_size, n_layer=2, d_model=64, n_head=4, d_ff=128)
    model = EncoderDecoderTransformer(config).to(device)
    adapter = SlicedAdapter(model, tokenizer, device)

    results = adapter.loglikelihood([_loglikelihood_instance("hello", " world")])
    assert len(results) == 1
    logprob, is_greedy = results[0]
    assert isinstance(logprob, float)
    assert isinstance(is_greedy, bool)


# --- oracle-style correctness tests: a memorized continuation must score
# higher than an unrelated one, same "overfit a tiny batch" discipline used
# throughout this project (test_coconut_lab_cracked.py / _sliced.py) ---

@pytest.mark.slow
def test_gpt_family_adapter_scores_the_memorized_continuation_higher(tokenizer):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _tiny_gpt(tokenizer).to(device)

    examples = [{"prompt": "The sky is ", "response": "blue."}]
    train_ds = InstructionDataset(examples, tokenizer, block_size=model.config.block_size)
    optimizer = build_gpt_optimizer(model, lr=1e-3, weight_decay=0.0)
    gpt_train_steps(model, train_ds, optimizer, device=device, max_steps=300, batch_size=1)

    adapter = GPTFamilyAdapter(model, tokenizer, device, "test")
    correct_lp, correct_greedy = adapter._loglikelihood_one("The sky is ", "blue.")
    wrong_lp, _ = adapter._loglikelihood_one("The sky is ", "a dinosaur.")

    assert correct_lp > wrong_lp
    assert correct_greedy is True


@pytest.mark.slow
def test_sliced_adapter_scores_the_memorized_continuation_higher(tokenizer):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_config(tokenizer.vocab_size, n_layer=2, d_model=64, n_head=4, d_ff=128)
    model = EncoderDecoderTransformer(config).to(device)

    examples = [{"src": "The sky is ", "tgt": "blue."}]
    train_ds = Seq2SeqDataset(examples, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    optimizer = build_sliced_optimizer(model, lr=1e-3, weight_decay=0.0)
    sliced_train_steps(model, train_ds, optimizer, device, max_steps=600, batch_size=1, pad_id=train_ds.pad_id)

    adapter = SlicedAdapter(model, tokenizer, device)
    correct_lp, correct_greedy = adapter._loglikelihood_one("The sky is ", "blue.")
    wrong_lp, _ = adapter._loglikelihood_one("The sky is ", "a dinosaur.")

    assert correct_lp > wrong_lp
    assert correct_greedy is True
