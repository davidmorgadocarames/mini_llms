import pytest
import torch
from huggingface_hub import hf_hub_download

from coconut_lab.data.loader import InstructionDataset
from coconut_lab.models.cracked import build_optimizer, train_steps
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import BPETokenizer
from mini_llm.train.checkpoint import load_checkpoint, save_checkpoint

HF_REPO = "davidmorgado/coconut-mini-llm"


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    vocab_path = hf_hub_download(HF_REPO, "tokenizer/vocab.json")
    merges_path = hf_hub_download(HF_REPO, "tokenizer/merges.txt")
    return BPETokenizer(vocab_path, merges_path)


def _tiny_gpt(tokenizer: BPETokenizer) -> GPT:
    config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64, n_layer=2, n_embd=64, n_head=4, n_kv_head=2)
    return GPT(config)


def test_save_and_load_checkpoint_round_trips_model_and_optimizer_state(tokenizer, tmp_path):
    model = _tiny_gpt(tokenizer)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)

    # Take a real step so the optimizer's Adam state (momentum/variance) is
    # non-empty -- an empty/default optimizer state would trivially "match"
    # even if load_checkpoint never actually restored anything.
    x = torch.randint(0, tokenizer.vocab_size, (1, 8))
    logits, loss = model(x, x)
    loss.backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=42, config=model.config)

    fresh_model = _tiny_gpt(tokenizer)
    fresh_optimizer = build_optimizer(fresh_model, lr=1e-3, weight_decay=0.0)
    loaded_step = load_checkpoint(path, fresh_model, fresh_optimizer, device="cpu")

    assert loaded_step == 42
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh_model.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2)
    assert optimizer.state_dict()["state"].keys() == fresh_optimizer.state_dict()["state"].keys()
    for key in optimizer.state_dict()["state"]:
        orig, loaded = optimizer.state_dict()["state"][key], fresh_optimizer.state_dict()["state"][key]
        assert torch.equal(orig["exp_avg"], loaded["exp_avg"])
        assert torch.equal(orig["exp_avg_sq"], loaded["exp_avg_sq"])


def test_load_checkpoint_defaults_to_step_zero_when_key_missing(tokenizer, tmp_path):
    model = _tiny_gpt(tokenizer)
    path = tmp_path / "no_step.pt"
    torch.save({"model": model.state_dict()}, path)

    fresh_model = _tiny_gpt(tokenizer)
    step = load_checkpoint(path, fresh_model, device="cpu")
    assert step == 0


@pytest.mark.slow
def test_train_steps_does_not_retrain_when_checkpoint_already_reached_max_steps(tokenizer, tmp_path):
    device = "cpu"
    model = _tiny_gpt(tokenizer).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    path = tmp_path / "cracked.pt"
    save_checkpoint(path, model, optimizer, step=50, config=model.config)
    weights_before = {n: p.clone() for n, p in model.state_dict().items()}

    examples = [{"prompt": "Hello ", "response": "world."}]
    train_ds = InstructionDataset(examples, tokenizer, block_size=model.config.block_size)
    train_steps(model, train_ds, optimizer, device=device, max_steps=50, batch_size=1,
                checkpoint_path=path, checkpoint_interval=10)

    for n, p in model.state_dict().items():
        assert torch.equal(p, weights_before[n])


@pytest.mark.slow
def test_train_steps_resumes_from_the_saved_step_instead_of_restarting(tokenizer, tmp_path):
    device = "cpu"
    torch.manual_seed(0)
    model = _tiny_gpt(tokenizer).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    path = tmp_path / "cracked.pt"

    examples = [{"prompt": "Hello ", "response": "world."}]
    train_ds = InstructionDataset(examples, tokenizer, block_size=model.config.block_size)

    # First "session": trains halfway and checkpoints.
    train_steps(model, train_ds, optimizer, device=device, max_steps=20, batch_size=1,
                checkpoint_path=path, checkpoint_interval=20)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert ckpt["step"] == 20

    # Second "session" (simulating a resume after a crash): a *fresh*
    # model/optimizer, same checkpoint_path, higher max_steps -- must pick
    # up from step 20, not step 0.
    resumed_model = _tiny_gpt(tokenizer).to(device)
    resumed_optimizer = build_optimizer(resumed_model, lr=1e-3, weight_decay=0.0)
    train_steps(resumed_model, train_ds, resumed_optimizer, device=device, max_steps=40, batch_size=1,
                checkpoint_path=path, checkpoint_interval=40)
    final_ckpt = torch.load(path, map_location=device, weights_only=False)
    assert final_ckpt["step"] == 40
