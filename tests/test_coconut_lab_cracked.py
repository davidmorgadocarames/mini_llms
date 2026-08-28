import pytest
import torch

from coconut_lab.data.loader import InstructionDataset
from coconut_lab.models.cracked import (
    build_optimizer,
    generate_response,
    load_base_model,
    masked_loss,
    train_steps,
)


def test_masked_loss_ignores_masked_positions():
    torch.manual_seed(0)
    vocab_size = 5
    logits = torch.randn(1, 3, vocab_size, requires_grad=True)
    y = torch.tensor([[0, 1, 2]])

    mask_all = torch.tensor([[1.0, 1.0, 1.0]])
    mask_last_only = torch.tensor([[0.0, 0.0, 1.0]])

    loss_all = masked_loss(logits, y, mask_all)
    loss_last = masked_loss(logits, y, mask_last_only)

    # loss restricted to the last position should equal the per-token CE
    # at that position alone, not an average over all three
    import torch.nn.functional as F
    expected_last = F.cross_entropy(logits[0, 2:3], y[0, 2:3])
    assert torch.allclose(loss_last, expected_last, atol=1e-5)
    assert not torch.allclose(loss_all, loss_last)


@pytest.mark.slow
def test_cracked_overfits_a_tiny_batch_of_instructions():
    """Same sanity check used throughout this project: a model this size
    fine-tuning on a handful of short instruction/response pairs should be
    able to memorize them almost verbatim within a few hundred steps."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_base_model(device)

    examples = [
        {
            "prompt": "Below is an instruction that describes a task. Write a response that "
                      "appropriately completes the request.\n\n### Instruction:\nName the color of the sky.\n\n### Response:\n",
            "response": "The sky is blue.",
        },
        {
            "prompt": "Below is an instruction that describes a task. Write a response that "
                      "appropriately completes the request.\n\n### Instruction:\nName a common pet.\n\n### Response:\n",
            "response": "A dog is a common pet.",
        },
    ]
    train_ds = InstructionDataset(examples, tokenizer, block_size=model.config.block_size)
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.0)
    train_steps(model, train_ds, optimizer, device=device, max_steps=300, batch_size=len(examples))

    response = generate_response(model, tokenizer, examples[0]["prompt"], device,
                                  max_new_tokens=20, temperature=1e-6, top_k=1)
    assert "blue" in response.lower()
