import torch

from depth_lab.data.generator import generate_dataset
from depth_lab.data.loader import Seq2SeqDataset
from depth_lab.models.encoder_decoder import (
    EncDecConfig,
    EncoderDecoderTransformer,
    MultiHeadAttention,
    build_optimizer,
    evaluate_exact_match,
    sinusoidal_positional_encoding,
    train_steps,
)
from depth_lab.tokenizer import CharTokenizer

SRC_BLOCK_SIZE = 48  # generous enough for unbounded depth-2 bool expressions
TGT_BLOCK_SIZE = 6


def _tiny_model(vocab_size: int) -> EncoderDecoderTransformer:
    config = EncDecConfig(vocab_size=vocab_size, d_model=32, n_head=2, n_layer=2, d_ff=64,
                           max_src_len=SRC_BLOCK_SIZE, max_tgt_len=TGT_BLOCK_SIZE + 1)
    return EncoderDecoderTransformer(config)


def test_sinusoidal_positional_encoding_shape_and_bounds():
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    assert pe.shape == (10, 16)
    assert torch.all(pe <= 1.0) and torch.all(pe >= -1.0)


def test_encoder_output_is_unaffected_by_causal_ordering():
    """The whole point of the encoder is bidirectional self-attention: unlike
    the decoder-only baseline, a token's representation can depend on tokens
    that come *after* it, not just before."""
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()
    ids = torch.tensor([[tok.stoi["T"], tok.stoi["r"], tok.stoi["u"], tok.stoi["e"]] + [tok.pad_id] * (SRC_BLOCK_SIZE - 4)])
    memory = model.encode(ids)
    # perturb only the *last* real token and check the *first* token's
    # representation changes -- impossible under a causal mask.
    ids_perturbed = ids.clone()
    ids_perturbed[0, 3] = tok.stoi["F"]
    memory_perturbed = model.encode(ids_perturbed)
    assert not torch.allclose(memory[0, 0], memory_perturbed[0, 0])


def test_decoder_self_attention_is_causal():
    """Perturbing a *future* target token must not change an earlier
    position's logits -- the decoder must not be able to see ahead."""
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()
    src = torch.full((1, SRC_BLOCK_SIZE), tok.pad_id, dtype=torch.long)
    tgt = torch.tensor([[tok.bos_id, tok.stoi["T"], tok.stoi["r"], tok.stoi["u"], tok.stoi["e"], tok.eos_id]])

    memory = model.encode(src)
    logits = model.decode(tgt, memory)

    tgt_perturbed = tgt.clone()
    tgt_perturbed[0, -1] = tok.pad_id
    logits_perturbed = model.decode(tgt_perturbed, memory)

    assert torch.allclose(logits[0, 0], logits_perturbed[0, 0], atol=1e-5)


def test_cross_attention_lets_decoder_see_the_full_encoder_memory():
    """Changing the encoder input must change the decoder's output -- proof
    the cross-attention path is actually wired up (not just self-attention
    on the target)."""
    tok = CharTokenizer()
    model = _tiny_model(tok.vocab_size)
    model.eval()
    tgt = torch.tensor([[tok.bos_id, tok.stoi["T"]]])

    src_a = torch.tensor([[tok.stoi["T"], tok.stoi["r"], tok.stoi["u"], tok.stoi["e"]] + [tok.pad_id] * (SRC_BLOCK_SIZE - 4)])
    src_b = torch.tensor([[tok.stoi["F"], tok.stoi["a"], tok.stoi["l"], tok.stoi["s"]] + [tok.pad_id] * (SRC_BLOCK_SIZE - 4)])

    logits_a = model.decode(tgt, model.encode(src_a))
    logits_b = model.decode(tgt, model.encode(src_b))
    assert not torch.allclose(logits_a, logits_b)


def test_key_padding_mask_blocks_attention_to_padding():
    """A query attending with the padding mask on must be unaffected by what
    value the padding positions hold."""
    d_model, n_head = 16, 2
    mha = MultiHeadAttention(d_model, n_head)
    mha.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 4, d_model)
    key_padding_mask = torch.tensor([[False, False, True, True]])  # last two are pad

    out = mha(x, x, x, key_padding_mask=key_padding_mask)

    x_perturbed = x.clone()
    x_perturbed[0, 2:] = torch.randn(2, d_model) * 100  # trash the padding content
    out_perturbed = mha(x_perturbed, x_perturbed, x_perturbed, key_padding_mask=key_padding_mask)

    assert torch.allclose(out[0, :2], out_perturbed[0, :2], atol=1e-5)


def test_encoder_decoder_overfits_a_tiny_batch():
    torch.manual_seed(0)
    tokenizer = CharTokenizer()
    examples = [ex.__dict__ for ex in generate_dataset("bool", depths=range(0, 3), n_per_depth=4, seed=1)]

    model = _tiny_model(tokenizer.vocab_size)
    train_ds = Seq2SeqDataset(examples, tokenizer, SRC_BLOCK_SIZE, TGT_BLOCK_SIZE)
    optimizer = build_optimizer(model, lr=5e-3, weight_decay=0.0)

    train_steps(model, tokenizer, train_ds, optimizer, device="cpu", max_steps=400, batch_size=len(examples))

    acc = evaluate_exact_match(model, tokenizer, examples, device="cpu", src_block_size=SRC_BLOCK_SIZE)
    assert acc >= 0.9
