from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 32
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
