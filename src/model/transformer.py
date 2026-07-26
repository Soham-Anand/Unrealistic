import json
import mlx.core as mx
from .attention import MultiHeadAttention
from .mlp import SwiGLU
from .norm import RMSNorm
from .embeddings import TokenEmbeddings


class TransformerBlock:
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, intermediate_size: int):
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = SwiGLU(hidden_size, intermediate_size)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        x = x + self.attention(self.attention_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class UnrealisticModel:
    def __init__(self, config_path: str = "configs/model_config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        self.config = cfg
        self.embeddings = TokenEmbeddings(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [
            TransformerBlock(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_attention_heads"],
                num_kv_heads=cfg["num_key_value_heads"],
                intermediate_size=cfg["intermediate_size"],
            )
            for _ in range(cfg["num_hidden_layers"])
        ]
        self.final_norm = RMSNorm(cfg["hidden_size"])

        if cfg["tie_word_embeddings"]:
            self.lm_head = None
        else:
            self.lm_head = mx.random.normal((cfg["hidden_size"], cfg["vocab_size"]), std=0.02)

    def _causal_mask(self, seq_len: int) -> mx.array:
        mask = mx.full((seq_len, seq_len), float("-inf"))
        mask = mx.triu(mask, k=1)
        return mask.reshape(1, 1, seq_len, seq_len)

    def __call__(self, tokens: mx.array) -> mx.array:
        B, T = tokens.shape
        x = self.embeddings(tokens)
        mask = self._causal_mask(T)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        if self.lm_head is not None:
            logits = x @ self.lm_head
        else:
            logits = x @ self.embeddings.token_embedding.T

        return logits

    def count_parameters(self) -> int:
        total = 0
        total += self.embeddings.token_embedding.size
        for layer in self.layers:
            total += layer.attention_norm.weight.size
            total += layer.attention.Wq.size + layer.attention.Wk.size
            total += layer.attention.Wv.size + layer.attention.Wo.size
            total += layer.ffn_norm.weight.size
            total += layer.ffn.w_gate.size + layer.ffn.w_up.size + layer.ffn.w_down.size
        total += self.final_norm.weight.size
        if self.lm_head is not None:
            total += self.lm_head.size
        return total
