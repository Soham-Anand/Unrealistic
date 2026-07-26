import mlx.core as mx


class TokenEmbeddings:
    def __init__(self, vocab_size: int, hidden_size: int):
        self.token_embedding = mx.random.normal((vocab_size, hidden_size), std=0.02)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.token_embedding[tokens]
