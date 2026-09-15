import mlx.core as mx


def init_embeddings(vocab_size: int, hidden_size: int) -> dict:
    return {"token_embedding": mx.random.normal((vocab_size, hidden_size), scale=0.02)}


def embed_tokens(tokens: mx.array, params: dict) -> mx.array:
    return params["token_embedding"][tokens]
