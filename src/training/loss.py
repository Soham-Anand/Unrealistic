import mlx.core as mx


def cross_entropy_loss(logits: mx.array, targets: mx.array) -> mx.array:
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size).astype(mx.float32)
    targets = targets.reshape(-1)

    logits_max = mx.max(logits, axis=-1, keepdims=True)
    logits = logits - logits_max

    log_sum_exp = mx.logsumexp(logits, axis=-1)
    selected = mx.take_along_axis(logits, targets.reshape(-1, 1), axis=-1).squeeze(-1)

    loss = log_sum_exp - selected
    return mx.mean(loss)
