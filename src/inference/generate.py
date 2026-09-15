import mlx.core as mx
from ..model.transformer import forward


def generate(params: dict, buffers: dict, cfg: dict, prompt_tokens: list[int],
             max_new_tokens: int = 100, temperature: float = 0.8,
             top_k: int = 50, top_p: float = 0.0,
             repetition_penalty: float = 1.0) -> list[int]:
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        x = mx.array([tokens], dtype=mx.int32)
        logits = forward(params, buffers, x, cfg, training=False)
        next_logits = logits[0, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature
            next_logits = _apply_repetition_penalty(next_logits, tokens, repetition_penalty)
            if top_k > 0:
                topk_vals = mx.topk(next_logits, top_k)
                threshold = mx.min(topk_vals)
                next_logits = mx.where(next_logits < threshold, float("-inf"), next_logits)
            if top_p > 0.0:
                next_logits = _apply_top_p(next_logits, top_p)
            next_token = int(mx.random.categorical(next_logits))
        else:
            next_logits = _apply_repetition_penalty(next_logits, tokens, repetition_penalty)
            next_token = int(mx.argmax(next_logits))

        tokens.append(next_token)

        if next_token == cfg.get("eos_token_id", 2):
            break

    return tokens


def _apply_repetition_penalty(logits: mx.array, tokens: list[int],
                              penalty: float = 1.0) -> mx.array:
    if penalty <= 1.0 or not tokens:
        return logits
    idx = mx.array(list(set(tokens)))
    vals = logits[idx]
    penalized = mx.where(vals > 0, vals / penalty, vals * penalty)
    delta = mx.zeros_like(logits)
    delta = delta.at[idx].add(penalized - vals)
    return logits + delta


def _apply_top_p(logits: mx.array, p: float) -> mx.array:
    if p <= 0.0 or p >= 1.0:
        return logits
    probs = mx.softmax(logits)
    sorted_asc = mx.sort(probs)
    cum_asc = mx.cumsum(sorted_asc)
    k = int(mx.sum(cum_asc < (1.0 - p)))
    if k == 0:
        return logits
    if k >= probs.size:
        return logits
    threshold = sorted_asc[k - 1]
    return mx.where(probs < threshold, float("-inf"), logits)
