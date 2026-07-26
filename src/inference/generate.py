import mlx.core as mx
from ..model.transformer import UnrealisticModel


def generate(model: UnrealisticModel, prompt_tokens: list[int], max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> list[int]:
    tokens = list(prompt_tokens)

    for _ in range(max_new_tokens):
        x = mx.array([tokens], dtype=mx.int32)
        logits = model(x)
        next_logits = logits[0, -1, :]

        if temperature > 0:
            next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals = mx.topk(next_logits, top_k)
                threshold = topk_vals[-1]
                next_logits = mx.where(next_logits < threshold, float("-inf"), next_logits)
            probs = mx.softmax(next_logits)
            next_token = int(mx.random.categorical(probs))
        else:
            next_token = int(mx.argmax(next_logits))

        tokens.append(next_token)

        if next_token == model.config.get("eos_token_id", 2):
            break

    return tokens
