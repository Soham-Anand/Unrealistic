import os
import time
import mlx.core as mx
from ..data.tokenizer import Tokenizer
from .generate import generate

DEFAULT_PROMPTS = [
    "The",
    "Once upon a time,",
    "In this article,",
    "The capital of France is",
    "The opposite of hot is",
]


def write_samples(params: dict, buffers: dict, model_cfg: dict, save_dir: str, step: int,
                  sample_cfg: dict | None = None,
                  prompts: list[str] | None = None,
                  temperatures: list[float] | None = None,
                  max_new_tokens: int | None = None,
                  top_k: int = 50, top_p: float = 0.0,
                  repetition_penalty: float = 1.0,
                  seed: int = 42, tokenizer_path: str = "data/tokenizer/phase1.model"):
    os.makedirs(save_dir, exist_ok=True)
    sample_cfg = sample_cfg if sample_cfg is not None else {}
    prompts = prompts if prompts is not None else sample_cfg.get("sample_prompts", DEFAULT_PROMPTS)
    temperatures = temperatures if temperatures is not None else sample_cfg.get("sample_temperatures", [0.0, 0.5])
    max_new_tokens = max_new_tokens if max_new_tokens is not None else sample_cfg.get("sample_max_tokens", 100)

    tokenizer = Tokenizer(tokenizer_path)

    lines = [
        "Unrealistic 187M — Sample Timeline",
        f"Checkpoint: step_{step:06d}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]

    for temperature in temperatures:
        for i, prompt in enumerate(prompts):
            block_seed = seed + i
            mx.random.seed(block_seed)
            tokens = tokenizer.encode(prompt)
            out_tokens = generate(params, buffers, model_cfg, tokens,
                                  max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  top_k=top_k, top_p=top_p,
                                  repetition_penalty=repetition_penalty)
            text = tokenizer.decode(out_tokens[len(tokens):])

            lines += [
                "",
                "=== Prompt ===",
                prompt,
                "",
                "=== Settings ===",
                f"temperature = {temperature}",
                f"top_p = {top_p}",
                f"top_k = {1 if temperature <= 0 else top_k}",
                f"repetition_penalty = {repetition_penalty}",
                f"seed = {block_seed}",
                f"max_tokens = {max_new_tokens}",
                "",
                "=== Output ===",
                text,
                "",
                "=" * 60,
            ]

    path = os.path.join(save_dir, "samples.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [samples] wrote {path}")
