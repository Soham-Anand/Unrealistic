#!/usr/bin/env python3
"""Head-to-head inference across all GGUF quants (llama.cpp).
Usage: /tmp/gguf_venv/bin/python scripts/infer_gguf.py
  -> evals/battery/gguf_compare/<quant>.txt
"""
import os
from llama_cpp import Llama

PROMPTS = [
    "What is 2 + 2?",
    "What is 8 + 7?",
    "The capital of France is",
    "The largest planet in the Solar System is",
    "The lead singer of Linkin Park was",
    "Who built the Taj Mahal?",
    "Who is Narendra Modi?",
    "Hey Dude!",
    "Complete: The sun rises in the",
    "Write a Python function to add two numbers:",
]

QUANTS = ["f16", "Q8_0", "Q4_K_M"]


def main():
    os.makedirs("evals/battery/gguf_compare", exist_ok=True)
    for q in QUANTS:
        print(f"===== {q} =====", flush=True)
        llm = Llama(model_path=f"gguf/unrealistic-v1-{q}.gguf",
                    n_ctx=512, n_threads=8, verbose=False)
        lines = []
        for p in PROMPTS:
            out = llm(p, max_tokens=60, temperature=0.4, top_p=0.9,
                      repeat_penalty=1.25, stop=["User:"])
            txt = out["choices"][0]["text"].strip().replace("\n", " ")[:200]
            lines.append(f"--- {p!r} ---\n{txt}\n")
            print(f"--- {p!r} ---\n{txt}\n", flush=True)
        open(f"evals/battery/gguf_compare/{q}.txt", "w").write("\n".join(lines))
        del llm


if __name__ == "__main__":
    main()
