#!/usr/bin/env python3
"""RFT — Rejection Sampling Fine-tuning (RL-light answer-quality booster).

Phase 8: After Phase 7 SFT. Generates candidate answers from the model for
a battery of math/QA prompts, verifies each numeric answer with a safe
calculator, keeps only the CORRECT + CLEAN ones, and writes a new training
bin in chat format. Fine-tune on that with the normal train.py loop.

Why this targets the model's specific pathology: the probes show it knows
the right *first token* but collapses to SO/code garbage after. By
filtering to only clean, correct completions, we teach it to *sustain* a
good answer.

Usage:
  python3 scripts/rft.py --checkpoint checkpoints/step_XXXXXX \
      --out data/phase8/rft_train.bin --num-samples 400 --tokens-per-prompt 64

Then fine-tune:
  python3 scripts/train.py --train-config configs/phase8_rft_config.json \
      --model-config configs/model_config.json --reset-optimizer
"""

import os, sys, argparse, re, random, glob, ast, operator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx
from src.model.transformer import init_model
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate

DATA_DIR = "data/phase8"
EOS = None

SAFE_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow,
    ast.Mod: operator.mod, ast.UAdd: operator.pos, ast.USub: operator.neg,
    ast.BitXor: operator.xor, ast.BitOr: operator.or_, ast.BitAnd: operator.and_,
    ast.LShift: operator.lshift, ast.RShift: operator.rshift,
}


def safe_math(source: str):
    """Safely evaluate a simple arithmetic expression string."""
    source = re.sub(r"[^0-9+\-*/().\s^%]", "", source)
    source = source.replace("^", "**")
    try:
        return safe_eval_ast(ast.parse(source, mode="eval").body)
    except Exception:
        return None


def safe_eval_ast(node):
    if isinstance(node, ast.Expression):
        node = node.body
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = safe_eval_ast(node.left)
        right = safe_eval_ast(node.right)
        if left is None or right is None:
            return None
        return SAFE_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        val = safe_eval_ast(node.operand)
        if val is None:
            return None
        return SAFE_OPERATORS[type(node.op)](val)
    return None


# ─── Math prompt battery with known answers ─────────────────────────
# (prompt, expected numeric answer, tolerance)
MATH_BATTERY = [
    ("2+2=", 4), ("3+5=", 8), ("7+8=", 15), ("10+10=", 20),
    ("12-4=", 8), ("20-7=", 13), ("100-42=", 58), ("15-9=", 6),
    ("3*4=", 12), ("6*7=", 42), ("8*9=", 72), ("5*5=", 25),
    ("12/3=", 4), ("20/5=", 4), ("49/7=", 7), ("72/8=", 9),
    ("10-3=", 7), ("2^3=", 8), ("2^4=", 16), ("5^2=", 25),
    ("sqrt(9)=", 3), ("sqrt(16)=", 4), ("sqrt(25)=", 5), ("sqrt(81)=", 9),
    ("3+2=", 5), ("9-4=", 5), ("6*5=", 30), ("30/6=", 5),
    ("11+9=", 20), ("18-9=", 9), ("7*6=", 42), ("144/12=", 12),
    ("What is 2 plus 2?", 4),
    ("What is 12 divided by 4?", 3),
    ("What is 7 times 8?", 56),
    ("What is 15 minus 7?", 8),
    ("What is 3 to the power 3?", 27),
    ("What is the square root of 64?", 8),
    ("What is 100 divided by 10?", 10),
    ("What is 5 times 6?", 30),
    ("If you have 3 apples and get 4 more, how many do you have?", 7),
    ("A dozen eggs, and you use 5. How many are left?", 7),
]


def latest_checkpoint(root: str = "checkpoints") -> str:
    dirs = glob.glob(os.path.join(root, "step_*"))
    if not dirs:
        raise FileNotFoundError(f"No checkpoints under {root}")
    return max(dirs, key=os.path.getmtime)


def extract_answer(text: str):
    """Pull the final numeric answer from a generated completion."""
    # Look for '=' followed by a number: "2+2=4", "equals 4", "is 4"
    m = re.findall(r"(?:=|equals|is|answer\s*(?:is)?\s*:)\s*(-?\d+\.?\d*)", text)
    if m:
        val = float(m[-1])
        if abs(val) < 1e9 and (val == int(val) or abs(val - round(val)) < 1e-6):
            return val
    # Fall back: whole answer is just a bare number ("4" or "4.")
    stripped = text.strip().strip(".").strip()
    if re.fullmatch(r"-?\d+", stripped):
        return float(stripped)
    return None


def is_clean(text: str, tokenizer: Tokenizer) -> bool:
    """Reject responses that collapse into SO/code formatting."""
    collapsed_tokens = [
        "#define", "```", "<issue_comment>", "< /code>", "{", "}",
        "import numpy", "#include", "@classmethod", "def test(", "//=====",
        "#ifndef", "__repr__", "class nnfunc", "Upvotes", "</a>",
        "Desired:", "Enchantment:", "Problem:", "Explanation:",
        # GSM8K/Orca solution artifacts (wiki-phase style bleed — never in RFT)
        "####", "<<", ">>", "years old", "friends", "apples", "candies",
        "eggs", "toys", "books",
    ]
    for tok in collapsed_tokens:
        if tok in text:
            return False
    # must be reasonably short (a clean answer, not a code dump)
    if len(text) > 200:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="RFT: Rejection Sampling FT data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Phase 7 model (default: newest checkpoint)")
    parser.add_argument("--out", type=str, default=f"{DATA_DIR}/rft_train.bin")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Samples per prompt (self-consistency) — VERY GOOD 32 (was 8)")
    parser.add_argument("--max-tokens", type=int,
                        help="Target total tokens in output bin (default: all accepted)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for exploration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    mx.set_default_device(mx.gpu)
    ckpt = args.checkpoint or latest_checkpoint()
    params, _ = load_checkpoint(ckpt)
    _, buffers, model_cfg = init_model("configs/model_config.json")
    tokenizer = Tokenizer("data/tokenizer/phase1.model")
    global EOS
    EOS = tokenizer.eos_token

    print(f"Checkpoint: {ckpt}")
    print(f"Prompts: {len(MATH_BATTERY)}  Samples/prompt: {args.num_samples}")
    print(f"Temp: {args.temperature}")

    accepted = []        # list of (prompt, good_answer token ids)
    total_generated = 0
    total_kept = 0

    for ip, (prompt, expected) in enumerate(MATH_BATTERY):
        prompt_tokens = tokenizer.encode(prompt)
        prompt_str = f"User: {prompt}\nAssistant: "
        # For the training target we want the FULL intended answer.
        # We generate several candidates and keep those that are correct & clean.
        good = None
        for s in range(args.num_samples):
            mx.random.seed(args.seed + ip * 100 + s)
            out = generate(params, buffers, model_cfg, tokenizer.encode(prompt_str),
                           max_new_tokens=64,
                           temperature=args.temperature,
                           top_k=0, top_p=0.95,
                           repetition_penalty=1.2)
            text = tokenizer.decode(out)
            total_generated += 1
            if not is_clean(text, tokenizer):
                continue
            ans = extract_answer(text)
            if ans is None:
                continue
            if abs(ans - expected) < 1e-6:
                good = text
                break
        if good is not None:
            accepted.append((prompt_str, good))
            total_kept += 1

    print(f"\nRejection sampling done:")
    print(f"  generated: {total_generated}")
    print(f"  accepted:  {total_kept}  ({total_kept/len(MATH_BATTERY)*100:.0f}% of prompts)")

    # Tokenize + save all accepted samples
    n = 0
    with open(args.out, "wb") as f:
        for prompt_str, answer in accepted:
            full = tokenizer.encode(prompt_str + answer.strip())
            if not full:
                continue
            f.write(np.array(full, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(full)
    print(f"\nWrote {n:,} tokens -> {args.out}")

    # Also print a couple accepted & rejected examples for sanity
    print("\nSample of ACCEPTED answers:")
    for _, a in accepted[:3]:
        print(f"  {a!r}")

    if not accepted:
        print("\nWARNING: no answers accepted. Check quality of underlying model.")


if __name__ == "__main__":
    main()
