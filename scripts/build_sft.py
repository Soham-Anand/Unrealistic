#!/usr/bin/env python3
"""Build Phase 7 SFT dataset: Instruction tuning (~5-10M tokens).

Generates instruction-following data in chat format for the final
fine-tuning phase. Covers math, code, general knowledge, and reasoning.

Output: data/phase7/train.bin, data/phase7/eval.bin
"""

import os, sys, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase7"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/train.bin"
OUT_EVAL = f"{DATA_DIR}/eval.bin"
EOS = None

TOTAL_TARGET = 25_000_000  # VERY GOOD topper 25M (was 8M mediocre)


def gen_math_instructions(rng, n):
    # VERY GOOD: direct + step-by-step CoT math, covers + - * / powers roots fractions percents
    templates = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div", "pow", "sqrt", "pct", "frac", "cot_add", "cot_mul"])
        if kind == "add":
            a, b = rng.randint(1, 200), rng.randint(1, 200)
            templates.append(f"User: What is {a} + {b}?\nAssistant: {a} + {b} = {a+b}")
        elif kind == "sub":
            a = rng.randint(10, 200); b = rng.randint(1, a)
            templates.append(f"User: What is {a} - {b}?\nAssistant: {a} - {b} = {a-b}")
        elif kind == "mul":
            a, b = rng.randint(2, 25), rng.randint(2, 25)
            templates.append(f"User: What is {a} times {b}?\nAssistant: {a} × {b} = {a*b}")
        elif kind == "div":
            b = rng.randint(1, 12); q = rng.randint(1, 25); a = b * q
            templates.append(f"User: What is {a} divided by {b}?\nAssistant: {a} ÷ {b} = {q}")
        elif kind == "pow":
            a = rng.randint(2, 12); e = rng.choice([2, 3])
            templates.append(f"User: What is {a} to the power {e}?\nAssistant: {a}^{e} = {a**e}")
        elif kind == "sqrt":
            r = rng.randint(2, 20); sq = r * r
            templates.append(f"User: What is the square root of {sq}?\nAssistant: The square root of {sq} is {r}.")
        elif kind == "pct":
            p = rng.choice([10, 15, 20, 25, 50]); v = rng.choice([40, 60, 80, 100, 200])
            ans = p * v // 100
            templates.append(f"User: What is {p}% of {v}?\nAssistant: {p}% of {v} = {ans}.")
        elif kind == "frac":
            templates.append("User: What is 1/2 + 1/4?\nAssistant: Common denominator 4. 1/2 = 2/4. 2/4 + 1/4 = 3/4.")
        elif kind == "cot_add":
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            templates.append(f"User: Add {a} + {b} step by step.\nAssistant: Step 1: {a} + {b}. Step 2: ones, tens, hundreds with carries. Answer: {a+b}.")
        else:
            a, b = rng.randint(11, 99), rng.randint(2, 9)
            templates.append(f"User: Multiply {a} x {b} step by step.\nAssistant: Step 1: {a} x {b}. Step 2: partial products. Answer: {a*b}.")
    return templates


def gen_code_instructions(rng, n):
    templates = []
    snippets = [
        ("Write a Python function to check if a number is prime.",
         "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"),
        ("Write a Python function to reverse a string.",
         "def reverse(s):\n    return s[::-1]"),
        ("Write a Python function to find the factorial of n.",
         "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
        ("Write a Python function to check if a string is a palindrome.",
         "def is_palindrome(s):\n    return s == s[::-1]"),
        ("Write a Python function to find the GCD of two numbers.",
         "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
        ("Write a Python function to count vowels in a string.",
         "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Write a Python function to find the largest element in a list.",
         "def largest(lst):\n    return max(lst)"),
        ("Write a Python function to compute Fibonacci numbers.",
         "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
        ("Write a Python function to sort a list using bubble sort.",
         "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"),
        ("Write a Python function to check if a number is even.",
         "def is_even(n):\n    return n % 2 == 0"),
    ]
    for _ in range(n):
        q, a = rng.choice(snippets)
        templates.append(f"User: {q}\nAssistant:\n```python\n{a}\n```")
    return templates


def gen_general_instructions(rng, n):
    templates = []
    qa = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
        ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet around 1594-1596."),
        ("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."),
        ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for the development and functioning of all known living organisms."),
        ("What causes rain?", "Rain is caused by water evaporating from the earth, rising into the atmosphere, cooling into clouds, and falling back to the ground as precipitation."),
        ("What is Newton's second law?", "Newton's second law states that Force equals mass times acceleration (F = ma)."),
        ("What is the largest planet?", "Jupiter is the largest planet in our solar system."),
        ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides (a² + b² = c²)."),
        ("Water is made of hydrogen and what?", "Water is made of hydrogen and oxygen (H2O)."),
        ("What gas do plants absorb for photosynthesis?", "Plants absorb carbon dioxide and release oxygen."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("Why do we have seasons?", "We have seasons because Earth's axis is tilted, changing sunlight through the year."),
        ("Where is the Great Wall?", "The Great Wall is a series of fortifications built by successive Chinese dynasties, mainly Ming (1368-1644)."),
        ("What is the chemical symbol for water?", "The chemical symbol for water is H2O."),
        ("What force pulls objects toward Earth?", "Gravity pulls objects toward the Earth."),
        ("How long does Earth take to orbit the Sun?", "Earth takes about one year (365 days) to orbit the Sun."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean on Earth."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("Who was the first Prime Minister of India?", "Jawaharlal Nehru was the first Prime Minister of India."),
        ("Which monument in Agra is a wonder of the world?", "The Taj Mahal in Agra is one of the wonders of the world."),
        ("What is the national animal of India?", "The Bengal tiger is the national animal of India."),
        ("Which river is considered holy in India?", "The Ganga (Ganges) river is considered holy in India."),
        ("What is the currency of India?", "The currency of India is the Rupee (INR)."),
        ("In which year did India gain independence?", "India gained independence in 1947."),
        ("Who wrote the Indian national anthem?", "Rabindranath Tagore wrote the Indian national anthem (Jana Gana Mana)."),
        ("Which festival is the festival of lights in India?", "Diwali is the festival of lights in India."),
    ]
    for _ in range(n):
        q, a = rng.choice(qa)
        templates.append(f"User: {q}\nAssistant: {a}")
    return templates


def gen_reasoning_instructions(rng, n):
    templates = []
    problems = [
        ("If a train travels 60 km in 1 hour, how far does it travel in 3 hours?",
         "Step 1: Speed = 60 km/h\nStep 2: Distance = Speed × Time = 60 × 3 = 180 km\nAnswer: 180 km"),
        ("Janet has 3 apples. She gets 5 more. How many does she have?",
         "Step 1: Start = 3\nStep 2: Add 5. 3 + 5 = 8.\nAnswer: 8"),
        ("There are 4 boxes with 6 books each. How many books?",
         "Step 1: 4 boxes × 6 = 24.\nAnswer: 24 books"),
        ("Linda has 12 candies. She gives 4 to her brother. How many left?",
         "Step 1: 12 - 4 = 8.\nAnswer: 8"),
        ("If 3 workers make 9 toys each, how many toys total?",
         "Step 1: 3 × 9 = 27.\nAnswer: 27 toys"),
        ("A shirt costs $25. It's on sale for 20% off. What's the sale price?",
         "Step 1: Discount = 20% of $25 = $5\nStep 2: Sale price = $25 - $5 = $20\nAnswer: $20"),
        ("There are 30 students in a class. If 18 are girls, what fraction are boys?",
         "Step 1: Boys = 30 - 18 = 12\nStep 2: Fraction = 12/30 = 2/5\nAnswer: 2/5"),
        ("If you buy 5 apples at $0.50 each, how much do you pay?",
         "Step 1: Cost = 5 × $0.50 = $2.50\nAnswer: $2.50"),
        ("A rectangle has length 8 and width 5. What is its area?",
         "Step 1: Area = length × width = 8 × 5 = 40\nAnswer: 40"),
        ("A car travels 60 km in 2 hours. What is its speed?",
         "Step 1: Speed = Distance / Time = 60 / 2 = 30.\nAnswer: 30 km/h"),
        ("Someone says 7 + 8 = 16. Is that correct?",
         "No. 7 + 8 = 15, not 16. The claim 16 is false."),
        ("Verify: is x = 4 a solution to 2x + 6 = 14?",
         "Step 1: 2*4 + 6 = 8 + 6 = 14. Yes, x = 4 is correct."),
    ]
    for _ in range(n):
        q, a = rng.choice(problems)
        templates.append(f"User: {q}\nAssistant: {a}")
    return templates


def gen_creative_instructions(rng, n):
    templates = []
    pairs = [
        ("Complete: The sun rises in the", "The sun rises in the east."),
        ("What comes next: 2, 4, 6, 8, ?", "What comes next: 2, 4, 6, 8, 10. Each step adds 2."),
        ("If today is Monday, what day is tomorrow?", "If today is Monday, tomorrow is Tuesday."),
        ("What is the past tense of 'go'?", "The past tense of 'go' is 'went'."),
        ("Translate 'hello' to French.", "Hello in French is 'bonjour'."),
    ]
    for _ in range(n):
        q, a = rng.choice(pairs)
        templates.append(f"User: {q}\nAssistant: {a}")
    return templates


def _oa_quality_ok(row, thresh=0.5):
    # labels is a struct of parallel lists: {name:[...], value:[...]} (or dict form)
    try:
        labels = row.get("labels", None)
        if not labels:
            return True  # no labels -> fall back to rank/review gate only
        names = labels.get("name", []) if isinstance(labels, dict) else []
        vals = labels.get("value", []) if isinstance(labels, dict) else []
        for n, v in zip(names, vals):
            if n == "quality":
                return float(v) >= thresh
        return True  # no quality label present -> don't exclude
    except Exception:
        return True


def _oa_detox_ok(row, thresh=0.2):
    try:
        d = row.get("detoxify", None) or {}
        t = d.get("toxicity", 0.0)
        return float(t) < thresh
    except Exception:
        return True


def fetch_oa_instructions(tokenizer, target_tokens, seed=42, cache_dir="/tmp/oa_cache"):
    """Stream OASST1+OASST2 (en, rank-0, reviewed, quality-gated) -> (chat, remembrance) pairs.

    Returns two lists of 'User: ...\\nAssistant: ...' strings split by assistant
    length (<400 chars chat, >=400 remembrance). Streaming-only (disk is 95%+).
    """
    import os as _os
    _os.environ["HF_DATASETS_CACHE"] = cache_dir
    _os.makedirs(cache_dir, exist_ok=True)
    from datasets import load_dataset as _load

    rng = random.Random(seed)
    chat, remem = [], []
    seen = set()
    prompter_by_id = {}
    pending = {}  # parent_id -> list of assistant rows waiting for prompter text
    tok_count = {"chat": 0, "remem": 0}

    def try_emit(a_row, p_text):
        p = (p_text or "").strip()
        a = (a_row.get("text") or "").strip()
        if len(p) < 5 or len(p) > 2000 or len(a) < 10 or len(a) > 4000:
            return False
        pair = f"User: {p}\nAssistant: {a}"
        h = hash(pair)
        if h in seen:
            return False
        ids = tokenizer.encode(pair)
        if not ids:
            return False
        seen.add(h)
        if len(a) < 400:
            chat.append(pair)
            tok_count["chat"] += len(ids)
        else:
            remem.append(pair)
            tok_count["remem"] += len(ids)
        return True

    def row_ok(r):
        if r.get("lang") != "en":
            return False
        if r.get("deleted"):
            return False
        if r.get("review_result") is not True:
            return False
        if r.get("synthetic"):
            return False
        if not _oa_detox_ok(r):
            return False
        if not _oa_quality_ok(r):
            return False
        return True

    for repo in ("OpenAssistant/oasst1", "OpenAssistant/oasst2"):
        if tok_count["chat"] + tok_count["remem"] >= target_tokens:
            break
        print(f"  streaming {repo} ...")
        try:
            ds = _load(repo, split="train", streaming=True)
        except Exception as e:
            print(f"    failed ({e}); skipping repo")
            continue
        for r in ds:
            if tok_count["chat"] + tok_count["remem"] >= target_tokens:
                break
            try:
                role = r.get("role")
                if role == "prompter":
                    if r.get("lang") != "en" or r.get("deleted"):
                        continue
                    t = (r.get("text") or "").strip()
                    if not t:
                        continue
                    mid = r.get("message_id")
                    if mid:
                        prompter_by_id[mid] = t
                        if len(prompter_by_id) > 400000:
                            # bound memory: drop oldest batch
                            for k in list(prompter_by_id)[:50000]:
                                del prompter_by_id[k]
                    if mid in pending:
                        for a_row in pending.pop(mid):
                            if row_ok(a_row):
                                try_emit(a_row, t)
                elif role == "assistant":
                    if r.get("rank") != 0:
                        continue
                    if not row_ok(r):
                        continue
                    pid = r.get("parent_id")
                    p_text = prompter_by_id.get(pid) if pid else None
                    if p_text:
                        try_emit(r, p_text)
                    elif pid:
                        lst = pending.setdefault(pid, [])
                        if len(lst) < 8 and len(pending) < 50000:
                            lst.append(dict(r))
            except Exception:
                continue
        print(f"    so far chat={len(chat)} remem={len(remem)} tok={tok_count['chat']+tok_count['remem']:,}")
    print(f"  OA kept: chat {len(chat)} ({tok_count['chat']:,} tok) remem {len(remem)} ({tok_count['remem']:,} tok)")
    return chat, remem


def write_docs_cycled(tokenizer, templates, out_path, target_tokens, rng):
    # VERY GOOD: cycle through templates until token budget is met (fills 25M)
    n = 0
    idx = list(range(len(templates)))
    rng.shuffle(idx)
    pos = 0
    with open(out_path, "wb") as f:
        while n < target_tokens:
            text = templates[idx[pos % len(idx)]]
            pos += 1
            if pos % len(idx) == 0:
                rng.shuffle(idx)
            ids = tokenizer.encode(text.strip())
            if not ids:
                continue
            if n + len(ids) > target_tokens:
                ids = ids[:target_tokens - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
    return n


def main():
    parser = argparse.ArgumentParser(description="Build Phase 7 SFT dataset VERY GOOD 25M+6M OA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--oa-tokens", type=int, default=6_000_000,
                        help="OpenAssistant token budget (0 to disable)")
    parser.add_argument("--no-oa", action="store_true",
                        help="Skip OpenAssistant slice (synthetic only)")
    parser.add_argument("--oa-only", action="store_true",
                        help="Fetch OA only to side files (data/phase7/oa_train.bin + oa_eval.bin), leave train.bin untouched")
    args = parser.parse_args()
    global EOS, tok

    if args.oa_only:
        os.makedirs(DATA_DIR, exist_ok=True)
        tok = Tokenizer(TOKENIZER_PATH)
        EOS = tok.eos_token
        oa_budget = args.oa_tokens
        print(f"Fetching OpenAssistant ONLY (budget {oa_budget:,} tok) -> side files...")
        chat, remem = fetch_oa_instructions(tok, oa_budget, seed=args.seed)
        oa_pairs = chat + remem
        print(f"  OA pairs: chat {len(chat)} remem {len(remem)} total {len(oa_pairs)}")
        if not oa_pairs:
            print("  WARNING: 0 pairs -- nothing written")
            return
        rng = random.Random(args.seed)
        rng.shuffle(oa_pairs)
        oa_eval_pairs = oa_pairs[::12]
        oa_train_pairs = [p for i, p in enumerate(oa_pairs) if i % 12 != 0]
        n = write_docs_cycled(tok, oa_train_pairs, f"{DATA_DIR}/oa_train.bin", int(oa_budget * 11 / 12), rng)
        m = write_docs_cycled(tok, oa_eval_pairs, f"{DATA_DIR}/oa_eval.bin", oa_budget - int(oa_budget * 11 / 12), rng)
        print(f"  oa_train: {n:,} tok  oa_eval: {m:,} tok")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    EOS = tok.eos_token

    scale = (args.smoke / TOTAL_TARGET) if args.smoke > 0 else 1.0
    total = int(TOTAL_TARGET * scale)
    rng = random.Random(args.seed)

    print(f"Generating VERY GOOD SFT data ({total:,} tokens)...")

    # VERY GOOD weights: math-CoT 30%, code 25%, general/GK 25%, reasoning 12%, creative 8%
    math = gen_math_instructions(rng, 120000)
    code = gen_code_instructions(rng, 100000)
    general = gen_general_instructions(rng, 100000)
    reasoning = gen_reasoning_instructions(rng, 60000)
    creative = gen_creative_instructions(rng, 30000)

    pools = [
        (math, 0.30), (code, 0.25), (general, 0.25),
        (reasoning, 0.12), (creative, 0.08),
    ]
    # Build weighted master list for train, separate shuffle for eval
    all_docs = []
    for pool, w in pools:
        k = max(1, int(len(pool) * w * 3))
        all_docs.extend(rng.sample(pool, min(k, len(pool))))
        # top up by random choices to reach weight
        need = int(200000 * w) - k
        if need > 0:
            all_docs.extend([rng.choice(pool) for _ in range(min(need, 200000))])
    rng.shuffle(all_docs)
    print(f"  pools: math {len(math)} code {len(code)} general {len(general)} reasoning {len(reasoning)} creative {len(creative)} master {len(all_docs)}")

    n = write_docs_cycled(tok, all_docs, OUT_TRAIN, int(total * 0.9), rng)
    m = write_docs_cycled(tok, all_docs, OUT_EVAL, int(total * 0.1), rng)

    # OpenAssistant slice: 6M total (3M chat + 3M remembrance), every 12th pair to eval
    oa_train_n = oa_eval_n = 0
    if not args.no_oa and args.oa_tokens > 0:
        oa_budget = int(args.oa_tokens * scale) if args.smoke > 0 else args.oa_tokens
        print(f"Fetching OpenAssistant (oasst1+oasst2 en rank-0 quality-gated, budget {oa_budget:,} tok)...")
        chat, remem = fetch_oa_instructions(tok, oa_budget, seed=args.seed)
        oa_pairs = chat + remem
        if not oa_pairs:
            print("  WARNING: OA fetch returned 0 pairs (offline?) -- synthetic only")
        rng.shuffle(oa_pairs)
        oa_eval_pairs = oa_pairs[::12] if oa_pairs else []
        oa_train_pairs = [p for i, p in enumerate(oa_pairs) if i % 12 != 0]
        print(f"  OA split: train {len(oa_train_pairs)} eval {len(oa_eval_pairs)}")
        if not oa_pairs:
            print(f"  train: {n:,} tok -> {OUT_TRAIN}")
            print(f"  eval: {m:,} tok -> {OUT_EVAL}")
            for p in [OUT_TRAIN, OUT_EVAL]:
                print(f"  {p}: {os.path.getsize(p)/1e6:.1f} MB")
            return
        # append OA train pairs into train bin (top up to +11/12 of budget)
        oa_train_budget = int(oa_budget * 11 / 12)
        oa_train_n = write_docs_cycled(tok, oa_train_pairs or oa_pairs, OUT_TRAIN + ".oa", oa_train_budget, rng)
        # merge .oa into train bin
        with open(OUT_TRAIN, "ab") as fout, open(OUT_TRAIN + ".oa", "rb") as fin:
            fout.write(fin.read())
        os.remove(OUT_TRAIN + ".oa")
        n += oa_train_n
        if oa_eval_pairs:
            oa_eval_budget = oa_budget - oa_train_budget
            oa_eval_n = write_docs_cycled(tok, oa_eval_pairs, OUT_EVAL + ".oa", oa_eval_budget, rng)
            with open(OUT_EVAL, "ab") as fout, open(OUT_EVAL + ".oa", "rb") as fin:
                fout.write(fin.read())
            os.remove(OUT_EVAL + ".oa")
            m += oa_eval_n

    print(f"  train: {n:,} tok -> {OUT_TRAIN} (synth {n-oa_train_n:,} + OA {oa_train_n:,})")
    print(f"  eval: {m:,} tok -> {OUT_EVAL} (synth {m-oa_eval_n:,} + OA {oa_eval_n:,})")
    for p in [OUT_TRAIN, OUT_EVAL]:
        sz = os.path.getsize(p) / 1e6
        print(f"  {p}: {sz:.1f} MB")


if __name__ == "__main__":
    main()
