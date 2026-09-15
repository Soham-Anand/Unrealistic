#!/usr/bin/env python3
"""Build the "Realization" dataset — teaches the model to JUDGE truth, not
just produce answers, across math AND science (biology/chem/physics/earth),
plus revitalizes language by mixing in real Code, English, and General
Knowledge.

The name mirrors the curriculum intent: confronting the model with its own
confident-but-wrong outputs (which soham calls "HALLUCINATION") so it learns
to recognized that a plausible-sounding claim can be false, detect the
contradiction, and correct it.

Phase placement: after Prabhakar Stage C (296,584) and before Phase 6 Refresh.

Three core judgement tasks, each applied to math and science facts:

  1. NEGATION      : the true fact is presented, then immediately negated
                     ("2 + 2 = 5 is incorrect, because 2 + 2 = 4").
  2. CONTRADICTION : two competing claims are given; the model must pick the
                     true one and say why the other is false.
  3. CORRECTION    : a wrong claim (a confabulation) is given; the model must
                     produce the corrected right answer and explain the error.

A per-sample 'Role' line tells the model which task it is performing, so the
phase teaches clean, structured judgement language (in contrast to the
bare-number / code-collapse outputs that plagued Stage A).

Generated samples are tokenized and written to
  data/phase6/realization_<label>_syn.bin

The rest of the ~30M budget is drawn at doc boundaries from the EXISTING real
bins (see MIX), which restores clean English, code, and general knowledge
that the math-heavy stages suppress.

Usage:
  python3 scripts/generate_realization_data.py                 # full build
  python3 scripts/generate_realization_data.py --validate 200  # dry-run QC
  python3 scripts/generate_realization_data.py --smoke 200000  # tiny build
"""

import os, sys, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase6"
MATH_GEN_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
EOS = None

# ─────────────────────────────────────────────────────────────────────
# Source of REAL content (code / english / general knowledge). We slice
# these existing token bins at EOS doc boundaries (see build_phase5_curriculum).
# They need no download and provide genuine language + code recovery.
# ─────────────────────────────────────────────────────────────────────
REAL_BINS = {
    # (label, path, target_tokens) — VERY GOOD 24M real recovery (8M each)
    "code":    ("data/phase4/train.bin", 8_000_000),  # StarCoder code + StackExchange
    "english": ("data/phase2/fwe_train.bin", 8_000_000),  # FineWeb-Edu prose
    "gk":      ("data/phase3/train.bin", 8_000_000),      # Wikipedia general knowledge
}

# ─────────────────────────────────────────────────────────────────────
# Synthetic judgement budget (the remainder of the ~30M after real slice).
# We allocate across the three tasks and two domains.
# ─────────────────────────────────────────────────────────────────────
EVAL_TOKENS = 1_500_000
PER_DOC_TOKENS = {
    "negation":    60,
    "contradiction": 60,
    "correction":  60,
}


# ─────────────────────────────────────────────────────────────────────
# Curated science fact bank. Each entry: (domain, question, correct answer,
# one or more plausible WRONG answers / confabulations).
# The wrong answers are common misconceptions or near-miss confusions so the
# model learns to catch subtle errors, not just obvious nonsense.
# ─────────────────────────────────────────────────────────────────────
SCIENCE_FACTS = [
    # ── Biology ───────────────────────────────────────────────────
    ("biology", "What is the powerhouse of the cell?",
     "the mitochondrion", ["the nucleus", "the ribosome", "the cell membrane"]),
    ("biology", "What molecule carries genetic information in living cells?",
     "DNA", ["protein", "ATP", "glucose"]),
    ("biology", "Which organ pumps blood around the human body?",
     "the heart", ["the lungs", "the liver", "the brain"]),
    ("biology", "What process do plants use to make food from sunlight?",
     "photosynthesis", ["respiration", "digestion", "osmosis"]),
    ("biology", "How many chambers does the human heart have?",
     "four", ["two", "three", "five"]),
    ("biology", "What gas do plants absorb from the air for photosynthesis?",
     "carbon dioxide", ["oxygen", "nitrogen", "hydrogen"]),
    ("biology", "Which blood cells fight infection?",
     "white blood cells", ["red blood cells", "platelets", "plasma"]),
    ("biology", "What is the basic structural unit of all living organisms?",
     "the cell", ["the atom", "the tissue", "the organ"]),
    ("biology", "Which organ pumps and filters blood?",
     "the kidneys filter blood", ["the heart pumps blood", "the liver pumps blood", "the lungs filter blood"]),
    ("biology", "What is the main function of red blood cells?",
     "to carry oxygen", ["to fight infection", "to clot blood", "to digest food"]),
    ("biology", "Which part of the plant takes in water and nutrients from soil?",
     "the roots", ["the leaves", "the flowers", "the stem"]),
    ("biology", "What is the largest organ of the human body?",
     "the skin", ["the liver", "the brain", "the lungs"]),
    ("biology", "Which system breaks down food into nutrients?",
     "the digestive system", ["the respiratory system", "the circulatory system", "the nervous system"]),
    ("biology", "What is the powerhouse energy molecule used by cells?",
     "ATP", ["DNA", "RNA", "protein"]),
    ("biology", "Which organ protects the brain?",
     "the skull", ["the spine", "the ribs", "the pelvis"]),

    # ── Chemistry ─────────────────────────────────────────────────
    ("chemistry", "What is the chemical symbol for water?",
     "H2O", ["CO2", "O2", "NaCl"]),
    ("chemistry", "What is the chemical symbol for oxygen?",
     "O", ["O2 as an element symbol", "N", "C"]),
    ("chemistry", "What is the chemical symbol for gold?",
     "Au", ["Ag", "Gd", "Go"]),
    ("chemistry", "What is the chemical symbol for sodium chloride (table salt)?",
     "NaCl", ["KCl", "NaOH", "HCl"]),
    ("chemistry", "What is the chemical symbol for carbon dioxide?",
     "CO2", ["CO", "C2O", "CeO2"]),
    ("chemistry", "What is the most abundant gas in Earth's atmosphere?",
     "nitrogen", ["oxygen", "carbon dioxide", "argon"]),
    ("chemistry", "Which element has the chemical symbol 'Fe'?",
     "iron", ["gold", "fluorine", "francium"]),
    ("chemistry", "What is the chemical symbol for carbon?",
     "C", ["Ca", "Cu", "Cl"]),
    ("chemistry", "What is the chemical symbol for helium?",
     "He", ["H", "Ha", "Ho"]),
    ("chemistry", "What gas do humans breathe in to stay alive?",
     "oxygen", ["carbon dioxide", "nitrogen", "helium"]),
    ("chemistry", "What happens to water at 100 degrees Celsius?",
     "it boils", ["it freezes", "it evaporates instantly", "it becomes solid"]),
    ("chemistry", "What is a molecule made of two hydrogen atoms and one oxygen atom?",
     "water", ["hydrogen peroxide", "methane", "ammonia"]),
    ("chemistry", "Which state of matter has a fixed shape and volume?",
     "solid", ["liquid", "gas", "plasma"]),
    ("chemistry", "What is the chemical symbol for nitrogen?",
     "N", ["Ni", "No", "Na"]),
    ("chemistry", "Which gas is produced by photosynthesis and released into the air?",
     "oxygen", ["carbon dioxide", "nitrogen", "methane"]),

    # ── Physics ───────────────────────────────────────────────────
    ("physics", "What is the force that pulls objects toward the Earth?",
     "gravity", ["magnetism", "friction", "tension"]),
    ("physics", "What is the unit of force?",
     "the newton", ["the joule", "the watt", "the volt"]),
    ("physics", "What is the speed of light in a vacuum?",
     "about 300,000 kilometres per second", ["about 3,000 kilometres per second", "about 30,000 kilometres per second", "about 300 kilometres per second"]),
    ("physics", "What is the unit of energy?",
     "the joule", ["the newton", "the ampere", "the pascal"]),
    ("physics", "What is the force that slows a moving object down?",
     "friction", ["gravity", "inertia", "magnetism"]),
    ("physics", "What is the unit of electric current?",
     "the ampere", ["the volt", "the ohm", "the watt"]),
    ("physics", "What is the unit of electrical resistance?",
     "the ohm", ["the volt", "the ampere", "the farad"]),
    ("physics", "What is the unit of electric potential difference (voltage)?",
     "the volt", ["the ohm", "the ampere", "the watt"]),
    ("physics", "What keeps a moving object moving if no force acts on it?",
     "inertia", ["friction", "gravity", "magnetism"]),
    ("physics", "What is the opposite of a push in physics?",
     "a pull", ["a twist", "a lift", "a drop"]),
    ("physics", "What tool measures temperature?",
     "a thermometer", ["a barometer", "a ruler", "a scale"]),
    ("physics", "Sound travels fastest in which state of matter?",
     "a solid", ["a liquid", "a gas", "a vacuum"]),
    ("physics", "What is the measure of how much matter an object has?",
     "mass", ["weight", "volume", "density"]),
    ("physics", "What is the main source of energy for the planet Earth?",
     "the Sun", ["the Moon", "Earth's core", "the stars"]),
    ("physics", "What charge does an electron carry?",
     "a negative charge", ["a positive charge", "no charge", "a neutral charge"]),

    # ── Earth / space ─────────────────────────────────────────────
    ("earth", "What is the largest planet in our solar system?",
     "Jupiter", ["Saturn", "Earth", "Neptune"]),
    ("earth", "What planet is known as the Red Planet?",
     "Mars", ["Venus", "Jupiter", "Mercury"]),
    ("earth", "What is the closest planet to the Sun?",
     "Mercury", ["Venus", "Earth", "Mars"]),
    ("earth", "What planet do we live on?",
     "Earth", ["Mars", "Venus", "Jupiter"]),
    ("earth", "What is the star at the center of our solar system?",
     "the Sun", ["the Moon", "Sirius", "the North Star"]),
    ("earth", "What is the only natural satellite of Earth?",
     "the Moon", ["the Sun", "Mars", "Venus"]),
    ("earth", "About how long does Earth take to orbit the Sun?",
     "one year", ["one day", "one month", "one week"]),
    ("earth", "How long does Earth take to rotate once on its axis?",
     "about 24 hours", ["about 12 hours", "about one month", "about one year"]),
    ("earth", "What are the three states of matter?",
     "solid, liquid and gas", ["solid and liquid only", "liquid, gas and plasma", "solid, gas and ice"]),
    ("earth", "Where is the Sahara Desert located?",
     "in Africa", ["in Asia", "in Australia", "in South America"]),
    ("earth", "What is the largest ocean on Earth?",
     "the Pacific Ocean", ["the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean"]),
    ("earth", "What is the tallest mountain on Earth above sea level?",
     "Mount Everest", ["Mount Kilimanjaro", "Mount Fuji", "Denali"]),
    ("earth", "Which planet is known as the Morning Star?",
     "Venus", ["Mars", "Mercury", "Saturn"]),
    ("earth", "What is the outermost layer of the Earth?",
     "the crust", ["the mantle", "the core", "the magma"]),
     ("earth", "What keeps planets in orbit around the Sun?",
      "gravity", ["magnetism", "wind", "light"]),
 ]


GK_FACTS = [
    ("culture", "What is the capital of France?", "Paris", ["London", "Berlin", "Rome"]),
    ("culture", "Who wrote Romeo and Juliet?", "William Shakespeare", ["Charles Dickens", "Jane Austen", "Mark Twain"]),
    ("culture", "What is the largest planet in our solar system?", "Jupiter", ["Saturn", "Earth", "Mars"]),
    ("culture", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Picasso", "Rembrandt"]),
    ("culture", "What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Seoul"]),
    ("culture", "Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Isaac Newton"]),
    ("culture", "What is the chemical symbol for gold?", "Au", ["Ag", "Fe", "Cu"]),
    ("history", "Who was the first president of the United States?", "George Washington", ["Thomas Jefferson", "Abraham Lincoln", "John Adams"]),
    ("history", "In which year did World War II end?", "1945", ["1918", "1939", "1965"]),
    ("history", "Where was Napoleon born?", "Corsica", ["Paris", "Vienna", "Rome"]),
    ("history", "Who invented the telephone?", "Alexander Graham Bell", ["Thomas Edison", "Nikola Tesla", "Guglielmo Marconi"]),
    ("history", "What empire built the Colosseum?", "the Roman Empire", ["the Greek Empire", "the Egyptian Empire", "the Persian Empire"]),
    ("geography", "What is the largest ocean on Earth?", "the Pacific Ocean", ["the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean"]),
    ("geography", "What desert covers much of northern Africa?", "the Sahara Desert", ["the Gobi Desert", "the Arabian Desert", "the Kalahari Desert"]),
    ("geography", "Which country is known as the Land of the Rising Sun?", "Japan", ["China", "South Korea", "Thailand"]),
    ("geography", "What is the longest river in the world?", "the Nile River", ["the Amazon River", "the Yangtze River", "the Mississippi River"]),
    ("code", "What does the Python function len() do?", "it returns the length of an object", ["it deletes an object", "it sorts an object", "it converts an object to a string"]),
    ("code", "Which keyword defines a function in Python?", "def", ["func", "function", "define"]),
    ("code", "What is the correct way to create a list in Python?", "my_list = [1, 2, 3]", ["my_list = (1, 2, 3)", "my_list = {1, 2, 3}", "my_list = <1, 2, 3>"]),
    ("code", "What does HTML stand for?", "HyperText Markup Language", ["HighText Machine Language", "Hyper Transfer Markup Language", "Home Tool Markup Language"]),
 ]


def gen_science_task(rng, n):
    """Produce negations / contradictions / corrections over the science bank."""
    templates = []
    for _ in range(n):
        fact = rng.choice(SCIENCE_FACTS)
        domain, q, correct, wrongs = fact
        wrong = rng.choice(wrongs)
        task = rng.choice(["negation", "contradiction", "correction"])
        if task == "negation":
            fmt = (
                f"Science fact ({domain}): {q}\n"
                f"The correct answer is {correct}.\n"
                f"If someone claims the answer is {wrong}, that is incorrect. "
                f"The true answer is {correct}, not {wrong}."
            )
        elif task == "contradiction":
            fmt = (
                f"Science fact ({domain}): {q}\n"
                f"One person says the answer is {correct}. Another says it is {wrong}.\n"
                f"Which claim is true, and why is the other false?\n"
                f"The true answer is {correct}, not {wrong}. The claim {wrong} is false."
            )
        else:  # correction
            fmt = (
                f"Someone answered incorrectly.\n"
                f"Claim: {q} The answer is {wrong}.\n"
                f"That answer is wrong. The correct answer is {correct}, not {wrong}."
            )
        templates.append(fmt)
    return templates


def gen_gk_task(rng, n):
    """Negations / contradictions / corrections over GK/code facts."""
    templates = []
    for _ in range(n):
        fact = rng.choice(GK_FACTS)
        domain, q, correct, wrongs = fact
        wrong = rng.choice(wrongs)
        task = rng.choice(["negation", "contradiction", "correction"])
        if task == "negation":
            fmt = (
                f"GK fact ({domain}): {q}\n"
                f"The correct answer is {correct}.\n"
                f"If someone claims the answer is {wrong}, that is incorrect. "
                f"The true answer is {correct}, not {wrong}."
            )
        elif task == "contradiction":
            fmt = (
                f"GK fact ({domain}): {q}\n"
                f"One person says the answer is {correct}. Another says it is {wrong}.\n"
                f"Which claim is true, and why is the other false?\n"
                f"The true answer is {correct}, not {wrong}. The claim {wrong} is false."
            )
        else:
            fmt = (
                f"Someone answered incorrectly.\n"
                f"Claim: {q} The answer is {wrong}.\n"
                f"That answer is wrong. The correct answer is {correct}, not {wrong}."
            )
        templates.append(fmt)
    return templates


def gen_math_task(rng, n):
    """Negations / contradictions / corrections over simple arithmetic facts."""
    templates = []
    for _ in range(n):
        kind = rng.choice(["add", "sub", "mul", "div"])
        if kind == "add":
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            correct, op = a + b, "+"
        elif kind == "sub":
            a = rng.randint(10, 100); b = rng.randint(1, a - 1)
            correct, op = a - b, "-"
        elif kind == "mul":
            a, b = rng.randint(2, 15), rng.randint(2, 15)
            correct, op = a * b, "×"
        else:
            b = rng.randint(1, 12); q = rng.randint(1, 12); a = b * q
            correct, op = q, "÷"
        wrong = correct + rng.choice([-1, 1, 2, -2])
        wrong = max(0, wrong)
        expr = f"{a} {op} {b}"
        task = rng.choice(["negation", "contradiction", "correction"])
        if task == "negation":
            fmt = (
                f"Math fact: {expr} = ?\n"
                f"The correct answer is {correct}.\n"
                f"If someone claims {expr} = {wrong}, that is incorrect, because {expr} = {correct}."
            )
        elif task == "contradiction":
            fmt = (
                f"Math fact: {expr} = ?\n"
                f"One person says {expr} = {correct}. Another says {expr} = {wrong}.\n"
                f"Which is true, and why is the other false?\n"
                f"The true answer is {correct}, because {expr} = {correct}. "
                f"The claim {expr} = {wrong} is false."
            )
        else:
            fmt = (
                f"Someone answered incorrectly.\n"
                f"Claim: {expr} = {wrong}.\n"
                f"That answer is wrong. The correct answer is {expr} = {correct}."
            )
        templates.append(fmt)
    return templates


# generator registry with token-per-doc estimates — VERY GOOD 9 tasks all-domain
TASKS = [
    ("negation",        gen_science_task, 60),
    ("negation_m",      gen_math_task,    60),
    ("negation_gk",     gen_gk_task,      60),
    ("contradiction",   gen_science_task, 60),
    ("contradiction_m", gen_math_task,    60),
    ("contradiction_gk", gen_gk_task,     60),
    ("correction",      gen_science_task, 60),
    ("correction_m",    gen_math_task,    60),
    ("correction_gk",   gen_gk_task,      60),
]

TASK_MIX = {
    "negation": 0.15, "negation_m": 0.12, "negation_gk": 0.10,
    "contradiction": 0.15, "contradiction_m": 0.12, "contradiction_gk": 0.10,
    "correction": 0.15, "correction_m": 0.06, "correction_gk": 0.05,
}


def write_docs(templates, tok, out_path, target_tokens):
    n = 0
    with open(out_path, "wb") as f:
        for text in templates:
            ids = tok.encode(text.strip())
            if not ids:
                continue
            if n + len(ids) > target_tokens:
                ids = ids[:target_tokens - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
            if n >= target_tokens:
                break
    return n


def eos_indices(path):
    indices = []
    offset = 0
    with open(path, "rb") as f:
        while True:
            chunk = np.frombuffer(f.read(1 << 28), dtype=np.uint16)
            if chunk.size == 0:
                break
            idx = np.nonzero(chunk == EOS)[0]
            indices.extend((offset + idx).tolist())
            offset += chunk.size
    if not indices:
        return np.zeros(0, dtype=np.int64)
    return np.array(indices, dtype=np.int64)


def iter_docs(path):
    with open(path, "rb") as f:
        carry = np.zeros(0, dtype=np.uint16)
        while True:
            chunk = f.read(1 << 30)
            if not chunk:
                break
            arr = np.concatenate([carry, np.frombuffer(chunk, dtype=np.uint16)])
            carry = np.zeros(0, dtype=np.uint16)
            start = 0
            for i, v in enumerate(arr):
                if v == EOS:
                    yield arr[start:i].tolist()
                    start = i + 1
            if start < len(arr):
                carry = arr[start:]


def merge_sources(sources, out_path, seed=42):
    rng = random.Random(seed)
    active = list(sources)
    iters = {p: iter(iter_docs(p)) for p, _ in sources}
    remaining = {p: t for p, t in sources}
    total_used = 0
    t0 = time.time()
    with open(out_path, "wb") as out:
        while active:
            weights = [max(1, remaining[p]) for p, _ in active]
            chosen = rng.choices([p for p, _ in active], weights=weights, k=1)[0]
            try:
                doc = next(iters[chosen])
            except StopIteration:
                active = [s for s in active if s[0] != chosen]
                iters.pop(chosen, None)
                remaining.pop(chosen, None)
                continue
            out.write(np.array(doc, dtype=np.uint16).tobytes())
            out.write(np.uint16(EOS).tobytes())
            remaining[chosen] -= len(doc)
            total_used += len(doc)
            if remaining[chosen] <= 0:
                active = [s for s in active if s[0] != chosen]
                iters.pop(chosen, None)
                remaining.pop(chosen, None)
    print(f"  merged {total_used:,} tok -> {os.path.basename(out_path)}  {time.time()-t0:.0f}s")


def slice_head(path, target, out_path):
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    seps = eos_indices(path)
    n = 0
    with open(out_path, "wb") as f:
        start = 0
        for se in seps:
            lo, hi = int(start), int(se)
            if hi - lo == 0:
                start = se + 1
                continue
            f.write(arr[lo:hi].tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += hi - lo
            start = se + 1
            if n >= target:
                break
    del arr
    print(f"  eval -> {n:,} tok -> {os.path.basename(out_path)}")


def main():
    ap = argparse.ArgumentParser(description="Build the Realization dataset (Phase 5.x)")
    ap.add_argument("--tokens", type=int, default=60_000_000,
                    help="Total Realization token budget (VERY GOOD 60M all-domain)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", type=int, default=0,
                    help="QC each task generator with N samples then exit")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Scale all budgets to this many total tokens")
    ap.add_argument("--skip-real", action="store_true",
                    help="Skip real bin slicing (test generated only)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token
    rng = random.Random(args.seed)

    if args.validate > 0:
        ok = True
        for label, gen, _ in TASKS:
            docs = gen(rng, args.validate)
            empty = sum(1 for d in docs if not d or not d.strip())
            dup = len(docs) - len(set(docs))
            nojudge = sum(1 for d in docs
                          if not any(k in d for k in ("incorrect", "true answer",
                                                       "is wrong", "false",
                                                       "correct answer")))
            print(f"{label:18s} n={len(docs):3d} empty={empty} dup={dup} no-judge={nojudge}")
            if empty:
                ok = False
            for s in docs[:1]:
                print(f"    EX>> {s[:110]!r}")
        print("ALL REALIZATION GENERATORS OK" if ok else "ISSUES DETECTED")
        sys.exit(0 if ok else 1)

    total = args.smoke if args.smoke > 0 else args.tokens
    scale = total / args.tokens if args.smoke > 0 else 1.0
    tag = "SMOKE" if scale < 1.0 else "FULL"
    print(f"[{tag}] Realization build  scale={scale:.4f}  total={total:,} tok")

    # Budget split: real recovery vs synthetic judgement.
    # Balanced mix per design: code ~13%, english ~13%, gk ~13% of total.
    real_share = 0.4
    real_budget = int(total * real_share)
    synth_budget = total - real_budget
    print(f"  real slice budget = {real_budget:,} tok (code/english/gk)")
    print(f"  synthetic judgement budget = {synth_budget:,} tok (negation/contradict/correct)")

    # 1) Real content slices (doc-delimited from existing bins)
    raw_parts = []
    if not args.skip_real:
        print("\nSlicing REAL content (code / english / gk)...")
        for label, (path, base_target) in REAL_BINS.items():
            tgt = int(base_target * scale) if args.smoke > 0 else int(base_target * (real_budget / 12_000_000))
            if not os.path.exists(path):
                print(f"    WARNING: missing real bin {path} -- skipping")
                continue
            raw = f"{DATA_DIR}/realization_real_{label}.bin"
            n = slice_real_bins(path, tgt, args.seed, raw)
            print(f"  {label:8s} {n:>12,}/{tgt:,} tok -> {os.path.basename(raw)}")
            if n > 0:
                raw_parts.append((raw, tgt))

    # 2) Synthetic judgement content
    print("\nGenerating synthetic judgement tasks...")
    wsum = sum(TASK_MIX.values())
    for label, gen, approx in TASKS:
        w = TASK_MIX[label]
        budget = int(synth_budget * w / wsum)
        n_docs = max(1, budget // approx)
        docs = gen(rng, n_docs)
        out_path = f"{DATA_DIR}/realization_{label}_syn.bin"
        n = write_docs(docs, tok, out_path, budget)
        print(f"  {label:18s} {n:>12,}/{budget:,} tok -> {os.path.basename(out_path)}")
        raw_parts.append((out_path, budget))

    # 3) Merge everything into one train bin
    train_path = f"{DATA_DIR}/realization_train.bin"
    print(f"\nMerging all {len(raw_parts)} parts -> realization_train.bin ...")
    merge_sources(raw_parts, train_path, seed=args.seed)

    # 4) Eval head
    eval_path = f"{DATA_DIR}/realization_eval.bin"
    print("Slicing eval head...")
    slice_head(train_path, max(1, int(EVAL_TOKENS * scale)), eval_path)

    print("\nRealization dataset build complete:")
    print(f"  {train_path} (train)")
    print(f"  {eval_path} (eval)")


def slice_real_bins(path, target, seed, out_path):
    """Slice N tokens worth of whole docs from a real bin."""
    import random as _r
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    seps = eos_indices(path)
    n_docs = len(seps)
    if n_docs == 0:
        del arr
        return 0
    starts = np.concatenate(([-1], seps[:-1] + 1))
    rng = _r.Random(seed)
    start_doc = rng.randrange(n_docs)
    target = max(1, int(target))
    n = 0
    with open(out_path, "wb") as f:
        for i in range(n_docs):
            di = (start_doc + i) % n_docs
            lo, hi = int(starts[di]), int(seps[di])
            f.write(arr[lo:hi].tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += hi - lo
            if n >= target:
                break
    del arr
    return n


if __name__ == "__main__":
    main()
