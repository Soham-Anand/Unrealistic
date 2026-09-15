#!/usr/bin/env python3
"""Phase 8b anchors: PRISTINE short Q->A targets, deterministically synthesized
and verified — no sampling. Repairs wiki-phase damage: Orca-ramble habit,
<<>> artifacts, lost Paris/Jupiter/east, unreachable Linkin Park facts.

Math answers are calculator-verified (safe_math); GK/code are hand-checked.
Each prompt x N phrasings. Output: data/phase8/anchor_train.bin (+eval).

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_anchor.py
"""

import os, sys, re, ast, operator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase8"
TRAIN_BIN = f"{OUT_DIR}/anchor_train.bin"
EVAL_BIN = f"{OUT_DIR}/anchor_eval.bin"
EOS = None

SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _ev(node):
    if isinstance(node, ast.Expression):
        node = node.body
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return SAFE_OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp):
        return SAFE_OPS[type(node.op)](_ev(node.operand))
    raise ValueError


def safe_math(expr: str):
    return _ev(ast.parse(expr.replace("^", "**"), mode="eval"))


# (question, expression-or-literal, display forms)
MATH = [
    ("What is 2 + 2?", "2+2", ["4", "2 + 2 = 4", "4."]),
    ("What is 8 + 7?", "8+7", ["15", "8 + 7 = 15"]),
    ("What is 12 x 4?", "12*4", ["48", "12 x 4 = 48"]),
    ("What is the square root of 49?", "49**0.5", ["7"]),
    ("What is 64 / 8?", "64/8", ["8", "64 / 8 = 8"]),
    ("What is 15 - 7?", "15-7", ["8"]),
    ("What is 7 x 8?", "7*8", ["56"]),
    ("What is 100 divided by 10?", "100/10", ["10"]),
    ("What is 9 x 9?", "9*9", ["81"]),
    ("What is 144 / 12?", "144/12", ["12"]),
    ("What is 3 + 5?", "3+5", ["8"]),
    ("What is 6 x 7?", "6*7", ["42"]),
    ("What is 20 - 7?", "20-7", ["13"]),
    ("What is 2 to the power 3?", "2**3", ["8"]),
    ("What is the square root of 64?", "64**0.5", ["8"]),
    ("What is 5 x 6?", "5*6", ["30"]),
    ("What is 10 + 10?", "10+10", ["20"]),
    ("What is 100 - 42?", "100-42", ["58"]),
    ("If you have 3 apples and get 4 more, how many do you have?", "3+4", ["7"]),
    ("A dozen eggs, and you use 5. How many are left?", "12-5", ["7"]),
]

# (question, short answer) — hand-verified facts
GK = [
    ("What is the capital of France?", "Paris."),
    ("The capital of France is", "Paris."),
    ("Water is made of hydrogen and", "oxygen."),
    ("The largest planet in the Solar System is", "Jupiter."),
    ("What is the opposite of hot?", "Cold."),
    ("Complete: The sun rises in the", "east."),
    ("What is the capital of India?", "New Delhi."),
    ("What is the national animal of India?", "The Bengal tiger."),
    ("What is the currency of India?", "The Rupee."),
    ("In which year did India gain independence?", "1947."),
    ("Which monument in Agra is one of the wonders of the world?", "The Taj Mahal."),
    ("Which river is considered holy in India?", "The Ganga."),
    ("Who wrote the Indian national anthem?", "Rabindranath Tagore."),
    ("Which festival is known as the festival of lights in India?", "Diwali."),
    ("The capital of Japan is", "Tokyo."),
    ("The Nile is the longest", "river."),
    ("Mount Everest is the highest", "mountain."),
    ("The Earth revolves around the", "Sun."),
    ("The Moon orbits the", "Earth."),
    ("Atoms are made of protons, neutrons and", "electrons."),
    ("What planet is known as the Red Planet?", "Mars."),
    ("DNA is a", "molecule."),
    ("Why do we have seasons?", "Earth's axis is tilted, changing sunlight through the year."),
    ("Albert Einstein was born in", "1879."),
    ("The Second World War ended in", "1945."),
]

LP = [
    ("Linkin Park is a", "rock band from the United States, formed in 1996."),
    ("Who was the lead singer of Linkin Park?",
     "Chester Bennington was the lead singer of Linkin Park."),
    ("When was Linkin Park formed?", "Linkin Park was formed in 1996."),
    ("Chester Bennington was", "a singer from the United States, born in 1976."),
    ("In the End is a song by", "Linkin Park."),
]

MUGHAL = [
    ("Who founded the Mughal Empire in India?", "Babur founded the Mughal Empire in 1526."),
    ("Babur was the first", "Mughal emperor of India."),
    ("In which year did Babur win the Battle of Panipat?", "1526."),
    ("Humayun was the son of", "Babur."),
    ("Akbar was a", "Mughal emperor of India."),
    ("Akbar ruled India from 1556 to", "1605."),
    ("Fatehpur Sikri was built by", "Akbar."),
    ("Who built the Taj Mahal?", "Shah Jahan built the Taj Mahal."),
    ("Shah Jahan built the", "Taj Mahal."),
    ("Aurangzeb was the", "sixth Mughal emperor."),
    ("The last Mughal emperor was", "Bahadur Shah Zafar."),
    ("The Mughal Empire began in which year?", "1526."),
    ("Who is Narendra Modi?", "Narendra Modi is the current Prime Minister of India."),
    ("The current Prime Minister of India is", "Narendra Modi."),
    ("Narendra Modi became Prime Minister in", "2014."),
    ("Narendra Modi is the leader of which party?", "The BJP."),
    ("Who created Minecraft?", "Mojang created Minecraft."),
    ("Minecraft was released in", "2011."),
    ("GTA V was developed by", "Rockstar North."),
    ("GTA V was released in", "2013."),
    ("Mario is a character by", "Nintendo."),
    ("Tetris was created by", "Alexey Pajitnov."),
    ("Fortnite was developed by", "Epic Games."),
    ("Pikachu is a", "Pokemon."),
    ("Need for Speed is a", "racing video game series."),
    ("Need for Speed was developed by", "EA."),
    ("Need for Speed Most Wanted was released in", "2005."),
    ("EA makes", "video games."),
    ("What does LOL mean?", "Laughing out loud."),
    ("What does GOAT mean?", "Greatest of all time."),
    ("What is rizz?", "Charm when flirting."),
    ("No cap means", "no lie."),
    ("What does FOMO mean?", "Fear of missing out."),
    ("Cristiano Ronaldo is a", "footballer from Portugal."),
    ("Taylor Swift is a", "singer from the United States."),
    ("Elon Musk is the CEO of", "Tesla."),
    ("MrBeast is a", "YouTuber."),
    ("Virat Kohli is a", "cricketer from India."),
    ("Hinduism's oldest texts are the", "Vedas."),
    ("Diwali is the festival of", "lights."),
    ("Holi is the festival of", "colors."),
    ("The Bhagavad Gita is a", "Hindu scripture."),
    ("Lord Krishna is a", "Hindu god."),
    ("Lord Shiva is the god of", "destruction."),
    ("Mahavira was the 24th", "Tirthankara of Jainism."),
    ("Jainism teaches", "ahimsa."),
    ("The Buddha was born in", "Lumbini."),
    ("Bodh Gaya is important because", "the Buddha attained enlightenment there."),
    ("To cook rice, you", "boil it in water."),
    ("How to make chai?", "Boil tea leaves with milk, sugar, and spices."),
    ("To make an omelette, you need", "eggs."),
    ("Water boils at", "100 degrees Celsius."),
    ("To make roti, you need", "wheat flour and water."),
    ("How to cook pasta?", "Boil it in salted water."),
    ("Dal is made from", "lentils."),
    ("To fry an egg,", "heat oil in a pan."),
    ("CPU stands for", "Central Processing Unit."),
    ("RAM is used for", "temporary memory."),
    ("Python is a", "programming language."),
    ("What does WWW stand for?", "World Wide Web."),
    ("Binary code uses", "zeros and ones."),
    ("Gravity was discovered by", "Isaac Newton."),
    ("Photosynthesis happens in the", "leaves."),
    ("The chemical symbol for gold is", "Au."),
    ("The speed of light is", "300,000 kilometers per second."),
    ("How many planets are in the solar system?", "8."),
    ("The human body has how many bones?", "206."),
    ("The Internet was invented", "by scientists."),
    ("Hey Dude!", "Hey! How can I help you?"),
    ("Hey!", "Hello! How can I help?"),
    ("Hello!", "Hi there! What can I do for you?"),
    ("Hi!", "Hello! How can I help you today?"),
    ("How are you?", "I am good. How can I help you?"),
    ("Good morning!", "Good morning! How can I help you?"),
    ("Bye!", "Goodbye! Have a great day!"),
    ("Thank you!", "You are welcome!"),
    ("What is your name?", "I am Unrealistic, an AI assistant."),
    ("Who are you?", "I am Unrealistic, a small AI model."),
    ("What can you do?", "I can answer questions and help with math, science, and general knowledge."),
]

CODE = [
    ("Write a Python function to add two numbers:",
     "def add(a, b):\n    return a + b"),
    ("def factorial(n):",
     "result = 1\nfor i in range(2, n + 1):\n    result *= i\nreturn result"),
    ("What does print(2 + 2) output?", "4"),
]


def main():
    docs = []
    # math: verify every answer with the calculator, 3 phrasings each
    n_math = 0
    for q, expr, forms in MATH:
        v = safe_math(expr)
        ans = int(v) if float(v) == int(float(v)) else round(float(v), 6)
        for f in forms:
            num = re.findall(r"-?\d+\.?\d*", f)
            assert num, f"no number in form {f!r}"
            assert abs(float(num[-1]) - float(ans)) < 1e-6, f"mismatch {f!r} vs {ans}"
            docs.append(f"User: {q}\nAssistant: {f}")
            n_math += 1
    print(f"math: {len(MATH)} prompts x forms = {n_math} docs, all calculator-verified", flush=True)
    for q, a in GK + LP + MUGHAL + CODE:
        docs.append(f"User: {q}\nAssistant: {a}")
        # statement form too (completion-style grounding)
        if not q.startswith(("User:", "What", "Who", "Why", "Which", "When", "In ", "How",
                             "Complete:", "Write", "If ", "A ")):
            docs.append(f"{q} {a}")
    print(f"total docs: {len(docs)}", flush=True)

    global EOS
    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token
    os.makedirs(OUT_DIR, exist_ok=True)

    # repeat anchors for a tiny-but-dense bin (deterministic order)
    REPEAT = 12
    big = docs * REPEAT
    n_eval = max(50, len(big) // 10)
    eval_docs, train_docs = big[:n_eval], big[n_eval:]

    def write(ds, path):
        n = 0
        with open(path, "wb") as f:
            for d in ds:
                ids = tok.encode(d)
                if not ids:
                    continue
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                f.write(np.uint16(EOS).tobytes())
                n += len(ids)
        return n

    nt = write(train_docs, TRAIN_BIN)
    ne = write(eval_docs, EVAL_BIN)
    print(f"WROTE train {nt:,} tokens ({len(train_docs)} docs) -> {TRAIN_BIN}", flush=True)
    print(f"WROTE eval {ne:,} tokens -> {EVAL_BIN}", flush=True)


if __name__ == "__main__":
    main()
