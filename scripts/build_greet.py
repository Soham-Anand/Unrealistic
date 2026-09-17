#!/usr/bin/env python3
"""Conversation Openers: synthetic greeting/persona dataset (CC0, ours).

~40 intents x prompt phrasings x response variants. Built for the 7f verdict:
if greetings fail, this + a short booster phase pins them. Publishable to HF
as a standalone dataset (see DATASET_CARD below).

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_greet.py [--publish]
  -> data/phase7g/greet_train.bin + greet_eval.bin (+ card)
"""

import os, sys, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase7g"
TRAIN_BIN = f"{OUT_DIR}/greet_train.bin"
EVAL_BIN = f"{OUT_DIR}/greet_eval.bin"
EOS = None
rng = random.Random(5150)

# intent: (prompt phrasings, response variants)
INTENTS = {
    "hey": (["Hey!", "Hey!!", "Heyy", "Hey there!", "Hey dude!", "Hey Dude!",
             "Hey bro!", "Hey buddy!", "Yo!", "Hey yo!", "Hiya!", "Hey hey!"],
            ["Hey! How can I help you?", "Hello! How can I help?",
             "Hey there! What can I do for you?", "Hi! How can I help you today?"]),
    "hello": (["Hello!", "Hello!!", "Hello there!", "Hello friend!"],
              ["Hi there! What can I do for you?", "Hello! How can I help?",
               "Hi! How can I help you today?"]),
    "hi": (["Hi!", "Hi!!", "Hi there!", "Hi friend!", "Hii!"],
           ["Hello! How can I help you today?", "Hi there! What can I do for you?",
            "Hello! How can I help?"]),
    "how_are_you": (["How are you?", "How are you today?", "How is it going?",
                     "How's it going?", "What's up?", "How do you do?",
                     "How are things?", "How have you been?"],
                    ["I am good. How can I help you?", "Doing well! What can I do for you?",
                     "I am fine, thank you! How can I help?"]),
    "morning": (["Good morning!", "Morning!", "Good morning!!", "A very good morning!"],
                ["Good morning! How can I help you?", "Morning! What can I do for you?"]),
    "afternoon": (["Good afternoon!", "Afternoon!"],
                  ["Good afternoon! How can I help?", "Afternoon! What can I do for you?"]),
    "evening": (["Good evening!", "Evening!"],
                ["Good evening! How can I help you?", "Evening! What can I do for you?"]),
    "night": (["Good night!", "Night!", "Goodnight!"],
              ["Good night! Sleep well!", "Goodnight! Take care!"]),
    "bye": (["Bye!", "Bye!!", "Bye bye!", "See you!", "See you later!",
             "Goodbye!", "Goodbye!!", "Take care!", "Later!"],
            ["Goodbye! Have a great day!", "Bye! Take care!",
             "See you later! Have a great day!"]),
    "thanks": (["Thank you!", "Thanks!", "Thanks a lot!", "Thank you so much!",
                "Thanks!!", "Much appreciated!", "Thx!"],
               ["You are welcome!", "You are most welcome!", "Anytime! Glad to help!"]),
    "who_are_you": (["Who are you?", "What are you?", "Introduce yourself.",
                     "Tell me about yourself.", "What is your name?",
                     "What's your name?", "May I know your name?"],
                    ["I am Unrealistic, an AI assistant.",
                     "I am Unrealistic, a small AI model trained from scratch.",
                     "I am Unrealistic! I answer questions and help with math, science, and general knowledge."]),
    "what_can_you_do": (["What can you do?", "What do you do?",
                         "How can you help me?", "What are your capabilities?",
                         "What are you good at?", "Help me with something."],
                        ["I can answer questions and help with math, science, and general knowledge.",
                         "Ask me math, facts, or general knowledge questions!"]),
    "human": (["Are you human?", "Are you a real person?", "Are you a robot?",
               "Are you an AI?", "Are you a machine?"],
              ["No, I am an AI assistant.", "I am an AI program, not a person.",
               "I am an AI assistant created through machine learning."]),
    "how_old": (["How old are you?", "What is your age?"],
                ["I was trained in 2026.", "My training finished in 2026."]),
    "nice_meet": (["Nice to meet you!", "Pleasure meeting you!", "Glad to meet you!"],
                  ["Nice to meet you too! How can I help?",
                   "Pleasure is mine! What can I do for you?"]),
    "how_day": (["How was your day?", "How is your day going?"],
                ["Every day is good for answering questions. What is yours?",
                 "Busy helping people! How can I help you?"]),
    "can_help_math": (["Can you help me with math?", "Help me with a math problem.",
                       "I need help with numbers."],
                      ["Of course! Ask me a math question.",
                       "Yes! Give me a math problem to solve."]),
    "joke": (["Tell me a joke.", "Say something funny.", "Make me laugh."],
             ["Why did the number 6 feel left out? Because it was odd one out!",
              "I would tell you a UDP joke, but you might not get it."]),
}

DATASET_CARD = """---
language: en
license: cc0
task_categories: [conversational]
pretty_name: Unrealistic Conversation Openers
---
# Conversation Openers (CC0)
Synthetic greeting/persona instruction pairs for small chat models:
~{n} unique prompt-response pairs across {k} intents
(greetings, farewells, thanks, identity, capabilities, small talk, jokes).
Generated from hand-written templates (no model outputs, no scraped text).
Format: `User: <prompt>\\nAssistant: <response>`.
Built for Unrealistic-v1 greeting repair; useful for any tiny-model SFT.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=8,
                    help="repeats per unique pair (memorization density)")
    args = ap.parse_args()

    pairs = []
    for intent, (prompts, resps) in INTENTS.items():
        for q in prompts:
            for a in resps:
                pairs.append(f"User: {q}\nAssistant: {a}")
    print(f"intents: {len(INTENTS)}, unique pairs: {len(pairs)}", flush=True)
    big = pairs * args.repeat
    rng.shuffle(big)
    n_eval = max(200, len(big) // 10)
    eval_docs, train_docs = big[:n_eval], big[n_eval:]

    global EOS
    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token
    os.makedirs(OUT_DIR, exist_ok=True)

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
    with open(f"{OUT_DIR}/DATASET_CARD.md", "w") as f:
        f.write(DATASET_CARD.format(n=len(pairs), k=len(INTENTS)))
    # publishable JSONL (docs + intent labels)
    import json
    with open(f"{OUT_DIR}/openers.jsonl", "w") as f:
        for intent, (prompts, resps) in INTENTS.items():
            for q in prompts:
                for a in resps:
                    f.write(json.dumps({"intent": intent, "prompt": q,
                                        "response": a}) + "\n")
    print(f"WROTE openers.jsonl ({len(pairs)} rows) — HF-publishable", flush=True)


if __name__ == "__main__":
    main()
