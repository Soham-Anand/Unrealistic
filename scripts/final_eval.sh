#!/usr/bin/env bash
# Final evaluation battery for the finished model (default: latest checkpoint).
# Runs probe + all infer batteries with locked decode (0.4/0.9/1.25).
#   bash scripts/final_eval.sh [checkpoints/step_XXXXXX]
set -u
CKPT="${1:-$(ls -td checkpoints/step_* | head -1)}"
NAME="$(basename "$CKPT")"
OUT="evals/battery/final_${NAME}"
mkdir -p "$OUT"
echo "Final eval on ${CKPT} -> ${OUT}/"

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/probe_first_token.py \
    --checkpoint "$CKPT" --out "${OUT}/probe.jsonl" 2>&1 | tail -12

infer() { # $1=outfile $2...=prompts (EVAL_TEMP/EVAL_TOPP/EVAL_REPP override)
    local out="$1"; shift
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/spot_infer.py \
        --checkpoint "$CKPT" --temperature "${EVAL_TEMP:-0.4}" --top-p "${EVAL_TOPP:-0.9}" \
        --repetition-penalty "${EVAL_REPP:-1.25}" --max-new-tokens 70 \
        --prompts "$@" > "${OUT}/${out}.txt" 2>&1
    echo "saved ${out}"
}

infer math \
  "What is 8 + 7?" "What is 12 x 4?" "What is the square root of 49?" \
  "What is 2 + 2?" "What is 64 / 8?" "What is 15 - 7?" "What is 7 x 8?" \
  "What is 100 divided by 10?" "What is 9 x 9?" "What is 144 / 12?" \
  "Janet has 3 apples. She gets 5 more. How many?" \
  "A car travels 60 km in 2 hours. What is its speed?" \
  "There are 4 boxes with 6 books each. How many books?" \
  "Linda has 12 candies. She gives 4 away. How many left?" \
  "If 3 workers make 9 toys each, how many toys total?"

infer gk \
  "The capital of France is" "Water is made of hydrogen and" \
  "Photosynthesis produces" "The largest planet in the Solar System is" \
  "Albert Einstein was born in" "The capital of India is" \
  "Who was the first Prime Minister of India?" \
  "Which monument in Agra is one of the wonders of the world?" \
  "What is the national animal of India?" "What is the currency of India?" \
  "In which year did India gain independence?" \
  "Which festival is known as the festival of lights in India?" \
  "What is the opposite of hot is" "Why do we have seasons?" \
  "Complete: The sun rises in the" "DNA is a"

infer code \
  "Write a Python function to add two numbers:" "def factorial(n):" \
  "Write a Python function to check if a number is prime:" \
  "def fibonacci(n):" "What does print(2 + 2) output?"

infer culture \
  "Linkin Park is a" "The lead singer of Linkin Park was" \
  "Chester Bennington was" "In the End is a song by" \
  "The Beatles is a" "Titanic is a film directed by" "Hey Dude"

echo "DONE: ${OUT}/"
