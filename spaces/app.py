"""Unrealistic v1 — ZeroGPU exhibition demo (Science Exhibition build)."""
import os
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
import spaces
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SohamProgrammer/Unrealistic-v1"
MAX_INPUT = 300
HISTORY_TURNS = 3

DTYPE = torch.float16 if torch.cuda.is_available() else torch.bfloat16  # T4 has no bf16
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


@spaces.GPU(duration=20)
def chat(message, history):
    message = (message or "")[:MAX_INPUT]
    turns = []
    for u, a in history[-HISTORY_TURNS:]:
        turns.append(f"User: {u[:MAX_INPUT]}\nAssistant: {a[:400]}")
    prompt = "".join(turns) + f"User: {message}\nAssistant:"
    ids = tok(prompt, return_tensors="pt")["input_ids"][:, -900:]
    if torch.cuda.is_available():
        ids = ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=80, do_sample=True,
            temperature=0.4, top_p=0.9, repetition_penalty=1.25,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:"):].strip()
    return text


STARTERS = [
    "What can you do?",
    "What is 8 + 7?",
    "What is the capital of France?",
    "Who was the lead singer of Linkin Park?",
    "Who built the Taj Mahal?",
    "Write a Python function to add two numbers:",
]

with gr.Blocks(title="Unrealistic v1 — Science Exhibition") as demo:
    gr.Markdown("# Unrealistic v1 — a 190M AI trained from scratch on a MacBook Air")
    gr.Markdown("Built by a 13-year-old student. Try a starter below — free chat may hallucinate "
                "(known limits: biology, Euro-history, big-number word problems). English only.")
    gr.ChatInterface(chat, examples=STARTERS)

demo.launch()
