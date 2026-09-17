"""Unrealistic v1 — ZeroGPU Gradio chat demo."""
import spaces
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "SohamProgrammer/Unrealistic-v1"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


@spaces.GPU
def chat(message, history):
    turns = []
    for u, a in history:
        turns.append(f"User: {u}\nAssistant: {a}")
    prompt = "".join(turns) + f"User: {message}\nAssistant:"
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    if torch.cuda.is_available():
        ids = ids.cuda()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=120, do_sample=True,
            temperature=0.4, top_p=0.9, repetition_penalty=1.25,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    chat,
    title="Unrealistic v1 (190M, from-scratch)",
    description="Tiny LLM trained from zero on a MacBook Air. Try math, capitals, Linkin Park, greetings.",
    examples=["What is 8 + 7?", "The capital of France is",
              "Who was the lead singer of Linkin Park?", "Hey Dude!"],
)
demo.launch()
