# Unrealistic v1 — Architecture

Dict-config MLX transformer, LLaMA family. No framework magic: every tensor
is an explicit dict entry (`src/model/transformer.py`).

## Config (`configs/model_config.json`)

| Parameter | Value |
|---|---|
| Params (counted) | 189,748,992 |
| `hidden_size` | 768 |
| `intermediate_size` | 4096 (SwiGLU → 2 matrices) |
| `num_hidden_layers` | 14 |
| `num_attention_heads` / `num_key_value_heads` | 8 / 8 (full MHA) |
| `max_position_embeddings` | 1024 |
| `rope_theta` | 10000.0, no scaling |
| `hidden_act` | silu |
| `rms_norm_eps` | 1e-6 (pre-norm: attn_norm + ffn_norm per layer, final_norm) |
| `vocab_size` | 32000 (SentencePiece BPE, `data/tokenizer/phase1.model`) |
| `tie_word_embeddings` | true (lm_head = embedding transpose) |
| `initializer_range` | 0.02, `dropout` 0.1 (train only) |
| Train dtype | bfloat16 |

Param tree: `embeddings.token_embedding`, per-layer
`attn_norm/ffn_norm/attn.{q,k,v,o}/ffn.{gate,up,down}`, `final_norm`
(no separate `lm_head` — tied).

## Training setup

- Optimizer: AdamW (MLX), β 0.9/0.999, wd 0.1, clip 1.0, cosine + warmup.
  `--reset-optimizer` at phase starts, kept on mid-phase resumes.
- Batch 1 × grad-accum (8 @seq1024 → 8192 tok/step; 4 @seq512 → 2048).
- Gradient checkpointing on. Checkpoint every 500 steps (200 for short
  phases) + hourly, keep_last 3. **Final save lands at max_steps−1**
  (loop `range(step, max)` then saves last iterated step) — all downstream
  gates accept `≥ FINAL−1` and use the actual latest dir.
- Inference: `src/inference/generate.py` (temp/top-p/top-k/rep-penalty).

## Export map (MLX npz → HF → GGUF)

HF `LlamaForCausalLM` layout: `model.embed_tokens`,
per-layer `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`input_layernorm/post_attention_layernorm`, `model.norm`.
(`scripts/export_pt.py`, post-training.)
