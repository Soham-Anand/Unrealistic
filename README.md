# Unrealistic v1 — 190M LLM trained from scratch on a MacBook Air

A 190M-parameter decoder-only transformer language model, trained from zero
on an Apple M4 MacBook Air (16GB) with MLX. No pretraining shortcuts: every
weight comes from this repo's curriculum, ~356K steps / ~1.1B tokens.

**Final model: `checkpoints/step_355789`** (also `baselines/post_final`).

## Quickstart

```bash
# chat / probe (MLX, Apple Silicon)
/opt/homebrew/bin/python3 scripts/spot_infer.py \
    --checkpoint checkpoints/step_355789 \
    --temperature 0.4 --top-p 0.9 --repetition-penalty 1.25 \
    --prompts "What is 8 + 7?" "The capital of France is"

# full final battery (probe + math/GK/code/culture)
bash scripts/final_eval.sh checkpoints/step_355789

# standard benchmarks (MLX-direct, no torch)
PYTHONPATH=. python3 scripts/bench.py \
    --checkpoint checkpoints/step_355789 --out evals/benchmarks/final
```

Recommended decode: `temperature 0.4, top_p 0.9, repetition_penalty 1.25`.

## What it can do (measured on final)

| Area | Score |
|---|---|
| Math free-gen (15 Qs) | 10/15 |
| General knowledge (16 Qs) | 16/16 |
| Code (5 Qs) | 3/5 |
| Culture incl. greetings (7 Qs) | 4/7 |
| First-token probes (40) | 20/40 top-3 |

Honest limits: big-number word problems, deep code, biology, multi-step
reasoning, and judgment tasks are weak or absent. See `docs/EVAL_REPORT.md`.

## Layout

```text
src/            model (transformer), data, training (AdamW trainer, DPO), inference
scripts/        builders, train.py, dpo.py, rft.py, spot_infer.py, probe_first_token.py,
                bench.py, final_eval.sh, run_curriculum.sh, handoff/watch scripts
configs/        model_config.json + per-phase training configs
data/           token bins per phase (git-ignored; see docs/DATA_REPORT.md)
checkpoints/    step checkpoints (git-ignored)
baselines/      one archived checkpoint per phase boundary (git-ignored)
evals/          battery history, probes, benchmark results
docs/           ARCHITECTURE, TRAINING_REPORT, DATA_REPORT, EVAL_REPORT
Alpha/          Unrealistic Alpha 450M constitution (separate project)
TIMELINE.md     project history with step spine
```

## Training chain (absolute steps)

Stage C 296,583 → CoT 306,583 → Refresh 330,790 → Realization 336,790 →
SFT 346,790 → Wiki 349,790 → SFT-refresh 351,789 → DPO 352,789 →
diverse-SFT 354,790 → anchors 355,790 (**final**).

Details: `docs/TRAINING_REPORT.md`. Data + licenses: `docs/DATA_REPORT.md`.

## License

- Code: MIT (`LICENSE`)
- Model weights: Apache-2.0 (`LICENSE_WEIGHTS`)
- Docs: CC BY 4.0
