# Unrealistic Alpha (450M) — local home

Kaggle is the execution environment; this directory is the project home.
Constitution: `Alpha/ALPHA_CONSTITUTION.md` (repo root).

```text
alpha/
├── notebooks/alpha_kaggle_01_04.ipynb  # upload to Kaggle, GPU ON, Run All
├── src/model/alpha_lm.py               # 450M LLaMA-style + tiny proxy
├── scripts/build_tokenizer.py          # SPM-48K trainer (mirror of Cell 02)
├── configs/alpha_450m.json             # arch + hyperparams (hash-pinned in runs)
└── manifests/                          # provenance records (filled by pipeline)
```

Notebook cells are self-contained (inline model) so Kaggle runs with zero
setup friction; these files are the versioned source of truth.
