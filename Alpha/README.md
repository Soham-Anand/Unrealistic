# Alpha 450M — Autonomous 100% TPU-XLA Pipeline

Foundation model pretraining on Kaggle Free Tier **TPU VM v3-8** (8 cores, 128GB HBM) with PyTorch/XLA and stateless SinkGD optimization.

---

## 1. Model & Hardware Architecture

| Property | Value |
|---|---|
| **Parameters** | **450M** (444M effective with tied embeddings) |
| **Layers / Hidden / Intermediate** | 22 Layers / 1280 Hidden / 3456 Intermediate (SwiGLU) |
| **Attention** | 20 Attention Heads, 5 Key-Value Heads (GQA 4:1) + QK-Norm |
| **Optimizer** | **Stateless SinkGD** (2D Weights) + **AdamW** (Embeddings/Norms) |
| **Compute Precision** | **Native `torch.bfloat16`** on TPU Matrix Multiply Units (MXUs) |
| **Training Budget** | **7.0B – 8.5B Tokens** (English, Science, Math Proofs, Python Code) |
| **Hardware** | **Kaggle TPU v5e-8 (ViperLite, 8 Cores, 128GB HBM, ~788 TFLOPS BF16)** |
| **TPU Session Limit** | **9.0 Hours max runtime** (Session Watchdog cleanly exits at **8.4 Hours**) |
| **Global Batch Size** | **1,048,576 Tokens / Step** (512 Sequences $\times$ 2048 Context) |
| **Throughput (v5e-8)** | **~180,000 – 250,000+ Tokens / Second** |
| **Total Training Time** | **~9.5 – 13.0 Total Wall-Clock Hours (~1 to 2 TPU Sessions!)** |

---

## 2. Directory Layout

```text
Alpha/
├── ALPHA_CONSTITUTION.md                 # Master constitutional plan & license governance
├── KAGGLE_WORKFLOW.md                    # TPU v5e-8 execution & resumption guide
├── README.md                             # Project overview and architecture scorecard
├── configs/
│   └── alpha_450m.json                   # Hyperparameters & model architecture spec
└── notebooks/
    ├── alpha_benchmark_tpu.ipynb         # ⚡ 50M Token Blast Benchmark (TPU v5e-8)
    └── alpha_training_tpu.ipynb          # 🚀 Master Self-Contained 8-Core Training Engine (with Muon)
```

---

## 3. Quickstart: Launching on Kaggle TPU

1. **Add Hugging Face Secret**: In Kaggle $\to$ **Add-ons $\to$ Secrets** $\to$ Add `HF_TOKEN` (with Write permissions).
2. **Upload Master Notebook**: Upload [`notebooks/alpha_450m_kaggle_tpu_master.ipynb`](file:///Users/sohamanand/Unrealistic/Alpha/notebooks/alpha_450m_kaggle_tpu_master.ipynb).
3. **Set Accelerator**: In Notebook Settings $\to$ select **TPU VM v3-8** and enable **Internet ON**.
4. **Run All**: Click **Save Version** $\to$ *Save & Run All (Commit)* to run unattended. Checkpoints automatically sync to your private Hugging Face repo every 1,000 steps and at the 11.25-hour watchdog cutoff.
