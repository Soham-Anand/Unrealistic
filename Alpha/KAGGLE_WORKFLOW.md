# Alpha 450M — Autonomous 100% Kaggle TPU v5e-8 Guide

This guide walks you through benchmarking and training the **Alpha 450M** model entirely on **Kaggle TPU v5e-8 (ViperLite)** (8 parallel cores, 128GB HBM, ~788 TFLOPS BF16).

---

## 1. Hardware & Performance Overview

| Feature | TPU v5e-8 (ViperLite) |
|---|---|
| **Cores** | 8 TPU Cores running natively in parallel |
| **Precision** | Native `bfloat16` Matrix Multiply Units |
| **Global Batch** | **1,048,576 Tokens / Step** (512 Sequences $\times$ 2048 Context) |
| **Speed (v5e-8)** | **~180,000 – 250,000+ Tokens/Sec** |
| **TPU Session Limit**| **9.0 Hours max runtime** (Session Watchdog cleanly exits at **8.4 Hours**) |
| **8.5B Budget** | **~9.5 to 13.0 Total Wall-Clock Hours (~1 to 2 Sessions!)** |

---

## 2. Step 1: 50M Token Blast-Off Benchmark (3 Minutes)

Test your TPU hardware throughput before starting the full run:
1. Open Kaggle $\to$ Create Notebook $\to$ Upload [`Alpha/notebooks/alpha_throughput_benchmark.ipynb`](file:///Users/sohamanand/Unrealistic/Alpha/notebooks/alpha_throughput_benchmark.ipynb).
2. Set Accelerator to **TPU VM v3-8** and Internet to **ON**.
3. Run All. It will blast through **50,000,000 tokens** across heavy batches and print sustained tokens/sec and exact time estimates.

---

## 3. Step 2: One-Time Kaggle Setup (2 Minutes)

1. **Hugging Face Token**:
   - Go to [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens) and grab a **Write** token.
2. **Add Secret to Kaggle**:
   - In Kaggle Notebook $\to$ **Add-ons** $\to$ **Secrets**.
   - Add label: `HF_TOKEN` with your write token.
   - Check the box to enable it for the notebook.

---

## 4. Step 3: Full 8.5B Autonomous Training Launch

1. Upload [`Alpha/notebooks/alpha_450m_kaggle_tpu_master.ipynb`](file:///Users/sohamanand/Unrealistic/Alpha/notebooks/alpha_450m_kaggle_tpu_master.ipynb) to Kaggle.
2. Set Accelerator to **TPU VM v3-8**.
3. Click **Save Version** $\to$ **Save & Run All (Commit)**.

---

## 5. How Multi-Session Resumption Works (3-Session Schedule)

- **Session 1 (0 $\to$ 2.85B tokens):** Runs for 8.4 hours, automatically pushes checkpoint to `sohamanand/alpha-450m-checkpoints`, and exits before the 9h cap.
- **Session 2 (2.85B $\to$ 5.70B tokens):** Resumes immediately from checkpoint, trains for 8.4 hours, syncs, and exits.
- **Session 3 (5.70B $\to$ 8.50B tokens):** Resumes and completes the 8.5B Near-Chinchilla pretraining run!
- **Total Quota Used:** $\approx 25.2\text{ hours}$ (fits within Kaggle's 20–30h weekly free TPU quota).
