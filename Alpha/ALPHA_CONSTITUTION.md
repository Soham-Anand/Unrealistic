# Unrealistic Alpha — Constitution & Master Plan

> **Status:** FROZEN (spec) · DEFERRED (execution until exams over + v1 ships)
> **Last updated:** 2026-09-14
> **Companion:** `ALPHA_PLAN.md` (original 9B sketch — superseded by this document)

---

## 1. Identity

**Unrealistic Alpha** — 450M-parameter research + education foundation model.
Strong scientific reasoning, excellent code, excellent English, broad factual
knowledge. Deliberately **not** a generic internet-trained chatbot.

Success = maximizing **useful knowledge learned per clean, legally usable token**.

---

## 2. Architecture

| Property | Value |
|---|---|
| Params | 450M |
| Family | LLaMA-style decoder (RMSNorm, SiLU/SwiGLU, RoPE, tied embeddings) |
| Context | 2048 tokens |
| Vocab | 45–50K SentencePiece Unigram (EN 40% / code 40% / math 20%) |
| Framework | **PyTorch + HF transformers** (Kaggle NVIDIA; MLX is Apple-only) |
| Optimizer (constitutional) | **AdamW** |
| Optimizer (experimental) | SWAN — tiny smoke → 10M-token pilot → promote only on measured win (loss / stability / throughput / memory), versioned recipe |

---

## 3. Corpus Constitution — 6 audited distributions, 100B training budget

**Core rule: 100B training budget ≠ 100B unique tokens.**
Targets, not mandatory quotas. Unused capacity flows to reasoning/DCLM —
never to questionable sources. Clean > complete.

| Pillar | Budget | Unique pool (est.) | Exposure strategy |
|---|---|---|---|
| 🔬 Science 25% | 25B | ~5–8B + USPTO depth | Light textbook multi-epoch + **Technical-patents sub-branch** (Physics / Chemistry / Biology / Medicine / Space / Engineering / Technical patents). Prevents textbook monoculture. |
| 💻 Code 25% | 25B | Tens of B | Near-single-pass. Stack v2 license-filtered (MIT/Apache/BSD only). C++/C#/graphics-engine weighted. 4-stage filters: heuristics → exact + MinHash dedup → quality → HumanEval/MBPP decontamination. |
| 🧮 Math 15% | 15B | 14B+ | ~Single-pass. OER math, OpenWebMath, FineMath, OMI-2 + OMR (CC-BY), Orca/GSM8K/MetaMath (MIT, capped — synthetic garbage filtered). |
| 📚 English 12.5% | 12.5B | ~7.5B+ | 5% pure (Gutenberg PD, LibreTexts humanities, PD style guides) + **7.5% DCLM-glue** (natural-language binder). Controlled repetition, oversample best. |
| 🌍 GK 10% | 10B | ~0.5–1B | **Multi-representation exposure, never uniform replay**: statement / Q→A / description / relation→sentence / comparison / chronology / geographic relation / MCQ-style / short explanation / embedded-in-passage. Wikidata-verbalized (CC0) + Factbook (PD) + OER history/geo + **🇮🇳 OGD India ~1% of GK** (GODL-verified, per-dataset gate, structured→text). **No Wikipedia** (BY-SA). |
| 🧠 Reasoning 12.5% | 12.5B | ~3–4B+ | Aggressive-gated synthetic expansion (OMI-2, OMR, UltraFeedback, OER proofs). |

---

## 4. License System (Tier B — strict, no blanket claims)

**Never call the corpus "100% clean" from dataset licenses alone.**
Claim: *"6 audited distributions, every shard license-verified."*

### Allowlist
CC0 · CC-BY · ODC-BY · MIT · Apache-2.0 · public-domain · GODL-India.
Anything else fails loudly at the gate cell.

### Explicitly excluded
StarCoderData (OpenRAIL-M) · arXiv NC/SA papers (per-paper filter keeps CC-BY/CC0 only) ·
StackOverflow (BY-SA) · CK-12 (BY-NC) · SciQ-as-training (BY-NC; **eval-only OK**) ·
Python-Edu-unfiltered · unknown-license sets · gated sets.

### Notes
- SmolTalk: check **per-subset** (bundles carry original licenses).
- NC/SA sets remain valid as **eval-only** (never trained on, never redistributed in weights).
- GODL attribution collapses to a single link-page in the release report.

### Per-shard provenance JSON (mandatory for every shard)
```json
{
  "pillar": "science",
  "branch": "physics",
  "source": "OpenStax",
  "license": "CC-BY",
  "license_verified": true,
  "commercial_use": true,
  "redistribution": true,
  "terms_notes": "",
  "documents": 123456,
  "unique_clean_tokens": 0,
  "effective_training_tokens": 0,
  "repeat_factor": 0,
  "synthetic_tokens": 0,
  "deduplication": "...",
  "filter_version": "...",
  "tokenizer_version": "..."
}
```

### Final manifest rule
Release states honestly: **X B unique licensed tokens · Y B synthetic/augmented ·
Z B repeated/resampled = 100B training budget.**

---

## 5. Tokenizer Design

- Train SPM-48K on EN/code/math sample **before** bulk processing.
- **Fertility hard gate** (tokens/char, tokens/word per language; code ≤1.2×/1.3×
  vs English; checks notation, symbols, whitespace, unk-rate).
- Fail = adjust vocab/mix. Never process billions of tokens blind.
- Tokenizer is the pipeline's **measurement system**.

---

## 6. Data Pipeline Design

- Streaming: source → stream → license verify → filter → dedup → tokenize →
  shard → persistent store → delete raw. **≤5GB raw resident ever.**
- Cross-pillar dedup after balancing. Branch quotas per science split.
- Coverage probes per branch. Contamination gates (HumanEval/MBPP/MATH/GSM8K).

---

## 7. Kaggle Notebook (20 cells — orchestrator, not implementation dump)

```text
01 env / reproducibility (pinned versions, GPU fingerprint, seeds, config hash)
02 tokenizer construction
03 tokenizer smoke tests
04 tiny end-to-end (batch→step→ckpt→kill→resume-bit-identical)
05 license policy + provenance schema
06 source acquisition
07 global filtering / dedup
08 science   09 code   10 math   11 English   12 GK   13 reasoning
14 pillar balancing / quotas
15 cross-pillar deduplication
16 fertility + contamination gates
17 validation + coverage probes
18 final packing (streaming packer)
19 final provenance manifest (AUDIT GATE — no 100B launch until it passes)
20 pretraining (DDP, resume-mandatory)
```

Order: 01–04 first (validate machinery), 05–20 after smoke passes.

---

## 8. Kaggle Execution Workflow

```text
LOCAL (alpha/ repo: notebooks/ src/{model,tokenizer,data,training,eval}
       configs/alpha_450m.json scripts/ manifests/ tests/ README.md)
  → KAGGLE (01–04 validate → 05–17 construct/audit → 18–19 package → 20 train)
  → PERSISTENT ARTIFACTS (shards, checkpoints, tokenizer, manifests, evals)
  → HF RELEASE
```

- Sessions die: attach artifacts → resume exactly (weights + optimizer +
  scheduler + step + tokens_seen + RNG + shard position + git commit).
- Many short sessions = one long run (e.g. 0→1.2B→2.5B→…→100B).
- v1 scars designed out: finals saved by explicit step number; append-only logs;
  gates accept ≥ target−1 with actual-latest dirs.

---

## 9. Training Plan

| Stage | Config |
|---|---|
| Pretrain | 100B budget (~2.7e23 FLOPs ≈ ~3,000 T4-h). Free-tier-first with resume engineering; paid cloud (~$0.5–2K) optional, never a redesign trigger. Hardware-independent recipe. |
| SFT | ~300–500K instructions: **35% custom Alpha packs** (science/math/code/GK/explanations/reasoning) / 25% OMI-2 / 20% SmolTalk / 20% UltraChat-MIT. Generic chit-chat minimized. |
| DPO | UltraFeedback (MIT; v1-proven pattern). |
| Bars | HumanEval >20% · MATH >10% · MMLU >40% · IFEval 35–45 (peer: SmolLM2-360M 41.0). |

---

## 10. Eval Plan (per pillar)

Code: HumanEval, MBPP · Math: MATH, GSM8K · Reasoning: ARC, HellaSwag ·
Knowledge: MMLU, TriviaQA · Instruction: IFEval, MT-Bench · Truthfulness: TruthfulQA.
NC/SA sets eval-only.

---

## 11. Release Plan

Weights (safetensors + GGUF Q8/Q4) + notebook + provenance manifest +
model/data/eval reports on HF. Licenses: MIT code + Apache-2.0 weights
(v1 precedent) + CC BY 4.0 docs.

---

## 12. Decisions Log (frozen)

1. Compute: free-tier-first, cloud optional.
2. SFT: 35/25/20/20 custom/OMI-2/SmolTalk/UltraChat + DPO.
3. Notebook: 01–04 parallel now; 05–20 after smoke.
4. Optimizer: AdamW constitutional; SWAN experimental via pilot.
5. India: 2% of GK budget (~200M), ~1% OGD-sourced.
6. Wikipedia: excluded (option a). DCLM: 7.5% glue inside English.
7. English: Tier B strict, smaller-but-clean. 100B target kept.
8. Config format: JSON (v1 precedent).

## 13. Open / Deferred

- Repo scaffold + cells 01–04 (blocked: exams + plan-mode lift; zero GPU needed locally to write).
- SWAN pilot results (post-smoke).
- Exact per-source sampling multipliers (implementation-versioned, not constitutional).

---

*Constitution frozen 2026-09-14. Good luck with exams — Alpha waits; v1 finishes itself.*
