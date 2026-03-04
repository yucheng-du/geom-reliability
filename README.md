# Geometric Deviation as an Unsupervised Pre-Generation Reliability Signal

Code and data for the paper:

> **Geometric Deviation as an Unsupervised Pre-Generation Reliability Signal: Probing LLM Representations for Answerability**  
> TrustNLP Workshop @ ACL 2026

## Setup

```bash
pip install -r requirements.txt
```

Models are loaded via HuggingFace Transformers. Set `--model` to `llama`, `qwen`, or `mistral`; the scripts expect the following model paths (edit `run_extract.py` to adjust):

| Model | Expected path |
|---|---|
| Llama 3.1-8B-Instruct | `~/.llama/checkpoints/Llama3.1-8B-Instruct-HF` |
| Qwen 2.5-7B-Instruct | HuggingFace hub: `Qwen/Qwen2.5-7B-Instruct` |
| Mistral-7B-Instruct-v0.3 | HuggingFace hub: `mistralai/Mistral-7B-Instruct-v0.3` |

## Data

`data/` contains the matched-pair prompt datasets:

- `data/math50_fact10.jsonl` — 50 MATH pairs + 10 FACT pairs (120 prompts A+U)
- `data/code30.jsonl` — 30 CODE pairs (60 prompts A+U)

Each line is a JSON object with fields: `id`, `form`, `answerable` (`A`/`U`), `prompt`.

## Reproducing the Paper

### Step 1: Extract representations

```bash
# MATH + FACT (Llama)
python run_extract.py --model llama --run-dir experiments/runs/run_main_llama

# MATH + FACT (Qwen / Mistral)
python run_extract.py --model qwen    --run-dir experiments/runs/run_main_qwen
python run_extract.py --model mistral --run-dir experiments/runs/run_main_mistral

# CODE (all three models)
python run_extract.py --model llama   --run-dir experiments/runs/run_code_llama
python run_extract.py --model qwen    --run-dir experiments/runs/run_code_qwen
python run_extract.py --model mistral --run-dir experiments/runs/run_code_mistral
```

### Step 2: Run controlled answerability analysis

```bash
python analyze_controlled.py \
  --run-dirs experiments/runs/run_main_llama \
  --label llama_math50

python analyze_controlled.py \
  --run-dirs experiments/runs/run_code_llama \
  --label llama_code30
```

### Step 3: Layer-wise signal profile

```bash
python run_layerwise.py --model llama
python run_layerwise.py --model qwen
python run_layerwise.py --model mistral
python analyze_layerwise.py --both
```

### Step 4: Reliability prediction (AUC table)

```bash
python compute_auc_v2.py --latex
```

### Step 5: Self-consistency baseline

```bash
python run_selfconsistency.py --model llama --n-samples 5 \
  --run-dirs experiments/runs/run_main_llama \
  --out-file experiments/selfconsistency/llama_main.jsonl

python compute_selfcons_auc.py \
  --sc-file experiments/selfconsistency/llama_main.jsonl \
  --geo-run-dirs experiments/runs/run_main_llama \
  --model llama --label llama_main
```

## File Overview

| Script | Purpose |
|---|---|
| `run_extract.py` | Extract last-layer hidden states, save `reps_last.npy` |
| `run_layerwise.py` | Extract hidden states at every layer |
| `run_selfconsistency.py` | Generate 5 samples per prompt for SC baseline |
| `analyze_controlled.py` | Permutation test + own_dist analysis |
| `analyze_layerwise.py` | Layer-wise gap figure |
| `compute_auc_v2.py` | ROC-AUC table (Geometry vs Refusal) |
| `compute_selfcons_auc.py` | ROC-AUC table including SC baseline |
| `gen_paper_figs.py` | PCA and boxplot figures |
