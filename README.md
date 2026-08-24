# Geometric Deviation as an Unsupervised Pre-Generation Reliability Signal

Code and researcher-constructed matched-pair prompts for:

> Yucheng Du. **Geometric Deviation as an Unsupervised Pre-Generation
> Reliability Signal: Probing LLM Representations for Answerability.**
> Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP 2026),
> pages 353–363.

[ACL Anthology](https://aclanthology.org/2026.trustnlp-main.22/) ·
[DOI](https://doi.org/10.18653/v1/2026.trustnlp-main.22) ·
[arXiv](https://arxiv.org/abs/2605.03196)

## Repository scope

The repository contains the prompt data and scripts for representation
extraction, the controlled geometry analysis, layer-wise analysis, reliability
prediction, and the self-consistency baseline. Model weights, generated model
outputs, and hidden-state arrays are not redistributed; the scripts create
those artifacts under `experiments/`, which is ignored by Git.

## Setup

Python 3.9 or newer is required. Create an isolated environment and install the
dependencies:

```bash
git clone https://github.com/yucheng-du/geom-reliability.git
cd geom-reliability
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts use these Hugging Face model identifiers by default:

| CLI key | Model identifier |
|---|---|
| `llama` | [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| `qwen` | [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `mistral` | [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |

Llama access requires accepting Meta's model terms and authenticating with
Hugging Face. All extraction commands accept `--model-id` to use a local model
directory or a different compatible identifier. A GPU or Apple-silicon MPS
device is strongly recommended for the 7–8B models.

## Quick start

This example runs the released MATH/FACT prompts with Qwen and produces the
controlled-analysis report and figures:

```bash
python run_extract.py \
  --model qwen \
  --prompts data/math50_fact10.jsonl \
  --run-dir experiments/runs/run_main_qwen

python analyze_controlled.py \
  --run-dirs experiments/runs/run_main_qwen \
  --label qwen_math50_fact10

python compute_auc_v2.py --model qwen --latex
```

`run_extract.py` copies the input prompt file into the run directory so that
the downstream generation baselines can consume the same frozen prompts.

## Reproducing the analyses

### 1. Extract last-layer representations

Run MATH + FACT and CODE for each model:

```bash
for model in llama qwen mistral; do
  python run_extract.py \
    --model "$model" \
    --prompts data/math50_fact10.jsonl \
    --run-dir "experiments/runs/run_main_${model}"

  python run_extract.py \
    --model "$model" \
    --prompts data/code30.jsonl \
    --run-dir "experiments/runs/run_code_${model}"
done
```

Each run writes:

- `prompts.jsonl`: the frozen input prompts;
- `reps/reps_last_raw.npy`: raw last-layer, mean-pooled representations;
- `reps/meta.jsonl`: row-aligned metadata;
- `generations.jsonl`: generated continuations used by the refusal baseline.

### 2. Controlled answerability analysis

```bash
for model in llama qwen mistral; do
  python analyze_controlled.py \
    --run-dirs "experiments/runs/run_main_${model}" \
    --label "${model}_math50_fact10"

  python analyze_controlled.py \
    --run-dirs "experiments/runs/run_code_${model}" \
    --label "${model}_code30"
done
```

Reports and diagnostic figures are written to
`experiments/runs/results_<label>/`.

### 3. Layer-wise signal profile

The paper's layer-wise experiment uses the first 20 MATH matched pairs:

```bash
for model in llama qwen mistral; do
  python run_layerwise.py \
    --model "$model" \
    --prompts data/math50_fact10.jsonl \
    --n-pairs 20
done

python analyze_layerwise.py --all-models
```

### 4. Geometry and refusal-baseline table

After completing Step 1 for all models:

```bash
python compute_auc_v2.py --latex
```

The default directory map in `compute_auc_v2.py` matches the commands above.

### 5. Self-consistency baseline

The following runs and evaluates five-sample self-consistency for the
MATH/FACT prompts. Repeat with `run_code_<model>` and an appropriate output
filename for CODE.

```bash
for model in llama qwen mistral; do
  python run_selfconsistency.py \
    --model "$model" \
    --n-samples 5 \
    --run-dirs "experiments/runs/run_main_${model}" \
    --out-file "experiments/selfconsistency/${model}_math50_fact10.jsonl"

  python compute_selfcons_auc.py \
    --sc-file "experiments/selfconsistency/${model}_math50_fact10.jsonl" \
    --run-dirs "experiments/runs/run_main_${model}" \
    --model "$model" \
    --label "${model}_math50_fact10" \
    --latex
done
```

Generation is stochastic, so refusal and self-consistency values can vary
across runs and software/hardware backends. The repository does not claim
bit-for-bit reproduction of the unpublished generated outputs.

## Data

`data/` contains the released matched-pair prompt sets:

- `data/math50_fact10.jsonl`: 50 MATH pairs and 10 FACT pairs (120 prompts);
- `data/code30.jsonl`: 30 CODE pairs (60 prompts).

Each JSONL row contains `id`, `form`, `answerable` (`A` or `U`), and `prompt`.
The prompts were constructed for this study; no personal data is included.

## File overview

| File | Purpose |
|---|---|
| `run_extract.py` | Extract last-layer representations and model continuations |
| `analyze_controlled.py` | Permutation tests, own-distance analysis, and diagnostic figures |
| `run_layerwise.py` | Extract hidden states at every layer for MATH prompts |
| `analyze_layerwise.py` | Compute and plot the layer-wise answerability gap |
| `compute_auc_v2.py` | Compare geometry with the refusal-keyword baseline |
| `run_selfconsistency.py` | Generate repeated samples for self-consistency |
| `compute_selfcons_auc.py` | Compare geometry, self-consistency, and refusal baselines |
| `gen_paper_figs.py` | Legacy combined-figure helper for the archived run layout |

## Citation

```bibtex
@inproceedings{du-2026-geometric,
  title     = {Geometric Deviation as an Unsupervised Pre-Generation
               Reliability Signal: Probing {LLM} Representations for Answerability},
  author    = {Du, Yucheng},
  booktitle = {Proceedings of the 6th Workshop on Trustworthy {NLP}
               ({T}rust{NLP} 2026)},
  month     = jul,
  year      = {2026},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.trustnlp-main.22/},
  doi       = {10.18653/v1/2026.trustnlp-main.22},
  pages     = {353--363}
}
```

## License

Unless otherwise noted, the original code and included prompt data are
released under the [MIT License](LICENSE). Model weights are not included and
remain subject to their respective model licenses and terms of use.
