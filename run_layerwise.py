"""
run_layerwise.py — Extract all-layer hidden states for MATH matched pairs.

For each prompt, saves a matrix of shape (n_layers, hidden_dim) where each
row is the mean-pooled hidden state at that layer (including embedding layer 0).

Uses the merged MATH n=20 paired prompts (same as analyze_003_full.py Llama mode,
or run_003_qwen for Qwen). Only MATH prompts are extracted (FACT excluded to keep
output manageable and focus the layer-wise analysis on the significant form).

Usage:
    python run_layerwise.py --model llama
    python run_layerwise.py --model qwen
    python run_layerwise.py --model mistral

Output (in experiments/runs/run_layerwise_{model}/):
    reps_layerwise.npy    shape (n_prompts, n_layers, hidden_dim), float32
    meta.jsonl            prompt metadata
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATHS = {
    "llama": os.path.expanduser(
        "~/.llama/checkpoints/Llama3.1-8B-Instruct-HF"
    ),
    "qwen": os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
        "/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ),
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B-Instruct",
    "qwen":    "Qwen2.5-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
}

# Source dirs for MATH n=20 prompts per model
MATH_PROMPT_DIRS = {
    "llama":   ["experiments/runs/run_003", "experiments/runs/run_003b"],
    "qwen":    ["experiments/runs/run_003_qwen"],
    "mistral": ["experiments/runs/run_003_mistral"],
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_all_layers(model, inputs):
    """Return mean-pooled hidden state at every layer, shape (n_layers, hidden_dim)."""
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    # out.hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    # index 0 = embedding layer, 1..N = transformer layers
    vecs = []
    for hs in out.hidden_states:
        vec = hs.mean(dim=1).squeeze(0).float().cpu().numpy()
        vecs.append(vec)
    return np.stack(vecs, axis=0)  # (n_layers+1, hidden_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "mistral"])
    args = parser.parse_args()

    out_dir = f"experiments/runs/run_layerwise_{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    device     = get_device()
    model_id   = MODEL_PATHS[args.model]
    model_name = MODEL_LABELS[args.model]

    print(f"Model   : {model_name}")
    print(f"Device  : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    # Load MATH prompts only (filter out FACT)
    all_prompts = []
    for src_dir in MATH_PROMPT_DIRS[args.model]:
        rows = load_prompts(os.path.join(src_dir, "prompts.jsonl"))
        all_prompts.extend([r for r in rows if r["form"] == "MATH"])

    print(f"MATH prompts: {len(all_prompts)}")

    all_layer_vecs = []
    meta_rows      = []

    for idx, row in enumerate(all_prompts):
        pid, form, answerable, prompt = (
            row["id"], row["form"], row["answerable"], row["prompt"]
        )
        inputs     = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        layer_vecs = extract_all_layers(model, inputs)  # (n_layers, hidden_dim)

        all_layer_vecs.append(layer_vecs)
        meta_rows.append({
            "row_idx": idx, "id": pid,
            "form": form, "answerable": answerable, "prompt": prompt,
        })

        print(f"  [{idx+1:>2}/{len(all_prompts)}] {pid} ({answerable})  "
              f"layers={layer_vecs.shape[0]}  dim={layer_vecs.shape[1]}")

    reps = np.stack(all_layer_vecs, axis=0).astype(np.float32)
    # reps shape: (n_prompts, n_layers, hidden_dim)
    np.save(os.path.join(out_dir, "reps_layerwise.npy"), reps)

    with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as mf:
        for r in meta_rows:
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. reps_layerwise: {reps.shape}")
    print(f"  (n_prompts={reps.shape[0]}, n_layers={reps.shape[1]}, "
          f"hidden_dim={reps.shape[2]})")
    print(f"Saved to: {out_dir}")
    print("Next: run analyze_layerwise.py")


if __name__ == "__main__":
    main()
