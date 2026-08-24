"""
run_layerwise.py — Extract all-layer hidden states for MATH matched pairs.

For each prompt, saves a matrix of shape (n_layers, hidden_dim) where each
row is the mean-pooled hidden state at that layer (including embedding layer 0).

Uses the first 20 MATH matched pairs from the released prompt file, matching the
paper's layer-wise experiment. FACT prompts are excluded.

Usage:
    python run_layerwise.py --model llama \
        --prompts data/math50_fact10.jsonl
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

MODEL_IDS = {
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B-Instruct",
    "qwen":    "Qwen2.5-7B-Instruct",
    "mistral": "Mistral-7B-Instruct-v0.3",
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
    parser.add_argument(
        "--prompts", default="data/math50_fact10.jsonl",
        help="Prompt JSONL (default: data/math50_fact10.jsonl)."
    )
    parser.add_argument(
        "--n-pairs", type=int, default=20,
        help="Number of MATH pairs to extract (default: 20, as in the paper)."
    )
    parser.add_argument(
        "--out-dir",
        help="Output directory (default: experiments/runs/run_layerwise_<model>)."
    )
    parser.add_argument(
        "--model-id",
        help="Override the default Hugging Face model id or use a local path."
    )
    args = parser.parse_args()

    if args.n_pairs < 1:
        parser.error("--n-pairs must be at least 1")

    out_dir = args.out_dir or f"experiments/runs/run_layerwise_{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    device     = get_device()
    model_id   = args.model_id or MODEL_IDS[args.model]
    model_name = MODEL_LABELS[args.model]
    dtype      = torch.float32 if device.type == "cpu" else torch.float16

    print(f"Model   : {model_name}")
    print(f"Path    : {model_id}")
    print(f"Device  : {device}")
    print(f"Dtype   : {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()

    rows = load_prompts(args.prompts)
    math_prompts = [r for r in rows if r["form"] == "MATH"]
    expected = 2 * args.n_pairs
    if len(math_prompts) < expected:
        raise ValueError(
            f"Requested {args.n_pairs} MATH pairs, but {args.prompts} contains "
            f"only {len(math_prompts)} MATH prompts."
        )
    all_prompts = math_prompts[:expected]

    print(f"MATH prompts: {len(all_prompts)} ({args.n_pairs} pairs)")

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
