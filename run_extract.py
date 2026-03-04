"""
run_extract.py — Unified representation extraction for any model and prompt set.

Loads prompts.jsonl from a run directory, extracts last-layer mean-pooled
representations, and saves generations. Supports Llama, Qwen, and Mistral.

Usage:
    python run_extract.py --model llama   --run-dir experiments/runs/run_003c
    python run_extract.py --model qwen    --run-dir experiments/runs/run_003c_qwen
    python run_extract.py --model mistral --run-dir experiments/runs/run_003_mistral
    python run_extract.py --model mistral --run-dir experiments/runs/run_004_code_mistral

Output (within run-dir):
    reps/reps_last_raw.npy    raw last-layer mean-pooled vectors
    reps/meta.jsonl           prompt metadata (id, form, answerable, prompt, row_idx)
    generations.jsonl         prompt + generated text
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── model paths ─────────────────────────────────────────────────────────────────

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


# ── helpers ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean_pool_last_layer(model, inputs):
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu().numpy()


# ── main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract last-layer representations for a prompt set."
    )
    parser.add_argument(
        "--model", required=True, choices=["llama", "qwen", "mistral"],
        help="Model to use."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Run directory containing prompts.jsonl. Outputs written here."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Max tokens for generation (default: 80)."
    )
    args = parser.parse_args()

    prompts_path = os.path.join(args.run_dir, "prompts.jsonl")
    reps_dir     = os.path.join(args.run_dir, "reps")
    gen_path     = os.path.join(args.run_dir, "generations.jsonl")

    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"prompts.jsonl not found in {args.run_dir}")

    os.makedirs(reps_dir, exist_ok=True)

    device     = get_device()
    model_id   = MODEL_PATHS[args.model]
    model_name = MODEL_LABELS[args.model]

    print(f"Model   : {model_name}")
    print(f"Path    : {model_id}")
    print(f"Run dir : {args.run_dir}")
    print(f"Device  : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    prompts   = load_prompts(prompts_path)
    all_vecs  = []
    meta_rows = []

    print(f"Loaded  : {len(prompts)} prompts\n")

    with open(gen_path, "w", encoding="utf-8") as gen_out:
        for idx, row in enumerate(prompts):
            pid, form, answerable, prompt = (
                row["id"], row["form"], row["answerable"], row["prompt"]
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)

            vec  = mean_pool_last_layer(model, inputs)
            norm = float(np.linalg.norm(vec))

            all_vecs.append(vec)
            meta_rows.append({
                "row_idx":    idx,
                "id":         pid,
                "form":       form,
                "answerable": answerable,
                "prompt":     prompt,
            })

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            gen_out.write(json.dumps({
                "id": pid, "form": form, "answerable": answerable,
                "prompt": prompt, "text": text, "model": model_name,
            }, ensure_ascii=False) + "\n")

            print(f"  [{idx+1:>3}/{len(prompts)}] {pid} ({form}/{answerable}) "
                  f"norm={norm:.3f}")

    reps_raw = np.stack(all_vecs, axis=0).astype(np.float32)
    np.save(os.path.join(reps_dir, "reps_last_raw.npy"), reps_raw)

    with open(os.path.join(reps_dir, "meta.jsonl"), "w", encoding="utf-8") as mf:
        for r in meta_rows:
            mf.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  reps_last_raw : {reps_raw.shape}")
    print(f"  generations   : {gen_path}")
    print(f"\nNext: run analyze_controlled.py with appropriate --run-dirs flags.")


if __name__ == "__main__":
    main()
