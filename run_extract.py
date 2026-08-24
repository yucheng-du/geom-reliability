"""
run_extract.py — Unified representation extraction for any model and prompt set.

Loads a prompt JSONL file, extracts last-layer mean-pooled representations,
and saves generations. Supports Llama, Qwen, and Mistral.

Usage:
    python run_extract.py --model llama \
        --prompts data/math50_fact10.jsonl \
        --run-dir experiments/runs/run_main_llama

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

# ── model identifiers ───────────────────────────────────────────────────────────

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
        help="Run directory for outputs."
    )
    parser.add_argument(
        "--prompts",
        help="Input prompt JSONL. Defaults to <run-dir>/prompts.jsonl."
    )
    parser.add_argument(
        "--model-id",
        help="Override the default Hugging Face model id or use a local path."
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80,
        help="Max tokens for generation (default: 80)."
    )
    args = parser.parse_args()

    prompts_path = args.prompts or os.path.join(args.run_dir, "prompts.jsonl")
    canonical_prompts_path = os.path.join(args.run_dir, "prompts.jsonl")
    reps_dir     = os.path.join(args.run_dir, "reps")
    gen_path     = os.path.join(args.run_dir, "generations.jsonl")

    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"Prompt file not found: {prompts_path}")

    os.makedirs(reps_dir, exist_ok=True)

    # Keep a copy beside the outputs so downstream scripts can consume the run
    # directory without relying on the caller's original data path.
    prompts = load_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"Prompt file is empty: {prompts_path}")
    if os.path.abspath(prompts_path) != os.path.abspath(canonical_prompts_path):
        with open(canonical_prompts_path, "w", encoding="utf-8") as pf:
            for row in prompts:
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")

    device     = get_device()
    model_id   = args.model_id or MODEL_IDS[args.model]
    model_name = MODEL_LABELS[args.model]
    dtype      = torch.float32 if device.type == "cpu" else torch.float16

    print(f"Model   : {model_name}")
    print(f"Path    : {model_id}")
    print(f"Run dir : {args.run_dir}")
    print(f"Device  : {device}")
    print(f"Dtype   : {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()

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
            prompt_len = inputs["input_ids"].shape[1]
            text = tokenizer.decode(
                gen_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()

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
