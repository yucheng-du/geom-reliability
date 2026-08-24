"""
run_selfconsistency.py — Generate N samples per prompt for self-consistency baseline.

Key design choices:
  - Uses num_return_sequences=N to generate all samples in a single forward pass
    (3-4x faster than looping N times per prompt).
  - Default max_new_tokens=50 (sufficient to capture the answer for short
    math/fact/code questions; saves ~50% decoding time vs. 100 tokens).
  - Tokenises each prompt once and reuses the tensor.
  - Supports resume: appends to existing output file, skipping already-done ids.

Usage:
  python run_selfconsistency.py --model llama --n-samples 5 \\
      --run-dirs experiments/runs/run_main_llama \\
      --out-file experiments/selfconsistency/llama_math50_fact10.jsonl

  # Replace run_main_llama with run_code_llama for the CODE prompts.

Output JSONL (one line per sample):
  {"id": "m01a", "form": "MATH", "answerable": "A",
   "sample_idx": 0, "response": "...", "model": "Llama-3.1-8B-Instruct"}
"""

import argparse
import json
import os

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


def load_prompts_from_dirs(run_dirs):
    """Load and deduplicate prompts from multiple run directories."""
    seen_ids = set()
    prompts = []
    for d in run_dirs:
        path = os.path.join(d, "prompts.jsonl")
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    prompts.append(row)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama", "qwen", "mistral"])
    parser.add_argument(
        "--model-id",
        help="Override the default Hugging Face model id or use a local path."
    )
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Samples per prompt (default 5). All generated in one pass.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Max tokens per sample (default 50; sufficient for "
                             "short math/fact/code answers and ~2x faster than 100).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)

    # Resume support: track which prompt ids are already fully done
    id_counts: dict[str, int] = {}
    if os.path.isfile(args.out_file):
        with open(args.out_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                id_counts[r["id"]] = id_counts.get(r["id"], 0) + 1
        overfull = {pid: cnt for pid, cnt in id_counts.items() if cnt > args.n_samples}
        if overfull:
            raise ValueError(
                "Output contains more than --n-samples rows for some prompts; "
                f"clean the file before resuming: {overfull}"
            )
        done_ids = {pid for pid, cnt in id_counts.items() if cnt == args.n_samples}
        print(f"Resume: {len(done_ids)} prompts already fully done.")
    else:
        done_ids = set()

    device     = get_device()
    model_id   = args.model_id or MODEL_IDS[args.model]
    model_name = MODEL_LABELS[args.model]
    dtype      = torch.float32 if device.type == "cpu" else torch.float16

    print(f"Model        : {model_name}")
    print(f"Path         : {model_id}")
    print(f"Device       : {device}")
    print(f"Dtype        : {dtype}")
    print(f"Samples/prompt: {args.n_samples}  (generated in one pass per prompt)")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"temperature  : {args.temperature}  top_p: {args.top_p}")
    print(f"Output       : {args.out_file}\n")

    prompts = load_prompts_from_dirs(args.run_dirs)
    todo = [p for p in prompts if p["id"] not in done_ids]
    print(f"Prompts total: {len(prompts)}  |  Remaining: {len(todo)}\n")

    if not todo:
        print("Nothing to do — all prompts already complete.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    model.eval()

    with open(args.out_file, "a", encoding="utf-8") as out_f:
        for idx, row in enumerate(todo):
            pid, form, answerable, prompt_text = (
                row["id"], row["form"], row["answerable"], row["prompt"]
            )

            # Tokenise once, reuse for all samples
            inputs = tokenizer(
                prompt_text, return_tensors="pt", padding=False
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            existing_count = id_counts.get(pid, 0)
            samples_needed = args.n_samples - existing_count

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=samples_needed,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # gen_ids: [samples_needed, seq_len]

            for s_idx, gen in enumerate(gen_ids, start=existing_count):
                response = tokenizer.decode(
                    gen[prompt_len:], skip_special_tokens=True
                ).strip()
                out_f.write(json.dumps({
                    "id":         pid,
                    "form":       form,
                    "answerable": answerable,
                    "sample_idx": s_idx,
                    "response":   response,
                    "model":      model_name,
                }, ensure_ascii=False) + "\n")
            out_f.flush()

            print(f"  [{idx+1:>3}/{len(todo)}] {pid} ({form}/{answerable})")

    print(f"\nDone. Output: {args.out_file}")


if __name__ == "__main__":
    main()
