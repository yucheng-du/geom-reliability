"""
analyze_layerwise.py — Layer-wise answerability signal profile for MATH form.

For each layer l, computes:
  gap(l) = mean cosine_dist(MATH-U, centroid_A(l)) - mean cosine_dist(MATH-A, centroid_A(l))

where centroid_A(l) is the mean-centered mean of MATH-A vectors at layer l.

Produces:
  - Console table: per-layer dist_A, dist_U, gap
  - Figure: gap vs. layer index for both models (side-by-side)

Usage:
    python analyze_layerwise.py                    # Llama only
    python analyze_layerwise.py --both             # Llama + Qwen
    python analyze_layerwise.py --model mistral    # Mistral only

Requires: run_layerwise.py to have been run first for each model.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LAYERWISE_DIRS = {
    "llama":   "experiments/runs/run_layerwise_llama",
    "qwen":    "experiments/runs/run_layerwise_qwen",
    "mistral": "experiments/runs/run_layerwise_mistral",
}

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B",
    "qwen":    "Qwen2.5-7B",
    "mistral": "Mistral-7B",
}

MODEL_COLORS = {
    "llama":   "#4C72B0",
    "qwen":    "#C44E52",
    "mistral": "#8172B2",
}


def load_layerwise(run_dir):
    reps = np.load(
        os.path.join(run_dir, "reps_layerwise.npy"), allow_pickle=False
    ).astype(np.float32)
    meta = []
    with open(os.path.join(run_dir, "meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    assert reps.shape[0] == len(meta), "reps/meta length mismatch"
    return reps, meta  # reps: (n_prompts, n_layers, hidden_dim)


def normalise(X):
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / (norms + 1e-12)


def cosine_dist_to_vec(X, vec):
    """X: (n, d), vec: (d,)  →  (n,) cosine distances."""
    Xn = normalise(X)
    vn = vec / (np.linalg.norm(vec) + 1e-12)
    return 1.0 - (Xn @ vn)


def compute_layer_gaps(reps, meta):
    """
    Returns dist_A, dist_U, gap arrays of shape (n_layers,).
    reps: (n_prompts, n_layers, hidden_dim)
    """
    ans_arr = np.array([r["answerable"] for r in meta])
    mask_A  = ans_arr == "A"
    mask_U  = ans_arr == "U"

    n_layers = reps.shape[1]
    dist_A_arr = np.zeros(n_layers)
    dist_U_arr = np.zeros(n_layers)

    for l in range(n_layers):
        layer_vecs = reps[:, l, :]           # (n_prompts, hidden_dim)

        # mean-center at this layer
        layer_vecs_c = layer_vecs - layer_vecs.mean(axis=0)

        centroid_A = layer_vecs_c[mask_A].mean(axis=0)
        d_A = cosine_dist_to_vec(layer_vecs_c[mask_A], centroid_A).mean()
        d_U = cosine_dist_to_vec(layer_vecs_c[mask_U], centroid_A).mean()

        dist_A_arr[l] = d_A
        dist_U_arr[l] = d_U

    return dist_A_arr, dist_U_arr, dist_U_arr - dist_A_arr


def print_layer_table(model_key, dist_A, dist_U, gap):
    label = MODEL_LABELS[model_key]
    print(f"\n{'='*62}")
    print(f"  {label}  —  Layer-wise MATH answerability signal")
    print(f"{'='*62}")
    print(f"  {'layer':>6}  {'dist_A':>8}  {'dist_U':>8}  {'gap(U-A)':>10}")
    print("  " + "-" * 38)
    for l, (da, du, g) in enumerate(zip(dist_A, dist_U, gap)):
        marker = " ←" if l == len(dist_A) - 1 else ""
        print(f"  {l:>6}  {da:>8.4f}  {du:>8.4f}  {g:>+10.4f}{marker}")
    print(f"\n  Peak gap: layer {int(np.argmax(gap))}  gap={gap.max():.4f}")
    print(f"  Final gap (last layer): {gap[-1]:+.4f}")


def plot_layerwise(results, out_path):
    """
    results: dict of model_key → (dist_A, dist_U, gap)
    Single figure, one panel per model.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_key, (dist_A, dist_U, gap)) in zip(axes, results.items()):
        n_layers = len(gap)
        layers   = np.arange(n_layers)
        color    = MODEL_COLORS[model_key]
        label    = MODEL_LABELS[model_key]

        ax.plot(layers, dist_U, color=color, linewidth=2,
                label="MATH-U", linestyle="-")
        ax.plot(layers, dist_A, color=color, linewidth=2,
                label="MATH-A", linestyle="--", alpha=0.65)
        ax.fill_between(layers, dist_A, dist_U, alpha=0.15, color=color)

        # Mark last layer
        ax.axvline(n_layers - 1, color="grey", linewidth=0.8, linestyle=":")

        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Mean cosine dist to MATH-A centroid", fontsize=9)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

    fig.suptitle(
        "Layer-wise answerability signal: MATH-U vs MATH-A deviation\n"
        "(gap = distance of U − distance of A from answerable centroid)",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


def plot_gap_only(results, out_path):
    """
    Plot gap (U-A) vs. layer for all models on one panel — cleaner for the paper.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for model_key, (dist_A, dist_U, gap) in results.items():
        n_layers = len(gap)
        layers   = np.arange(n_layers)
        color    = MODEL_COLORS[model_key]
        label    = MODEL_LABELS[model_key]
        ax.plot(layers, gap, color=color, linewidth=2.2, label=label)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("Gap: mean dist(U) − mean dist(A)", fontsize=10)
    ax.set_title("Layer-wise answerability signal (MATH form)", fontsize=11)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Gap-only figure: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama", "qwen", "mistral"], default="llama",
                        help="Single model to analyse (default: llama).")
    parser.add_argument("--both", action="store_true",
                        help="Analyse Llama + Qwen together.")
    parser.add_argument("--all-models", action="store_true",
                        help="Analyse all three models.")
    args = parser.parse_args()

    if args.all_models:
        model_keys = ["llama", "qwen", "mistral"]
    elif args.both:
        model_keys = ["llama", "qwen"]
    else:
        model_keys = [args.model]

    results = {}
    for model_key in model_keys:
        run_dir = LAYERWISE_DIRS[model_key]
        if not os.path.isfile(os.path.join(run_dir, "reps_layerwise.npy")):
            print(f"[skip] {model_key}: reps_layerwise.npy not found in {run_dir}")
            continue

        reps, meta = load_layerwise(run_dir)
        dist_A, dist_U, gap = compute_layer_gaps(reps, meta)
        results[model_key] = (dist_A, dist_U, gap)
        print_layer_table(model_key, dist_A, dist_U, gap)

    if not results:
        print("No layer-wise data found. Run run_layerwise.py first.")
        return

    # Figures
    fig_dir = "experiments/runs"
    tag = "_".join(results.keys())

    plot_layerwise(results, os.path.join(fig_dir, f"layerwise_traces_{tag}.png"))
    plot_gap_only(results,  os.path.join(fig_dir, f"layerwise_gap_{tag}.png"))


if __name__ == "__main__":
    main()
