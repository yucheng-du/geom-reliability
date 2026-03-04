"""
gen_paper_figs.py — Regenerate clean paper figures (no internal run labels).

Outputs (written to figures/):
  pca_llama_clean.png        — Llama PCA, clean title
  pca_qwen_clean.png         — Qwen PCA, clean title
  boxplot_paper.png          — 2-panel boxplot (Llama | Qwen), n=50 MATH + n=10 FACT
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.decomposition import PCA

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

FORM_COLORS = {
    "FACT":    "#4C72B0",
    "MATH":    "#55A868",
    "UNKNOWN": "#C44E52",
    "CODE":    "#DD8452",
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_centred(run_dir):
    """Load mean-centred reps (reps_last.npy) — used by run_002 / run_002_qwen."""
    reps = np.load(
        os.path.join(run_dir, "reps", "reps_last.npy"), allow_pickle=False
    ).astype(np.float32)
    meta = []
    with open(os.path.join(run_dir, "reps", "meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    return reps, meta


def load_raw_multi(run_dirs):
    """Load raw reps from multiple dirs, merge, then jointly mean-centre."""
    all_reps, all_meta = [], []
    offset = 0
    for d in run_dirs:
        reps = np.load(
            os.path.join(d, "reps", "reps_last_raw.npy"), allow_pickle=False
        ).astype(np.float32)
        with open(os.path.join(d, "reps", "meta.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["row_idx"] = r.get("row_idx", 0) + offset
                    all_meta.append(r)
        all_reps.append(reps)
        offset += len(reps)
    raw = np.vstack(all_reps)
    centred = raw - raw.mean(axis=0)
    return centred, all_meta


# ── geometry helpers ──────────────────────────────────────────────────────────

def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_to_vec(X, vec):
    Xn = normalise(X)
    vn = vec / (np.linalg.norm(vec) + 1e-12)
    return 1.0 - (Xn @ vn)


# ── Figure 1: PCA (single panel, clean title) ─────────────────────────────────

def plot_pca_clean(reps, meta, save_path, model_label):
    # run_002 uses "task" field; run_003 uses "form"
    key = "task" if "task" in meta[0] else "form"
    forms = sorted(set(r[key] for r in meta))
    centroids = {}
    for form in forms:
        mask = np.array([r[key] == form for r in meta])
        centroids[form] = reps[mask].mean(axis=0)

    all_vecs = np.vstack([reps] + [centroids[f].reshape(1, -1) for f in forms])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_vecs)
    var = pca.explained_variance_ratio_

    pt_coords   = coords[:len(reps)]
    cent_coords = coords[len(reps):]

    fig, ax = plt.subplots(figsize=(6, 5))
    for form in forms:
        mask = np.array([r[key] == form for r in meta])
        ax.scatter(pt_coords[mask, 0], pt_coords[mask, 1],
                   c=FORM_COLORS.get(form, "#888888"), s=55, alpha=0.72,
                   label=form.capitalize())
    for i, form in enumerate(forms):
        ax.scatter(cent_coords[i, 0], cent_coords[i, 1],
                   c=FORM_COLORS.get(form, "#888888"), marker="*", s=280,
                   edgecolors="black", linewidths=0.7, zorder=5)
        ax.annotate(f"{form}\ncentroid",
                    (cent_coords[i, 0], cent_coords[i, 1]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=10)
    ax.set_title(f"PCA of mean-centred representations — {model_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 2: Boxplot 2-panel (Llama | Qwen) ─────────────────────────────────

def compute_own_dists(reps, meta, forms=("FACT", "MATH")):
    label_arr = np.array([r["form"]       for r in meta])
    ans_arr   = np.array([r["answerable"] for r in meta])
    dist_by_group = {}
    for form in forms:
        mask_A = (label_arr == form) & (ans_arr == "A")
        if mask_A.sum() == 0:
            continue
        centroid = reps[mask_A].mean(axis=0)
        for ans in ["A", "U"]:
            mask = (label_arr == form) & (ans_arr == ans)
            if mask.sum() > 0:
                dist_by_group[(form, ans)] = cosine_dist_to_vec(reps[mask], centroid)
    return dist_by_group


def plot_boxplot_panel(ax, dist_by_group, title):
    keys   = [("FACT","A"), ("FACT","U"), ("MATH","A"), ("MATH","U")]
    labels = ["Fact-A", "Fact-U", "Math-A", "Math-U"]
    colors = [FORM_COLORS["FACT"]] * 2 + [FORM_COLORS["MATH"]] * 2
    data   = [dist_by_group.get(k, np.array([])) for k in keys]
    pos    = [1, 2, 4, 5]

    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="", linestyle="none"))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(0)
    for p, d, col in zip(pos, data, colors):
        if len(d) > 0:
            ax.scatter(np.full(len(d), p) + rng.uniform(-0.13, 0.13, len(d)),
                       d, color=col, alpha=0.55, s=22, zorder=3)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("cosine distance to\nanswerable centroid", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axvline(3, color="grey", linestyle="--", linewidth=0.8)


def plot_boxplot_combined(dist_llama, dist_qwen, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    plot_boxplot_panel(axes[0], dist_llama, "Llama 3.1-8B")
    plot_boxplot_panel(axes[1], dist_qwen,  "Qwen 2.5-7B")
    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures...")

    # ── PCA figures (from run_002 / run_002_qwen — mean-centred already) ──────
    llama_pca_dir = "experiments/runs/run_002"
    qwen_pca_dir  = "experiments/runs/run_002_qwen"

    for run_dir, label, out_name in [
        (llama_pca_dir, "Llama 3.1-8B", "pca_llama_clean.png"),
        (qwen_pca_dir,  "Qwen 2.5-7B",  "pca_qwen_clean.png"),
    ]:
        if os.path.isdir(run_dir):
            reps, meta = load_centred(run_dir)
            plot_pca_clean(reps, meta, os.path.join(FIG_DIR, out_name), label)
        else:
            print(f"  WARNING: {run_dir} not found, skipping PCA for {label}")

    # ── Boxplot: n=50 MATH + n=10 FACT ────────────────────────────────────────
    llama_dirs = [
        "experiments/runs/run_003",
        "experiments/runs/run_003b",
        "experiments/runs/run_003c",
    ]
    qwen_dirs = [
        "experiments/runs/run_003_qwen",
        "experiments/runs/run_003c_qwen",
    ]

    llama_dirs_exist = [d for d in llama_dirs if os.path.isdir(d)]
    qwen_dirs_exist  = [d for d in qwen_dirs  if os.path.isdir(d)]

    if llama_dirs_exist and qwen_dirs_exist:
        print(f"  Llama: merging {llama_dirs_exist}")
        reps_llama, meta_llama = load_raw_multi(llama_dirs_exist)
        print(f"  Qwen:  merging {qwen_dirs_exist}")
        reps_qwen,  meta_qwen  = load_raw_multi(qwen_dirs_exist)

        dist_llama = compute_own_dists(reps_llama, meta_llama)
        dist_qwen  = compute_own_dists(reps_qwen,  meta_qwen)

        n_math_llama = sum(1 for r in meta_llama if r["form"]=="MATH" and r["answerable"]=="A")
        n_math_qwen  = sum(1 for r in meta_qwen  if r["form"]=="MATH" and r["answerable"]=="A")
        print(f"  Llama MATH-A: n={n_math_llama},  Qwen MATH-A: n={n_math_qwen}")

        plot_boxplot_combined(dist_llama, dist_qwen,
                              os.path.join(FIG_DIR, "boxplot_paper.png"))
    else:
        print(f"  WARNING: some run dirs missing. Llama found: {llama_dirs_exist}, Qwen found: {qwen_dirs_exist}")
        print("  Skipping boxplot generation.")

    print("Done.")


if __name__ == "__main__":
    main()
