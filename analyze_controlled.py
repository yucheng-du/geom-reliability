"""
analyze_controlled.py — Generalized controlled answerability analysis.

Handles any set of forms (FACT / MATH / CODE) and any number of run directories
merged together. Replaces / supersedes analyze_003_full.py for multi-form,
multi-model experiments.

Usage examples:

  # Llama: FACT-10 + MATH-50
  python analyze_controlled.py \\
    --run-dirs experiments/runs/run_main_llama \\
    --label llama_math50_fact10

  # Llama CODE-30
  python analyze_controlled.py \\
    --run-dirs experiments/runs/run_code_llama \\
    --label llama_code

Output (in experiments/runs/results_{label}/):
  summary.txt     full text report
  cosine_dist_boxplot.png
  pca_full.png
"""

import argparse
import io
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FORM_COLORS = {
    "FACT": "#4C72B0",
    "MATH": "#55A868",
    "CODE": "#DD8452",
}


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="One or more run directories (merged in order)."
    )
    p.add_argument(
        "--label", default="analysis",
        help="Label for output directory and report header."
    )
    p.add_argument(
        "--forms", nargs="+", default=None,
        help="Restrict analysis to these forms (e.g. --forms MATH CODE). "
             "Default: all forms found in data."
    )
    p.add_argument(
        "--n-perm", type=int, default=5000,
        help="Number of permutations for permutation test (default: 5000)."
    )
    return p.parse_args()


# ── I/O ─────────────────────────────────────────────────────────────────────────

def load_raw(run_dir):
    reps = np.load(
        os.path.join(run_dir, "reps", "reps_last_raw.npy"),
        allow_pickle=False
    ).astype(np.float32)
    meta = []
    with open(os.path.join(run_dir, "reps", "meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    assert len(reps) == len(meta), f"len mismatch in {run_dir}"
    return reps, meta


def load_and_merge(run_dirs):
    all_reps, all_meta = [], []
    offset = 0
    for d in run_dirs:
        r, m = load_raw(d)
        for row in m:
            row["row_idx"] += offset
        all_reps.append(r)
        all_meta.extend(m)
        offset += len(m)
    reps_raw = np.vstack(all_reps)
    reps = reps_raw - reps_raw.mean(axis=0)
    return reps_raw, reps, all_meta


# ── Geometry ────────────────────────────────────────────────────────────────────

def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_to_vec(X, vec):
    Xn = normalise(X)
    vn = vec / (np.linalg.norm(vec) + 1e-12)
    return 1.0 - (Xn @ vn)


# ── Permutation test ────────────────────────────────────────────────────────────

def permutation_test(reps_A, reps_U, n_perm=5000, rng_seed=42):
    rng   = np.random.default_rng(rng_seed)
    n_A   = len(reps_A)
    pool  = np.vstack([reps_A, reps_U])

    centroid_obs = reps_A.mean(axis=0)
    dist_A_obs   = cosine_dist_to_vec(reps_A, centroid_obs).mean()
    dist_U_obs   = cosine_dist_to_vec(reps_U, centroid_obs).mean()
    obs_diff     = dist_U_obs - dist_A_obs

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(pool))
        pA   = pool[perm[:n_A]]
        pU   = pool[perm[n_A:]]
        c    = pA.mean(axis=0)
        if (cosine_dist_to_vec(pU, c).mean() - cosine_dist_to_vec(pA, c).mean()) >= obs_diff:
            count += 1

    return {
        "dist_A_mean": float(dist_A_obs),
        "dist_U_mean": float(dist_U_obs),
        "obs_diff":    float(obs_diff),
        "p_value":     count / n_perm,
        "n_A": n_A, "n_U": len(reps_U), "n_perm": n_perm,
    }


def cohen_d(a, b):
    """Cohen's d: mean(b) - mean(a) / pooled SD."""
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    return float((b.mean() - a.mean()) / (pooled_std + 1e-12))


# ── Cross-form assignment ────────────────────────────────────────────────────────

def cross_form_assignment(reps, meta, centroids_A, forms):
    cent_mat = np.stack([centroids_A[f] for f in forms])
    sims     = normalise(reps) @ normalise(cent_mat).T
    rows = []
    for i, row in enumerate(meta):
        sim_row     = sims[i]
        own_idx     = forms.index(row["form"]) if row["form"] in forms else -1
        nearest_idx = int(np.argmax(sim_row))
        rows.append({
            "id":           row["id"],
            "form":         row["form"],
            "answerable":   row["answerable"],
            "own_dist":     float(1.0 - sim_row[own_idx]) if own_idx >= 0 else float("nan"),
            "nearest_form": forms[nearest_idx],
            "nearest_dist": float(1.0 - sim_row[nearest_idx]),
            "misassigned":  forms[nearest_idx] != row["form"] if own_idx >= 0 else False,
        })
    return rows


# ── Figures ─────────────────────────────────────────────────────────────────────

def plot_boxplot(dist_by_group, forms, save_path, title_suffix=""):
    keys   = [(f, ans) for f in forms for ans in ["A", "U"]]
    labels = [f"{f}-{a}" for f, a in keys]
    colors = [FORM_COLORS.get(f, "#999999") for f, _ in keys]
    data   = [dist_by_group.get(k, np.array([])) for k in keys]

    # Position with gaps between form groups
    pos_map, p = {}, 1
    for i, (f, a) in enumerate(keys):
        if i > 0 and keys[i][0] != keys[i-1][0]:
            p += 1.5   # extra gap between form groups
        pos_map[i] = p
        p += 1
    pos = list(pos_map.values())

    fig, ax = plt.subplots(figsize=(max(6, len(keys) * 1.4), 5))
    bp = ax.boxplot(data, positions=pos, patch_artist=True, widths=0.6,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(0)
    for p_val, d, col in zip(pos, data, colors):
        if len(d):
            ax.scatter(
                np.full(len(d), p_val) + rng.uniform(-0.12, 0.12, len(d)),
                d, color=col, alpha=0.6, s=28, zorder=3
            )

    # Form-separator verticals
    prev_form = None
    for i, ((f, _), p_val) in enumerate(zip(keys, pos)):
        if prev_form and f != prev_form:
            ax.axvline((pos[i-1] + p_val) / 2, color="grey",
                       linestyle="--", linewidth=0.8)
        prev_form = f

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine distance to own-form A-centroid", fontsize=10)
    ax.set_title(f"A vs U deviation from A-centroid{title_suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pca(reps, meta, centroids_A, forms, save_path):
    cent_vecs = np.stack([centroids_A[f] for f in forms])
    all_data  = np.vstack([reps, cent_vecs])
    pca       = PCA(n_components=2, random_state=42).fit(all_data)
    coords    = pca.transform(all_data)
    var       = pca.explained_variance_ratio_

    pt_coords   = coords[:len(reps)]
    cent_coords = coords[len(reps):]

    fig, ax = plt.subplots(figsize=(7, 5))
    for form in forms:
        col = FORM_COLORS.get(form, "#999999")
        for ans, marker in [("A", "o"), ("U", "X")]:
            mask = np.array([(r["form"] == form and r["answerable"] == ans) for r in meta])
            ax.scatter(pt_coords[mask, 0], pt_coords[mask, 1],
                       c=col, marker=marker, s=65, alpha=0.75,
                       edgecolors="white" if ans == "A" else "black",
                       linewidths=0.5)
    for i, form in enumerate(forms):
        col = FORM_COLORS.get(form, "#999999")
        ax.scatter(cent_coords[i, 0], cent_coords[i, 1],
                   c=col, marker="*", s=300, edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.annotate(f"{form}\n(A-centroid)",
                    (cent_coords[i, 0], cent_coords[i, 1]),
                    textcoords="offset points", xytext=(5, 4), fontsize=8)

    handles = [mpatches.Patch(color=FORM_COLORS.get(f, "#999"), label=f) for f in forms]
    handles += [
        mlines.Line2D([], [], marker="o", color="grey", linestyle="None",
                      label="Answerable (A)"),
        mlines.Line2D([], [], marker="X", color="grey", linestyle="None",
                      label="Unanswerable (U)"),
    ]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=10)
    ax.set_title("PCA — form (colour) × answerability (marker)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    reps_raw, reps, meta = load_and_merge(args.run_dirs)
    n, d = reps.shape

    # Determine forms present
    all_forms_found = sorted(set(r["form"] for r in meta))
    forms = args.forms if args.forms else all_forms_found

    out_dir = f"experiments/runs/results_{args.label}"
    os.makedirs(out_dir, exist_ok=True)

    label_arr = np.array([r["form"]       for r in meta])
    ans_arr   = np.array([r["answerable"] for r in meta])

    buf = io.StringIO()
    def pr(line=""):
        print(line)
        buf.write(line + "\n")

    pr("=" * 72)
    pr(f"CONTROLLED ANSWERABILITY ANALYSIS  [{args.label}]")
    pr(f"{n} prompts × {d} dims | forms: {forms}")
    pr("=" * 72)

    counts = Counter((r["form"], r["answerable"]) for r in meta)
    for (form, ans), cnt in sorted(counts.items()):
        pr(f"  {form}-{ans}: n={cnt}")

    # ── [1] A-only centroids ───────────────────────────────────────────────────
    pr("\n[1] A-ONLY CENTROID NORMS")
    pr("-" * 72)
    centroids_A = {}
    for form in forms:
        if form not in all_forms_found:
            pr(f"  {form}: no data — skipped")
            continue
        mask = (label_arr == form) & (ans_arr == "A")
        if mask.sum() == 0:
            pr(f"  {form}: no A samples — skipped")
            continue
        centroids_A[form] = reps[mask].mean(axis=0)
        pr(f"  {form}  centroid norm={np.linalg.norm(centroids_A[form]):.4f}"
           f"  (n_A={mask.sum()})")

    active_forms = list(centroids_A.keys())

    # ── [2] Cosine distance to own-form A-centroid ─────────────────────────────
    pr("\n[2] COSINE DISTANCE TO OWN-FORM A-CENTROID  (A vs U)")
    pr("-" * 72)
    dist_by_group = {}
    pr(f"  {'Group':<14}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  n")
    pr("  " + "-" * 60)
    for form in active_forms:
        centroid = centroids_A[form]
        for ans in ["A", "U"]:
            mask  = (label_arr == form) & (ans_arr == ans)
            if mask.sum() == 0:
                continue
            dists = cosine_dist_to_vec(reps[mask], centroid)
            dist_by_group[(form, ans)] = dists
            pr(f"  {form}-{ans:<12}  {dists.mean():>8.4f}  {dists.std():>8.4f}"
               f"  {dists.min():>8.4f}  {dists.max():>8.4f}  {len(dists)}")

    # ── [3] Permutation test ───────────────────────────────────────────────────
    pr("\n[3] PERMUTATION TEST: U farther from A-centroid than A?")
    pr(f"    one-tailed, n_perm={args.n_perm}, centroid recomputed each permutation")
    pr("-" * 72)
    pr(f"  {'Form':<8}  {'n_A':>4}  {'n_U':>4}  {'dist_A':>8}  {'dist_U':>8}"
       f"  {'diff(U-A)':>10}  {'p-value':>8}  {'d':>6}")
    pr("  " + "-" * 68)
    for form in active_forms:
        mask_A = (label_arr == form) & (ans_arr == "A")
        mask_U = (label_arr == form) & (ans_arr == "U")
        if mask_A.sum() == 0 or mask_U.sum() == 0:
            continue
        r   = permutation_test(reps[mask_A], reps[mask_U], args.n_perm)
        d_A = dist_by_group.get((form, "A"), np.array([]))
        d_U = dist_by_group.get((form, "U"), np.array([]))
        cd  = cohen_d(d_A, d_U)
        sig = ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else "ns"))
        pr(f"  {form:<8}  {r['n_A']:>4}  {r['n_U']:>4}"
           f"  {r['dist_A_mean']:>8.4f}  {r['dist_U_mean']:>8.4f}"
           f"  {r['obs_diff']:>+10.4f}  {r['p_value']:>7.4f} {sig}  {cd:>+6.2f}")

    # ── [4] Cross-form misassignment ───────────────────────────────────────────
    pr("\n[4] CROSS-FORM MISASSIGNMENT (nearest A-centroid)")
    pr("-" * 72)
    if len(active_forms) > 1:
        assign = cross_form_assignment(reps, meta, centroids_A, active_forms)
        for form in active_forms:
            for ans in ["A", "U"]:
                sub = [r for r in assign if r["form"] == form and r["answerable"] == ans]
                mis = sum(r["misassigned"] for r in sub)
                pr(f"  {form}-{ans}  misassigned={mis}/{len(sub)}")

        pr(f"\n  Per-prompt detail (U only, own_dist > 0.8):")
        pr(f"  {'id':<8}  {'form':<6}  {'own_dist':>9}  {'nearest':>8}  drift?")
        pr("  " + "-" * 50)
        for r in sorted(assign, key=lambda x: -x["own_dist"]):
            if r["answerable"] == "U" and (r["misassigned"] or r["own_dist"] > 0.8):
                flag = " ← DRIFT" if r["misassigned"] else ""
                pr(f"  {r['id']:<8}  {r['form']:<6}  {r['own_dist']:>9.4f}"
                   f"  {r['nearest_form']:>8}{flag}")
    else:
        pr("  (only one form present — cross-form assignment skipped)")

    # ── [5] Norm statistics ────────────────────────────────────────────────────
    pr("\n[5] NORM STATISTICS (raw reps, before mean-centering)")
    pr("-" * 72)
    pr(f"  {'Group':<14}  {'mean norm':>10}  {'std':>8}")
    for form in active_forms:
        for ans in ["A", "U"]:
            mask  = (label_arr == form) & (ans_arr == ans)
            if mask.sum() == 0:
                continue
            norms = np.linalg.norm(reps_raw[mask], axis=1)
            pr(f"  {form}-{ans:<12}  {norms.mean():>10.2f}  {norms.std():>8.2f}")

    # ── Figures ────────────────────────────────────────────────────────────────
    plot_boxplot(dist_by_group, active_forms,
                 os.path.join(out_dir, "cosine_dist_boxplot.png"),
                 f" ({args.label})")
    if len(active_forms) >= 2:
        plot_pca(reps, meta, centroids_A, active_forms,
                 os.path.join(out_dir, "pca_full.png"))

    pr("\n" + "=" * 72)
    pr(f"Output: {out_dir}/")

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as sf:
        sf.write(buf.getvalue())
    pr(f"Summary written.")


if __name__ == "__main__":
    main()
