"""
compute_auc_v2.py — Reliability Prediction Evaluation (3 forms × 3 models).

Computes ROC-AUC and F1 for:
  - Geometric baseline (own_dist = cosine dist to own-form A-centroid)
  - Refusal-keyword baseline (from generations.jsonl)

Produces a formatted table: Form × Model × (Geometry AUC/F1 | Refusal AUC/F1)

Usage:
    python compute_auc_v2.py           # all models, FACT+MATH+CODE where available
    python compute_auc_v2.py --latex   # also print LaTeX table

Run configuration (edit RUNS dict below if paths differ):
  MATH n=50 + FACT n=10 per model (merged across run dirs)
  CODE n=30 per model
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Run directory configuration ─────────────────────────────────────────────────
# Maps model → list of ALL run directories (merged + jointly centered per model).
# FACT and MATH are loaded together so centering is shared — matching analyze_controlled.py.
# CODE is separate (different form cluster).

MODEL_RUNS = {
    "llama":   {
        "FACT+MATH": ["experiments/runs/run_003",
                      "experiments/runs/run_003b",
                      "experiments/runs/run_003c"],
        "CODE":      ["experiments/runs/run_004_code_llama"],
    },
    "qwen":    {
        "FACT+MATH": ["experiments/runs/run_003_qwen",
                      "experiments/runs/run_003c_qwen"],
        "CODE":      ["experiments/runs/run_004_code_qwen"],
    },
    "mistral": {
        "FACT+MATH": ["experiments/runs/run_003_mistral",
                      "experiments/runs/run_003c_mistral"],
        "CODE":      ["experiments/runs/run_004_code_mistral"],
    },
}

MODELS = ["llama", "qwen", "mistral"]
FORMS  = ["MATH", "FACT", "CODE"]

MODEL_LABELS = {
    "llama":   "Llama-3.1-8B",
    "qwen":    "Qwen2.5-7B",
    "mistral": "Mistral-7B",
}

REFUSAL_KEYWORDS = [
    "cannot", "can't", "can not", "unable to", "don't know", "do not know",
    "undefined", "no answer", "not defined", "impossible", "unanswerable",
    "no solution", "doesn't exist", "does not exist", "not possible",
    "meaningless", "invalid", "inapplicable", "not applicable",
    "no such", "indeterminate", "ill-defined", "not a valid",
    "raises", "error", "exception", "typeerror", "valueerror",
    "indexerror", "attributeerror", "never terminates", "infinite loop",
]


# ── Helpers ──────────────────────────────────────────────────────────────────────

def load_raw(run_dir):
    reps = np.load(
        os.path.join(run_dir, "reps", "reps_last_raw.npy"), allow_pickle=False
    ).astype(np.float32)
    meta = []
    with open(os.path.join(run_dir, "reps", "meta.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                meta.append(json.loads(line))
    assert len(reps) == len(meta)
    return reps, meta


def load_gens(run_dir):
    path = os.path.join(run_dir, "generations.jsonl")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d["text"] for d in (json.loads(l) for l in f if l.strip())}


def load_and_merge(run_dirs, form_filter=None):
    """Load + merge multiple run dirs, optionally filtering to one form."""
    all_reps, all_meta, all_gens = [], [], {}
    offset = 0
    for d in run_dirs:
        if not os.path.isdir(d):
            continue
        reps_path = os.path.join(d, "reps", "reps_last_raw.npy")
        if not os.path.isfile(reps_path):
            continue
        r, m = load_raw(d)
        for row in m:
            row["row_idx"] += offset
        if form_filter:
            keep = [i for i, row in enumerate(m) if row["form"] == form_filter]
            r = r[keep]
            m = [m[i] for i in keep]
        all_reps.append(r)
        all_meta.extend(m)
        all_gens.update(load_gens(d))
        offset += len(m)

    if not all_reps:
        return None, None, None

    reps_raw = np.vstack(all_reps)
    reps     = reps_raw - reps_raw.mean(axis=0)
    return reps, all_meta, all_gens


def normalise(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_to_vec(X, vec):
    return 1.0 - normalise(X) @ (vec / (np.linalg.norm(vec) + 1e-12))


def refusal_score(text):
    t = text.lower()
    return int(any(kw in t for kw in REFUSAL_KEYWORDS))


def compute_auc_f1(y_true, y_score):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return float("nan"), float("nan")
    auc = roc_auc_score(y_true, y_score)
    thresh = (y_score[y_true == 0].mean() + y_score[y_true == 1].mean()) / 2.0
    y_pred = (y_score >= thresh).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return float(auc), float(f1)


# ── Core evaluation ───────────────────────────────────────────────────────────────

def evaluate_model_form(model_key, form):
    """
    Load all run directories for this model (jointly centered), then compute
    per-form metrics. FACT and MATH share the same joint centering; CODE is
    loaded separately.
    """
    cfg = MODEL_RUNS.get(model_key, {})

    if form == "CODE":
        run_dirs = cfg.get("CODE", [])
        reps, meta, gens = load_and_merge(run_dirs, form_filter=None)
    else:
        # Load FACT+MATH together for joint centering, then filter to target form
        run_dirs = cfg.get("FACT+MATH", [])
        reps_all, meta_all, gens = load_and_merge(run_dirs, form_filter=None)
        if reps_all is None:
            return None
        # Filter to the requested form after joint centering
        keep = [i for i, r in enumerate(meta_all) if r["form"] == form]
        if not keep:
            return None
        reps = reps_all[keep]
        meta = [meta_all[i] for i in keep]

    if reps is None or not meta:
        return None

    ans_arr = np.array([r["answerable"] for r in meta])
    mask_A  = ans_arr == "A"
    mask_U  = ans_arr == "U"
    n_A, n_U = mask_A.sum(), mask_U.sum()

    if n_A == 0 or n_U == 0:
        return None

    centroid_A = reps[mask_A].mean(axis=0)
    dist_A     = cosine_dist_to_vec(reps[mask_A], centroid_A)
    dist_U     = cosine_dist_to_vec(reps[mask_U], centroid_A)

    y_true   = np.concatenate([np.zeros(n_A), np.ones(n_U)])
    y_scores = np.concatenate([dist_A, dist_U])
    geo_auc, geo_f1 = compute_auc_f1(y_true, y_scores)

    # Refusal baseline
    meta_A = [r for r in meta if r["answerable"] == "A"]
    meta_U = [r for r in meta if r["answerable"] == "U"]
    ref_A  = [refusal_score(gens[r["id"]]) for r in meta_A if r["id"] in gens]
    ref_U  = [refusal_score(gens[r["id"]]) for r in meta_U if r["id"] in gens]

    ref_auc, ref_f1 = float("nan"), float("nan")
    n_ref_A_pos, n_ref_U_pos = 0, 0
    if ref_A and ref_U:
        ref_y  = np.concatenate([np.zeros(len(ref_A)), np.ones(len(ref_U))])
        ref_sc = np.array(ref_A + ref_U, dtype=float)
        ref_auc, ref_f1 = compute_auc_f1(ref_y, ref_sc)
        n_ref_A_pos = sum(ref_A)
        n_ref_U_pos = sum(ref_U)

    return {
        "n_A": int(n_A), "n_U": int(n_U),
        "dist_A_mean": float(dist_A.mean()),
        "dist_U_mean": float(dist_U.mean()),
        "geo_auc": geo_auc, "geo_f1": geo_f1,
        "ref_auc": ref_auc, "ref_f1": ref_f1,
        "n_ref_A_pos": n_ref_A_pos, "n_ref_U_pos": n_ref_U_pos,
        "n_ref_A": len(ref_A),  "n_ref_U": len(ref_U),
    }


# ── Printing ─────────────────────────────────────────────────────────────────────

def fmt(val, decimals=3):
    if val != val:  # nan
        return " " * (decimals + 2) + "—"
    return f"{val:.{decimals}f}"


def print_results(results):
    print("\n" + "=" * 90)
    print("  RELIABILITY PREDICTION EVALUATION: Geometry vs Refusal-Keyword Baseline")
    print("  (own_dist = cosine distance to own-form A-centroid, last layer)")
    print("=" * 90)

    header = (f"  {'Form':<6}  {'Model':<16}  {'n_A':>4}  {'n_U':>4}  "
              f"{'d(A)':>6}  {'d(U)':>6}  │  "
              f"{'Geo-AUC':>8}  {'Geo-F1':>7}  │  "
              f"{'Ref-AUC':>8}  {'Ref-F1':>7}")
    print(header)
    print("  " + "-" * 88)

    for form in FORMS:
        first_in_form = True
        for model in MODELS:
            r = results.get((model, form))
            if r is None:
                continue
            form_label = form if first_in_form else ""
            first_in_form = False
            print(
                f"  {form_label:<6}  {MODEL_LABELS[model]:<16}  "
                f"{r['n_A']:>4}  {r['n_U']:>4}  "
                f"{fmt(r['dist_A_mean'],3):>6}  {fmt(r['dist_U_mean'],3):>6}  │  "
                f"{fmt(r['geo_auc']):>8}  {fmt(r['geo_f1']):>7}  │  "
                f"{fmt(r['ref_auc']):>8}  {fmt(r['ref_f1']):>7}"
            )
        if not first_in_form:
            print("  " + "-" * 88)

    print()
    print("  Note: Refusal keywords include exception names (TypeError, ValueError, etc.)")
    print("  for CODE form. Geometry scores are pre-generation; Refusal requires generation.")


def print_latex(results):
    print("\n\n% ── LaTeX table ─────────────────────────────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llrr rr rr}")
    print(r"\toprule")
    print(r"Form & Model & $n_A$ & $n_U$ & \multicolumn{2}{c}{Geometry} & "
          r"\multicolumn{2}{c}{Refusal} \\")
    print(r"& & & & AUC & F1 & AUC & F1 \\")
    print(r"\midrule")

    for form in FORMS:
        rows_for_form = [(model, results.get((model, form))) for model in MODELS
                         if results.get((model, form)) is not None]
        if not rows_for_form:
            continue
        n_rows = len(rows_for_form)
        for i, (model, r) in enumerate(rows_for_form):
            form_cell = (f"\\multirow{{{n_rows}}}{{*}}{{{form}}}"
                         if i == 0 else "")
            geo_auc = f"{r['geo_auc']:.3f}" if r['geo_auc'] == r['geo_auc'] else "--"
            geo_f1  = f"{r['geo_f1']:.3f}"  if r['geo_f1']  == r['geo_f1']  else "--"
            ref_auc = f"{r['ref_auc']:.3f}" if r['ref_auc'] == r['ref_auc'] else "--"
            ref_f1  = f"{r['ref_f1']:.3f}"  if r['ref_f1']  == r['ref_f1']  else "--"
            print(f"{form_cell} & {MODEL_LABELS[model]} & "
                  f"{r['n_A']} & {r['n_U']} & "
                  f"{geo_auc} & {geo_f1} & "
                  f"{ref_auc} & {ref_f1} \\\\")
        print(r"\midrule")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Reliability prediction evaluation. Geometry = cosine distance "
          r"to answerable centroid (pre-generation, unsupervised). "
          r"Refusal = keyword-based classifier on model output. "
          r"Best per row in \textbf{bold}.}")
    print(r"\label{tab:auc}")
    print(r"\end{table}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX table.")
    parser.add_argument("--form", choices=FORMS,
                        help="Restrict to one form.")
    parser.add_argument("--model", choices=MODELS,
                        help="Restrict to one model.")
    args = parser.parse_args()

    models = [args.model] if args.model else [m for m in MODELS if m in MODEL_RUNS]
    forms  = [args.form]  if args.form  else FORMS

    results = {}
    for model in models:
        for form in forms:
            res = evaluate_model_form(model, form)
            if res is not None:
                results[(model, form)] = res

    print_results(results)

    if args.latex:
        print_latex(results)


if __name__ == "__main__":
    main()
