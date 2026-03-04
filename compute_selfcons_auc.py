"""
compute_selfcons_auc.py — Evaluate self-consistency as answerability predictor.

Self-consistency scores (higher = more uncertain = more likely unanswerable):

  answer_disagree  (MATH / CODE only)
    Extract the "core answer token" (number for MATH, exception/value for CODE)
    from each sample, compute 1 - (majority_count / N).
    Uses the LAST line of the response (where models typically put the final
    answer), not the first number in the full text.

  rouge_disagree   (all forms; primary score for FACT)
    Compute 1 - mean pairwise ROUGE-1 F1 over all response pairs.
    Also applied to the last-line excerpt of each response for stability.

Final form-conditional score used for the main AUC comparison:
  MATH / CODE : answer_disagree  (answer extraction is reliable)
  FACT        : rouge_disagree   (no reliable answer token to extract)

F1 reporting: "best F1" scanned over all unique score thresholds on the same
set (oracle threshold; noted in paper as upper-bound F1 estimate).

Usage:
  python compute_selfcons_auc.py --latex \\
      --sc-file experiments/selfconsistency/llama_math50_fact10.jsonl \\
      --run-dirs experiments/runs/run_003 experiments/runs/run_003b experiments/runs/run_003c \\
      --model llama --label llama_math50_fact10

  python compute_selfcons_auc.py --latex \\
      --sc-file experiments/selfconsistency/qwen_code30.jsonl \\
      --run-dirs experiments/runs/run_004_code_qwen \\
      --model qwen --label qwen_code30
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

# ── response pre-processing ───────────────────────────────────────────────────

def extract_answer_line(text: str) -> str:
    """
    Return the most informative line for answer comparison.
    Priority:
      1. A line containing 'final answer' (case-insensitive)
      2. The last non-empty line (models often put the answer at the end)
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if "final answer" in line.lower() or "the answer is" in line.lower():
            return line
    return lines[-1] if lines else text


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

CODE_KEYWORDS = [
    "valueerror", "typeerror", "zerodivisionerror", "overflowerror",
    "runtimeerror", "stopiteration", "none", "true", "false",
    "notimplemented", "inf", "nan",
]


def extract_core_answer(text: str, form: str) -> str:
    """
    Extract the most salient answer token for majority-vote comparison.
    Only called for MATH and CODE forms.
    """
    # Use the answer line, not the full text
    line = extract_answer_line(text).lower()

    if form == "MATH":
        # Last number in the answer line is more reliable than first
        nums = NUMBER_RE.findall(line)
        if nums:
            return nums[-1]

    if form == "CODE":
        for kw in CODE_KEYWORDS:
            if kw in line:
                return kw
        nums = NUMBER_RE.findall(line)
        if nums:
            return nums[-1]

    # Fallback for both: first 4 normalised words of answer line
    words = re.sub(r"[^a-z0-9 ]", "", line).split()
    return " ".join(words[:4]) if words else line[:20]


# ── self-consistency scores ───────────────────────────────────────────────────

def answer_disagreement(responses: list[str], form: str) -> float:
    """
    1 - (majority_answer_count / N). Range [0, 1].
    Only reliable for MATH and CODE.
    """
    n = len(responses)
    if n <= 1:
        return 0.0
    cores = [extract_core_answer(r, form) for r in responses]
    majority_count = Counter(cores).most_common(1)[0][1]
    return 1.0 - majority_count / n


def tokenize_simple(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).split()


def rouge1_f1(ref_tokens: list, hyp_tokens: list) -> float:
    if not ref_tokens or not hyp_tokens:
        return 0.0
    ref_c = Counter(ref_tokens)
    hyp_c = Counter(hyp_tokens)
    overlap = sum((ref_c & hyp_c).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(hyp_tokens)
    r = overlap / len(ref_tokens)
    return 2 * p * r / (p + r)


def rouge_disagreement(responses: list[str]) -> float:
    """
    1 - mean pairwise ROUGE-1 F1 over answer-line excerpts. Range [0, 1].
    Used as the primary SC score for FACT; supplementary for MATH/CODE.
    """
    n = len(responses)
    if n <= 1:
        return 0.0
    # Apply answer-line extraction before computing overlap
    excerpts = [tokenize_simple(extract_answer_line(r)) for r in responses]
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(rouge1_f1(excerpts[i], excerpts[j]))
    mean_overlap = sum(scores) / len(scores) if scores else 0.0
    return 1.0 - mean_overlap


def sc_score(responses: list[str], form: str) -> float:
    """
    Form-conditional self-consistency disagreement score.
      MATH / CODE : answer_disagree  (answer tokens are extractable)
      FACT        : rouge_disagree   (no reliable answer token)
    """
    if form in ("MATH", "CODE"):
        return answer_disagreement(responses, form)
    return rouge_disagreement(responses)


# ── refusal baseline ──────────────────────────────────────────────────────────

REFUSAL_KEYWORDS = [
    "undefined", "cannot", "doesn't exist", "no such", "not defined",
    "infinite", "valueerror", "typeerror", "zerodivisionerror",
    "not possible", "impossible", "no answer", "no solution",
    "does not exist", "can't", "there is no",
]


def refusal_score(responses: list[str]) -> float:
    """1 if ANY sample contains a refusal keyword, else 0."""
    combined = " ".join(responses).lower()
    return float(any(kw in combined for kw in REFUSAL_KEYWORDS))


# ── geometry helpers ──────────────────────────────────────────────────────────

def normalise(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_to_vec(X: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return 1.0 - normalise(X) @ (vec / (np.linalg.norm(vec) + 1e-12))


def load_raw_multi(run_dirs: list[str]):
    seen: set[str] = set()
    all_reps, all_meta = [], []
    offset = 0
    for d in run_dirs:
        reps = np.load(
            os.path.join(d, "reps", "reps_last_raw.npy"), allow_pickle=False
        ).astype(np.float32)
        with open(os.path.join(d, "reps", "meta.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        for i, r in enumerate(rows):
            if r["id"] not in seen:
                seen.add(r["id"])
                r["_idx"] = offset + i
                all_meta.append(r)
                all_reps.append(reps[i])
        offset += len(rows)
    raw = np.stack(all_reps)
    return raw - raw.mean(axis=0), all_meta


def compute_own_dist(reps: np.ndarray, meta: list) -> dict[str, float]:
    label_arr = np.array([r["form"]       for r in meta])
    ans_arr   = np.array([r["answerable"] for r in meta])
    own_dists: dict[str, float] = {}
    for form in sorted(set(label_arr)):
        mask_A = (label_arr == form) & (ans_arr == "A")
        if mask_A.sum() == 0:
            continue
        centroid = reps[mask_A].mean(axis=0)
        mask_all = label_arr == form
        dists = cosine_dist_to_vec(reps[mask_all], centroid)
        for r, d in zip(np.array(meta)[mask_all], dists):
            own_dists[r["id"]] = float(d)
    return own_dists


# ── AUC / F1 helpers ──────────────────────────────────────────────────────────

def compute_auc_best_f1(
    scores: list[float], labels: list[int]
) -> tuple[float, float]:
    """
    Returns (ROC-AUC, best-F1).
    Best-F1 is the maximum F1 over all unique score thresholds (oracle threshold).
    This is standard for controlled evaluation; note in paper as oracle threshold.
    """
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")
    # Scan all unique thresholds
    best_f1 = 0.0
    for t in sorted(set(scores)):
        preds = [1 if s >= t else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        best_f1 = max(best_f1, f1)
    return auc, best_f1


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sc-file",   required=True)
    parser.add_argument("--run-dirs",  nargs="+", required=True)
    parser.add_argument("--model",     required=True)
    parser.add_argument("--label",     default="")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Expected samples per prompt; incomplete prompts "
                             "are excluded from evaluation.")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    # ── Load self-consistency responses ───────────────────────────────────────
    print(f"Loading: {args.sc_file}")
    sc_data: dict[str, list[str]] = defaultdict(list)
    sc_meta: dict[str, dict] = {}
    with open(args.sc_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sc_data[r["id"]].append(r["response"])
            sc_meta[r["id"]] = {"form": r["form"], "answerable": r["answerable"]}

    # Keep only prompts with exactly n_samples responses
    complete_ids = {
        pid for pid, samples in sc_data.items()
        if len(samples) == args.n_samples
    }
    incomplete = len(sc_data) - len(complete_ids)
    if incomplete:
        print(f"  Excluded {incomplete} prompts with != {args.n_samples} samples")
    print(f"  Complete prompts: {len(complete_ids)}")

    # ── Load geometry ─────────────────────────────────────────────────────────
    print(f"Loading geometry from {args.run_dirs}")
    reps, meta = load_raw_multi(args.run_dirs)
    own_dists  = compute_own_dist(reps, meta)

    # ── Align ─────────────────────────────────────────────────────────────────
    common_ids = sorted(complete_ids & set(own_dists.keys()))
    print(f"  Common prompts (complete SC ∩ geometry): {len(common_ids)}\n")

    forms = sorted(set(sc_meta[pid]["form"] for pid in common_ids))

    # ── Per-form evaluation ───────────────────────────────────────────────────
    col = 10
    print(f"{'Form':<8}  {'n_A':>4}  {'n_U':>4}  "
          f"{'Geo AUC':>{col}}  {'Geo F1':>{col}}  "
          f"{'SC AUC':>{col}}  {'SC F1*':>{col}}  "
          f"{'Ref AUC':>{col}}  {'Ref F1*':>{col}}")
    print("-" * 90)

    results: dict[str, dict] = {}
    for form in forms:
        ids_A = [p for p in common_ids
                 if sc_meta[p]["form"] == form and sc_meta[p]["answerable"] == "A"]
        ids_U = [p for p in common_ids
                 if sc_meta[p]["form"] == form and sc_meta[p]["answerable"] == "U"]
        if not ids_A or not ids_U:
            continue

        all_ids = ids_A + ids_U
        labels  = [0] * len(ids_A) + [1] * len(ids_U)

        geo_sc  = [own_dists[p]                        for p in all_ids]
        sc_sc   = [sc_score(sc_data[p], form)          for p in all_ids]
        ref_sc  = [refusal_score(sc_data[p])           for p in all_ids]

        geo_auc, geo_f1 = compute_auc_best_f1(geo_sc,  labels)
        sc_auc,  sc_f1  = compute_auc_best_f1(sc_sc,   labels)
        ref_auc, ref_f1 = compute_auc_best_f1(ref_sc,  labels)

        results[form] = dict(
            n_A=len(ids_A), n_U=len(ids_U),
            geo_auc=geo_auc, geo_f1=geo_f1,
            sc_auc=sc_auc,   sc_f1=sc_f1,
            ref_auc=ref_auc, ref_f1=ref_f1,
            sc_type="answer_disagree" if form in ("MATH","CODE") else "rouge_disagree",
        )

        def f(v):
            return f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else "  nan"

        print(f"{form:<8}  {len(ids_A):>4}  {len(ids_U):>4}  "
              f"{f(geo_auc):>{col}}  {f(geo_f1):>{col}}  "
              f"{f(sc_auc):>{col}}  {f(sc_f1):>{col}}  "
              f"{f(ref_auc):>{col}}  {f(ref_f1):>{col}}  "
              f"  [SC={results[form]['sc_type']}]")

    print("\n* F1 = best F1 over all unique score thresholds (oracle threshold).")

    # ── Supplementary: ROUGE disagree for MATH/CODE (as secondary check) ─────
    print("\n── Supplementary: rouge_disagree for ALL forms (secondary) ──")
    for form in forms:
        if form not in results:
            continue
        ids_A = [p for p in common_ids
                 if sc_meta[p]["form"] == form and sc_meta[p]["answerable"] == "A"]
        ids_U = [p for p in common_ids
                 if sc_meta[p]["form"] == form and sc_meta[p]["answerable"] == "U"]
        all_ids = ids_A + ids_U
        labels  = [0] * len(ids_A) + [1] * len(ids_U)
        rou_sc  = [rouge_disagreement(sc_data[p]) for p in all_ids]
        rou_auc, rou_f1 = compute_auc_best_f1(rou_sc, labels)
        print(f"  {form:<8}  rouge_disagree  AUC={rou_auc:.3f}  F1={rou_f1:.3f}")

    # ── LaTeX output ──────────────────────────────────────────────────────────
    if args.latex:
        MODEL_LABEL = {
            "llama": "Llama", "qwen": "Qwen", "mistral": "Mistral"
        }.get(args.model, args.model)

        print("\n% ── LaTeX rows for Table (tab:auc_sc) ─────────────────────────")
        print("% Columns: Form | Model | n | Geo-AUC | Geo-F1* | SC-AUC | SC-F1* | Ref-AUC | Ref-F1*")
        for form, r in results.items():
            n = r["n_A"] + r["n_U"]

            def bold_if_best(val, *others):
                s = f"{val:.3f}"
                if val == max(val, *others):
                    return f"\\textbf{{{s}}}"
                return s

            geo_auc_s = bold_if_best(r["geo_auc"], r["sc_auc"], r["ref_auc"])
            sc_auc_s  = bold_if_best(r["sc_auc"],  r["geo_auc"], r["ref_auc"])
            ref_auc_s = bold_if_best(r["ref_auc"], r["geo_auc"], r["sc_auc"])

            print(f"  \\textsc{{{form}}} & {MODEL_LABEL} & {n} "
                  f"& {geo_auc_s} & {r['geo_f1']:.3f} "
                  f"& {sc_auc_s} & {r['sc_f1']:.3f} "
                  f"& {ref_auc_s} & {r['ref_f1']:.3f} \\\\")

    print(f"\nLabel: {args.label}")


if __name__ == "__main__":
    main()
