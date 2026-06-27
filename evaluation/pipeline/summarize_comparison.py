#!/usr/bin/env python3
"""
Summarize Qwen2 vs GPT-2 downstream vaccine-selection results across years.
Usage: python3 summarize_comparison.py 2012 2013 2014 2015 2016 2017
Run from ~/vaxseer_run/evaluation
Auto-discovers the virus_set directory per year (no hardcoded strings).
"""
import sys, os, glob
import pandas as pd
import numpy as np

SUBTYPE = "a_h3n2"
GPT2_BASE = "../runs/pipeline"
QWEN2_BASE = "../runs/pipeline_qwen2"


def find_score_csv(base, year, vaccine_kind):
    """vaccine_kind: 'vaxseer' (the 3Y selected set) or 'who'.
    Returns the vaccine_score_and_gt.csv path, discovered by glob."""
    year_dir = f"{base}/{year}-02/{SUBTYPE}"
    if vaccine_kind == "who":
        pattern = f"{year_dir}/vaccine_set=who___virus_set=*/vaccine_scores/*/vaccine_score_and_gt.csv"
    else:
        # the selected set has vaccine_set equal to the virus_set (both 3Y), not 'who'
        pattern = f"{year_dir}/vaccine_set=*___virus_set=*/vaccine_scores/*/vaccine_score_and_gt.csv"
    hits = [p for p in glob.glob(pattern) if (vaccine_kind == "who") == ("vaccine_set=who" in p)]
    return hits[0] if hits else None


def chosen_score(csv_path):
    """Return (n_scored, model_chosen_gt_score, oracle_best_gt_score) or None."""
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["gt_score_seq", "score"])
    if len(df) == 0:
        return None
    chosen = df.iloc[int(np.argmin(df["score"].to_numpy()))]
    oracle = float(df["gt_score_seq"].min())
    return len(df), float(chosen["gt_score_seq"]), oracle


def who_median(csv_path):
    if csv_path is None or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["gt_score_seq"])
    if len(df) == 0:
        return None
    return float(df["gt_score_seq"].median())


def main():
    years = sys.argv[1:] if len(sys.argv) > 1 else ["2012", "2013", "2014", "2015", "2016", "2017"]

    rows = []
    gpt2_scores, qwen2_scores, who_scores = [], [], []

    for year in years:
        g_csv = find_score_csv(GPT2_BASE, year, "vaxseer")
        q_csv = find_score_csv(QWEN2_BASE, year, "vaxseer")
        w_csv = find_score_csv(GPT2_BASE, year, "who")  # WHO set is model-independent

        g = chosen_score(g_csv)
        q = chosen_score(q_csv)
        w = who_median(w_csv)

        g_val = g[1] if g else None
        q_val = q[1] if q else None
        oracle = g[2] if g else (q[2] if q else None)

        rows.append({
            "year": year,
            "gpt2": g_val,
            "qwen2": q_val,
            "who": w,
            "oracle": oracle,
            "n_gpt2": g[0] if g else 0,
            "n_qwen2": q[0] if q else 0,
        })
        if g_val is not None and q_val is not None:
            gpt2_scores.append(g_val)
            qwen2_scores.append(q_val)
            if w is not None:
                who_scores.append(w)

    # ---- print table ----
    print("\n================ PER-YEAR DOWNSTREAM COMPARISON ================")
    print(f"{'Year':<6}{'GPT-2':>10}{'Qwen2':>10}{'WHO':>10}{'Oracle':>10}{'n':>6}  Winner")
    print("-" * 70)
    for r in rows:
        g = f"{r['gpt2']:.4f}" if r['gpt2'] is not None else "   --   "
        q = f"{r['qwen2']:.4f}" if r['qwen2'] is not None else "   --   "
        w = f"{r['who']:.4f}" if r['who'] is not None else "   --   "
        o = f"{r['oracle']:.4f}" if r['oracle'] is not None else "   --   "
        if r['gpt2'] is not None and r['qwen2'] is not None:
            if abs(r['gpt2'] - r['qwen2']) < 1e-6:
                win = "tie"
            elif r['qwen2'] < r['gpt2']:
                win = "Qwen2"
            else:
                win = "GPT-2"
        else:
            win = "(incomplete)"
        print(f"{r['year']:<6}{g:>10}{q:>10}{w:>10}{o:>10}{r['n_gpt2']:>6}  {win}")

    # ---- aggregate ----
    print("\n================ AGGREGATE (years with both models) ================")
    if gpt2_scores:
        n = len(gpt2_scores)
        print(f"Years compared: {n}")
        print(f"  GPT-2  mean chosen gt_score: {np.mean(gpt2_scores):.4f}  (lower=better)")
        print(f"  Qwen2  mean chosen gt_score: {np.mean(qwen2_scores):.4f}")
        if who_scores:
            print(f"  WHO    mean median  gt_score: {np.mean(who_scores):.4f}")
        gwins = sum(1 for a, b in zip(gpt2_scores, qwen2_scores) if b > a + 1e-6)
        qwins = sum(1 for a, b in zip(gpt2_scores, qwen2_scores) if b < a - 1e-6)
        ties = n - gwins - qwins
        print(f"\n  Head-to-head:  Qwen2 better in {qwins}, GPT-2 better in {gwins}, tie in {ties}")
        diff = np.mean(qwen2_scores) - np.mean(gpt2_scores)
        verdict = ("Qwen2 better on average" if diff < -1e-6
                   else "GPT-2 better on average" if diff > 1e-6
                   else "exact average tie")
        print(f"  Mean difference (Qwen2 - GPT-2): {diff:+.4f}  ->  {verdict}")
        if who_scores:
            beat_who_g = np.mean(gpt2_scores) < np.mean(who_scores)
            beat_who_q = np.mean(qwen2_scores) < np.mean(who_scores)
            print(f"\n  Both beat WHO on average?  GPT-2: {beat_who_g}   Qwen2: {beat_who_q}")
    else:
        print("No years had both models complete. Check the pipeline logs.")
    print("=" * 68)


if __name__ == "__main__":
    main()
