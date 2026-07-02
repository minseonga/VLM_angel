#!/usr/bin/env python
"""
Compare Stage D.2 causal intervention runs against pre-committed criteria.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline/selective/random causal intervention runs.")
    parser.add_argument("--baseline-summary", type=str, required=True)
    parser.add_argument("--selective-summary", type=str, required=True)
    parser.add_argument("--random-summaries", nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="stage1_causal_comparison")
    parser.add_argument("--min-relative-chairi-reduction", type=float, default=0.15)
    parser.add_argument("--harm-tolerance", type=float, default=0.0)
    return parser.parse_args()


def load_summary(path):
    with open(path) as f:
        summary = json.load(f)
    details_path = summary["outputs"]["chair_details"]
    with open(details_path) as f:
        details = json.load(f)
    return summary, details


def sentence_by_image(details):
    return {
        int(item["image_id"]): item
        for item in details.get("sentences", [])
    }


def safe_metric(sentence, key):
    return float(sentence.get("metrics", {}).get(key, 0.0))


def paired_regressions(baseline_details, variant_details):
    base = sentence_by_image(baseline_details)
    var = sentence_by_image(variant_details)
    common = sorted(set(base) & set(var))

    harmful = 0
    fixed = 0
    recall_regressions = 0
    f1_regressions = 0
    chairi_worse = 0
    chairi_better = 0

    for image_id in common:
        b = base[image_id]
        v = var[image_id]
        b_chairs = safe_metric(b, "CHAIRs")
        v_chairs = safe_metric(v, "CHAIRs")
        b_chairi = safe_metric(b, "CHAIRi")
        v_chairi = safe_metric(v, "CHAIRi")
        if b_chairs == 0 and v_chairs > 0:
            harmful += 1
        if b_chairs > 0 and v_chairs == 0:
            fixed += 1
        if safe_metric(v, "Recall") < safe_metric(b, "Recall"):
            recall_regressions += 1
        if safe_metric(v, "F1") < safe_metric(b, "F1"):
            f1_regressions += 1
        if v_chairi > b_chairi:
            chairi_worse += 1
        if v_chairi < b_chairi:
            chairi_better += 1

    return {
        "paired_n": len(common),
        "harmful_regression_count": harmful,
        "fixed_hallucination_count": fixed,
        "recall_regression_count": recall_regressions,
        "f1_regression_count": f1_regressions,
        "chairi_worse_count": chairi_worse,
        "chairi_better_count": chairi_better,
    }


def run_row(name, summary, baseline_summary, baseline_details, details, group):
    metrics = summary["overall_metrics"]
    base_metrics = baseline_summary["overall_metrics"]
    chairi = float(metrics["CHAIRi"])
    base_chairi = float(base_metrics["CHAIRi"])
    chairi_improvement = base_chairi - chairi
    rel_reduction = chairi_improvement / base_chairi if base_chairi > 0 else 0.0
    regressions = paired_regressions(baseline_details, details)
    return {
        "name": name,
        "group": group,
        "CHAIRs": float(metrics["CHAIRs"]),
        "CHAIRi": chairi,
        "Recall": float(metrics["Recall"]),
        "Precision": float(metrics["Precision"]),
        "F1": float(metrics["F1"]),
        "Len": float(metrics["Len"]),
        "chairi_improvement_vs_baseline": chairi_improvement,
        "chairi_relative_reduction_vs_baseline": rel_reduction,
        "num_heads": int(summary.get("num_heads", 0)),
        "total_gated_steps": int(summary.get("total_gated_steps", 0)),
        **regressions,
    }


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def write_csv(rows, path):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary, baseline_details = load_summary(args.baseline_summary)
    selective_summary, selective_details = load_summary(args.selective_summary)
    random_runs = [load_summary(path) for path in args.random_summaries]

    rows = []
    rows.append(run_row("baseline", baseline_summary, baseline_summary, baseline_details, baseline_details, "baseline"))
    selective_row = run_row(
        selective_summary["variant"],
        selective_summary,
        baseline_summary,
        baseline_details,
        selective_details,
        "selective",
    )
    rows.append(selective_row)

    random_rows = []
    for summary, details in random_runs:
        row = run_row(
            summary["variant"],
            summary,
            baseline_summary,
            baseline_details,
            details,
            "random",
        )
        random_rows.append(row)
        rows.append(row)

    random_improvements = [row["chairi_improvement_vs_baseline"] for row in random_rows]
    random_harms = [row["harmful_regression_count"] for row in random_rows]
    random_chairi = [row["CHAIRi"] for row in random_rows]
    random_improve_mean, random_improve_std = mean_std(random_improvements)
    random_harm_mean, random_harm_std = mean_std(random_harms)
    random_chairi_mean, random_chairi_std = mean_std(random_chairi)

    crit1 = selective_row["chairi_relative_reduction_vs_baseline"] >= args.min_relative_chairi_reduction
    crit2 = selective_row["chairi_improvement_vs_baseline"] > random_improve_mean + 2.0 * random_improve_std
    crit3_strict = selective_row["harmful_regression_count"] <= random_harm_mean + args.harm_tolerance
    crit3_2std = selective_row["harmful_regression_count"] <= random_harm_mean + 2.0 * random_harm_std + args.harm_tolerance

    comparison = {
        "precommitted_success_criteria": {
            "D-crit-1": {
                "description": "Selective reduces CHAIR_i by at least the configured relative threshold.",
                "threshold": args.min_relative_chairi_reduction,
                "value": selective_row["chairi_relative_reduction_vs_baseline"],
                "passed": bool(crit1),
            },
            "D-crit-2": {
                "description": "Selective CHAIR_i improvement exceeds random improvement mean + 2 std.",
                "random_improvement_mean": random_improve_mean,
                "random_improvement_std": random_improve_std,
                "selective_improvement": selective_row["chairi_improvement_vs_baseline"],
                "passed": bool(crit2),
            },
            "D-crit-3": {
                "description": "Selective harmful regression count is not larger than random-head mean.",
                "random_harm_mean": random_harm_mean,
                "random_harm_std": random_harm_std,
                "selective_harmful_regression_count": selective_row["harmful_regression_count"],
                "passed_strict_mean": bool(crit3_strict),
                "passed_mean_plus_2std": bool(crit3_2std),
                "passed": bool(crit3_strict),
            },
        },
        "overall_pass": bool(crit1 and crit2 and crit3_strict),
        "random_summary": {
            "CHAIRi_mean": random_chairi_mean,
            "CHAIRi_std": random_chairi_std,
            "chairi_improvement_mean": random_improve_mean,
            "chairi_improvement_std": random_improve_std,
            "harmful_regression_mean": random_harm_mean,
            "harmful_regression_std": random_harm_std,
        },
        "rows": rows,
    }

    write_csv(rows, output_dir / "stage1_d2_causal_comparison_rows.csv")
    with open(output_dir / "stage1_d2_causal_comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps({
        "overall_pass": comparison["overall_pass"],
        "criteria": comparison["precommitted_success_criteria"],
        "random_summary": comparison["random_summary"],
        "outputs": {
            "summary": str(output_dir / "stage1_d2_causal_comparison_summary.json"),
            "rows": str(output_dir / "stage1_d2_causal_comparison_rows.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
