#!/usr/bin/env python
"""
Identify wrong-direction heads for Stage D.1.

The script uses the existing head-logit contribution trace. It compares each
head's contribution on hallucinated object tokens against the grounded
reference distribution, then exports a pre-registered H_wrong set and per-head
activation thresholds for gated ablation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, write_csv
from stage1_common import torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="D.1: identify wrong-direction contribution heads.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs/stage1_head_logit_contrib.pt")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_d1_wrong_heads")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument(
        "--selection-direction",
        choices=["hallucinated_high", "grounded_high", "abs"],
        default="hallucinated_high",
        help="hallucinated_high is the pre-registered H_wrong ablation target.",
    )
    parser.add_argument("--effect-top-frac", type=float, default=0.10)
    parser.add_argument("--min-heads", type=int, default=10)
    parser.add_argument("--max-heads", type=int, default=30)
    parser.add_argument(
        "--activation-ref",
        choices=["grounded", "all"],
        default="grounded",
        help="Reference distribution used for contribution-gated ablation thresholds.",
    )
    parser.add_argument("--activation-quantile", type=float, default=0.90)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=927)
    return parser.parse_args()


def stable_train_flag(image_id, seed, train_frac):
    key = f"{int(image_id)}:{int(seed)}".encode("utf-8")
    value = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return value < train_frac


def split_records(records, train_frac, seed):
    train = [r for r in records if stable_train_flag(r["image_id"], seed, train_frac)]
    test = [r for r in records if not stable_train_flag(r["image_id"], seed, train_frac)]
    return train, test


def selected_score(row, direction):
    if direction == "hallucinated_high":
        return row["h_minus_g"]
    if direction == "grounded_high":
        return -row["h_minus_g"]
    return abs(row["h_minus_g"])


def contribution_matrix(records):
    return np.stack([
        record["head_logit_contrib"].detach().cpu().numpy()[:, :, 0]
        for record in records
    ])


def labels_hallucinated(records):
    return np.array([1 if int(record["label"]) == 0 else 0 for record in records], dtype=np.int64)


def head_stats(records, start_layer, end_layer, activation_ref, activation_quantile):
    if not records:
        return []
    matrices = contribution_matrix(records)
    labels = labels_hallucinated(records)
    rows = []
    for layer_idx in range(start_layer, end_layer + 1):
        for head_idx in range(matrices.shape[2]):
            vals = matrices[:, layer_idx, head_idx]
            grounded = vals[labels == 0]
            hallucinated = vals[labels == 1]
            if len(grounded) == 0 or len(hallucinated) == 0:
                continue
            pooled = math.sqrt((grounded.std() ** 2 + hallucinated.std() ** 2) / 2.0)
            h_minus_g = float(hallucinated.mean() - grounded.mean())
            auc = binary_auc(labels, vals)
            ref_vals = grounded if activation_ref == "grounded" else vals
            threshold = float(np.quantile(ref_vals, activation_quantile))
            rows.append({
                "layer": int(layer_idx),
                "head": int(head_idx),
                "n_grounded": int(len(grounded)),
                "n_hallucinated": int(len(hallucinated)),
                "grounded_mean": float(grounded.mean()),
                "hallucinated_mean": float(hallucinated.mean()),
                "h_minus_g": h_minus_g,
                "cohens_d_h_minus_g": float(h_minus_g / pooled) if pooled > 0 else "",
                "auc_hallucinated_high": auc if auc is not None else "",
                "grounded_q50": float(np.quantile(grounded, 0.50)),
                "grounded_q90": float(np.quantile(grounded, 0.90)),
                "grounded_q95": float(np.quantile(grounded, 0.95)),
                "all_q50": float(np.quantile(vals, 0.50)),
                "all_q90": float(np.quantile(vals, 0.90)),
                "all_q95": float(np.quantile(vals, 0.95)),
                "activation_ref": activation_ref,
                "activation_quantile": float(activation_quantile),
                "activation_threshold": threshold,
                "activation_rule": f"contribution_gt_{activation_ref}_q{activation_quantile:g}",
            })
    return rows


def select_heads(rows, direction, top_frac, min_heads, max_heads):
    ranked = sorted(rows, key=lambda row: selected_score(row, direction), reverse=True)
    if direction == "hallucinated_high":
        ranked = [row for row in ranked if float(row["h_minus_g"]) > 0]
    elif direction == "grounded_high":
        ranked = [row for row in ranked if float(row["h_minus_g"]) < 0]

    target_n = int(math.ceil(len(rows) * top_frac))
    target_n = max(min_heads, target_n)
    target_n = min(max_heads, target_n, len(ranked))
    selected = ranked[:target_n]
    out = []
    for rank, row in enumerate(selected, start=1):
        item = dict(row)
        item["selection_rank"] = rank
        item["selection_direction"] = direction
        item["selection_score"] = float(selected_score(row, direction))
        out.append(item)
    return out


def all_head_reference(rows):
    return [
        {
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "activation_threshold": float(row["activation_threshold"]),
            "activation_ref": row["activation_ref"],
            "activation_quantile": float(row["activation_quantile"]),
            "h_minus_g": float(row["h_minus_g"]),
            "cohens_d_h_minus_g": (
                float(row["cohens_d_h_minus_g"]) if row["cohens_d_h_minus_g"] != "" else None
            ),
            "auc_hallucinated_high": (
                float(row["auc_hallucinated_high"]) if row["auc_hallucinated_high"] != "" else None
            ),
        }
        for row in rows
    ]


def json_head(row):
    return {
        "layer": int(row["layer"]),
        "head": int(row["head"]),
        "selection_rank": int(row["selection_rank"]),
        "selection_direction": row["selection_direction"],
        "selection_score": float(row["selection_score"]),
        "grounded_mean": float(row["grounded_mean"]),
        "hallucinated_mean": float(row["hallucinated_mean"]),
        "h_minus_g": float(row["h_minus_g"]),
        "cohens_d_h_minus_g": (
            float(row["cohens_d_h_minus_g"]) if row["cohens_d_h_minus_g"] != "" else None
        ),
        "auc_hallucinated_high": (
            float(row["auc_hallucinated_high"]) if row["auc_hallucinated_high"] != "" else None
        ),
        "activation_threshold": float(row["activation_threshold"]),
        "activation_ref": row["activation_ref"],
        "activation_quantile": float(row["activation_quantile"]),
        "activation_rule": row["activation_rule"],
    }


def main():
    args = parse_args()
    if args.effect_top_frac <= 0 or args.effect_top_frac > 1:
        raise ValueError("--effect-top-frac must be in (0, 1].")
    if args.activation_quantile <= 0 or args.activation_quantile >= 1:
        raise ValueError("--activation-quantile must be in (0, 1).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]
    train_records, test_records = split_records(records, args.train_frac, args.seed)

    train_rows = head_stats(
        train_records,
        args.start_layer,
        args.end_layer,
        args.activation_ref,
        args.activation_quantile,
    )
    test_rows = head_stats(
        test_records,
        args.start_layer,
        args.end_layer,
        args.activation_ref,
        args.activation_quantile,
    )
    all_rows = head_stats(
        records,
        args.start_layer,
        args.end_layer,
        args.activation_ref,
        args.activation_quantile,
    )

    wrong_heads = select_heads(
        train_rows,
        args.selection_direction,
        args.effect_top_frac,
        args.min_heads,
        args.max_heads,
    )

    payload_out = {
        "config": vars(args),
        "source": args.contrib_file,
        "precommitment": {
            "d1": "H_wrong is selected on train split only, by head-wise hallucinated-vs-grounded contribution shift.",
            "d2_success_criteria": [
                "D-crit-1: Selective reduces CHAIR_i by at least 15% relative to baseline.",
                "D-crit-2: Selective CHAIR_i improvement exceeds random-head improvement mean + 2 std.",
                "D-crit-3: Selective harmful regression count is not larger than random-head mean.",
            ],
            "d3": "Run threshold/head-count/backbone robustness only if all D.2 criteria pass.",
        },
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_test_records": len(test_records),
        "num_wrong_heads": len(wrong_heads),
        "wrong_heads": [json_head(row) for row in wrong_heads],
        "all_head_reference": all_head_reference(train_rows),
    }

    write_csv(train_rows, output_dir / "stage1_d1_train_head_reference.csv")
    write_csv(test_rows, output_dir / "stage1_d1_test_head_reference.csv")
    write_csv(all_rows, output_dir / "stage1_d1_all_head_reference.csv")
    write_csv(wrong_heads, output_dir / "stage1_d1_wrong_heads.csv")

    with open(output_dir / "stage1_d1_wrong_heads.json", "w") as f:
        json.dump(payload_out, f, indent=2)

    print(json.dumps({
        "num_records": len(records),
        "num_train_records": len(train_records),
        "num_test_records": len(test_records),
        "num_wrong_heads": len(wrong_heads),
        "wrong_heads_top5": payload_out["wrong_heads"][:5],
        "outputs": {
            "wrong_heads_json": str(output_dir / "stage1_d1_wrong_heads.json"),
            "wrong_heads_csv": str(output_dir / "stage1_d1_wrong_heads.csv"),
            "train_reference_csv": str(output_dir / "stage1_d1_train_head_reference.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
