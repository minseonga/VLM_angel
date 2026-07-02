#!/usr/bin/env python
"""
Analyze encoder/projector evidence after merging it onto object mentions.

Focus:
  - de-duplicate repeated max-length rows into image-object pairs
  - quantify evidence AUC with object identity controls
  - estimate incremental value above order/confidence signals
  - sample encoder-high/projector-low hallucination cases
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze encoder/projector evidence CSV.")
    parser.add_argument("--evidence-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="encoder_projector_evidence_analysis")
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def auc_summary(df, metrics, y_col="hall"):
    rows = []
    y = df[y_col].astype(int)
    for metric in metrics:
        if metric not in df.columns:
            continue
        x = pd.to_numeric(df[metric], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() == 0 or y[mask].nunique() < 2:
            continue
        auc = float(roc_auc_score(y[mask], x[mask]))
        rows.append({
            "metric": metric,
            "n": int(mask.sum()),
            "auc_hallucinated_high": auc,
            "auc_direction_adjusted": float(max(auc, 1.0 - auc)),
            "direction": "high_hall" if auc >= 0.5 else "low_hall",
            "mean_grounded": float(x[mask & (y == 0)].mean()),
            "mean_hallucinated": float(x[mask & (y == 1)].mean()),
        })
    return pd.DataFrame(rows).sort_values("auc_direction_adjusted", ascending=False)


def cv_auc(df, name, numeric_features, categorical_features=None, group_col=None, n_splits=5):
    categorical_features = categorical_features or []
    cols = numeric_features + categorical_features
    sub = df.dropna(subset=numeric_features + ["hall"]).copy()
    if len(sub) == 0 or sub["hall"].nunique() < 2:
        return None

    y = sub["hall"].astype(int).values
    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical_features))
    pre = ColumnTransformer(transformers)
    clf = make_pipeline(pre, LogisticRegression(max_iter=2000, class_weight="balanced"))

    try:
        if group_col is None:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            scores = cross_val_score(clf, sub[cols], y, cv=cv, scoring="roc_auc")
        else:
            groups = sub[group_col].values
            cv = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(clf, sub[cols], y, cv=cv, groups=groups, scoring="roc_auc")
    except Exception as exc:
        return {
            "model": name,
            "n": int(len(sub)),
            "error": str(exc),
        }

    return {
        "model": name,
        "n": int(len(sub)),
        "auc_mean": float(scores.mean()),
        "auc_std": float(scores.std()),
        "scores": [float(x) for x in scores],
        "group_col": group_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def pair_aggregate(df):
    # Collapse repeated max-length rows for the same image/object pair.
    agg_spec = {
        "hall": "max",
        "label": "min",
        "image_index": "first",
        "chair_word_idx": "max",
        "mention_idx": "max",
        "token_step": "max",
        "step_frac": "max",
        "actual_prob": "mean",
        "actual_surprisal": "mean",
        "vocab_entropy": "mean",
        "top1_top2_margin": "mean",
        "svar_band": "mean",
        "visual_attn_mean_band": "mean",
        "visual_attn_max_head_band": "mean",
        "encoder_clip_max": "first",
        "encoder_clip_top5_mean": "first",
        "encoder_clip_mean": "first",
        "encoder_clip_global": "first",
        "projector_llm_max": "first",
        "projector_llm_top5_mean": "first",
        "projector_llm_mean": "first",
        "encoder_patch_norm_mean": "first",
        "projector_patch_norm_mean": "first",
    }
    existing = {k: v for k, v in agg_spec.items() if k in df.columns}
    pair = df.groupby(["image_id", "node_word"], as_index=False).agg(existing)
    pair["label"] = 1 - pair["hall"]
    return pair


def quantile_case_samples(pair, raw, output_dir, prefix):
    # Find cases that look like encoder evidence exists but projector evidence is low.
    out = {}
    for col in ["encoder_clip_global", "encoder_clip_max", "projector_llm_top5_mean", "projector_llm_max"]:
        if col in pair.columns:
            pair[f"{col}_pct"] = pair[col].rank(pct=True)
    if {"encoder_clip_global_pct", "projector_llm_top5_mean_pct"}.issubset(pair.columns):
        pair["encoder_high_projector_low"] = pair["encoder_clip_global_pct"] - pair["projector_llm_top5_mean_pct"]
        sample = pair.sort_values(["hall", "encoder_high_projector_low"], ascending=[False, False]).head(30)
        cols = [
            "image_id", "node_word", "hall", "chair_word_idx", "mention_idx", "token_step", "step_frac",
            "encoder_clip_global", "encoder_clip_global_pct", "projector_llm_top5_mean", "projector_llm_top5_mean_pct",
            "encoder_high_projector_low", "actual_surprisal", "vocab_entropy",
        ]
        cols = [c for c in cols if c in sample.columns]
        sample[cols].to_csv(output_dir / f"{prefix}_encoder_high_projector_low_cases.csv", index=False)
        out["encoder_high_projector_low_cases"] = str(output_dir / f"{prefix}_encoder_high_projector_low_cases.csv")

    if {"encoder_clip_global_pct", "projector_llm_top5_mean_pct"}.issubset(pair.columns):
        bins = {
            "encoder_high_projector_low": pair[(pair.encoder_clip_global_pct >= 0.66) & (pair.projector_llm_top5_mean_pct <= 0.33)],
            "encoder_low_projector_low": pair[(pair.encoder_clip_global_pct <= 0.33) & (pair.projector_llm_top5_mean_pct <= 0.33)],
            "encoder_high_projector_high": pair[(pair.encoder_clip_global_pct >= 0.66) & (pair.projector_llm_top5_mean_pct >= 0.66)],
        }
        out["quadrant_summary"] = {
            name: {
                "n": int(len(sub)),
                "hall_rate": float(sub.hall.mean()) if len(sub) else None,
                "mean_chair_word_idx": float(sub.chair_word_idx.mean()) if len(sub) else None,
            }
            for name, sub in bins.items()
        }
    return out


def main():
    args = parse_args()
    evidence_csv = Path(args.evidence_csv)
    output_dir = Path(args.output_dir) if args.output_dir else evidence_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(evidence_csv)
    df["hall"] = (df["label"].astype(int) == 0).astype(int)
    pair = pair_aggregate(df)

    metrics = [
        "encoder_clip_max", "encoder_clip_top5_mean", "encoder_clip_mean", "encoder_clip_global",
        "projector_llm_max", "projector_llm_top5_mean", "projector_llm_mean",
        "encoder_patch_norm_mean", "projector_patch_norm_mean",
        "actual_prob", "actual_surprisal", "vocab_entropy", "top1_top2_margin",
        "svar_band", "visual_attn_mean_band", "visual_attn_max_head_band",
        "token_step", "step_frac", "chair_word_idx", "mention_idx",
    ]

    row_auc = auc_summary(df, metrics)
    pair_auc = auc_summary(pair, metrics)
    row_auc.to_csv(output_dir / f"{args.prefix}_row_auc.csv", index=False)
    pair_auc.to_csv(output_dir / f"{args.prefix}_pair_auc.csv", index=False)
    pair.to_csv(output_dir / f"{args.prefix}_image_object_pairs.csv", index=False)

    order = ["chair_word_idx", "mention_idx", "step_frac"]
    confidence = ["actual_surprisal", "vocab_entropy", "actual_prob"]
    attention = ["svar_band", "visual_attn_mean_band", "visual_attn_max_head_band"]
    encoder = ["encoder_clip_max", "encoder_clip_top5_mean", "encoder_clip_global"]
    projector = ["projector_llm_max", "projector_llm_top5_mean", "projector_llm_mean"]

    model_specs = [
        ("order", order, []),
        ("confidence", confidence, []),
        ("attention", attention, []),
        ("encoder", encoder, []),
        ("projector", projector, []),
        ("object_identity", [], ["node_word"]),
        ("order_confidence", order + confidence, []),
        ("order_encoder", order + encoder, []),
        ("order_projector", order + projector, []),
        ("order_conf_encoder_projector", order + confidence + encoder + projector, []),
        ("order_object_identity", order, ["node_word"]),
        ("order_conf_object_identity", order + confidence, ["node_word"]),
        ("order_conf_encoder_object_identity", order + confidence + encoder, ["node_word"]),
        ("order_conf_projector_object_identity", order + confidence + projector, ["node_word"]),
        ("order_conf_enc_proj_object_identity", order + confidence + encoder + projector, ["node_word"]),
    ]

    cv_rows = []
    cv_pair = []
    for name, nums, cats in model_specs:
        res = cv_auc(df, name, nums, cats, group_col=None, n_splits=args.n_splits)
        if res:
            res["level"] = "row_stratified"
            cv_rows.append(res)
        res = cv_auc(df, name, nums, cats, group_col="image_id", n_splits=args.n_splits)
        if res:
            res["level"] = "row_group_image"
            cv_rows.append(res)
        res = cv_auc(pair, name, nums, cats, group_col=None, n_splits=args.n_splits)
        if res:
            res["level"] = "pair_stratified"
            cv_pair.append(res)
        res = cv_auc(pair, name, nums, cats, group_col="image_id", n_splits=args.n_splits)
        if res:
            res["level"] = "pair_group_image"
            cv_pair.append(res)

    cv_df = pd.DataFrame(cv_rows + cv_pair)
    cv_df.to_csv(output_dir / f"{args.prefix}_cv_auc.csv", index=False)

    cases = quantile_case_samples(pair, df, output_dir, args.prefix)

    summary = {
        "input": str(evidence_csv),
        "num_rows": int(len(df)),
        "num_pairs": int(len(pair)),
        "num_images": int(df.image_id.nunique()),
        "num_objects": int(df.node_word.nunique()),
        "row_hall_rate": float(df.hall.mean()),
        "pair_hall_rate": float(pair.hall.mean()),
        "top_row_auc": row_auc.head(10).to_dict("records"),
        "top_pair_auc": pair_auc.head(10).to_dict("records"),
        "cv_auc": cv_df.to_dict("records"),
        **cases,
    }
    with open(output_dir / f"{args.prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_rows": summary["num_rows"],
        "num_pairs": summary["num_pairs"],
        "row_hall_rate": summary["row_hall_rate"],
        "pair_hall_rate": summary["pair_hall_rate"],
        "top_pair_auc": summary["top_pair_auc"][:5],
        "quadrant_summary": summary.get("quadrant_summary"),
        "outputs": {
            "summary": str(output_dir / f"{args.prefix}_summary.json"),
            "pair_auc": str(output_dir / f"{args.prefix}_pair_auc.csv"),
            "cv_auc": str(output_dir / f"{args.prefix}_cv_auc.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
