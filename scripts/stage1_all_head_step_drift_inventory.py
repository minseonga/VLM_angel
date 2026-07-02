#!/usr/bin/env python
"""
Inventory decoding-step drift for every head.

This is a forward-free diagnostic over an existing Stage 1 head-logit
contribution trace. It asks:
  - which heads drift most with decoding step?
  - is the drift generic, hallucination-specific, or grounded-risky?
  - how much do D.1 selected heads overlap with each group?

Attention mass/entropy are intentionally not computed here because the current
head_logit_contrib trace stores direct logit contributions only.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import write_csv
from stage1_common import torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="All-head decoding-step drift inventory.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs_n500/stage1_head_logit_contrib.pt")
    parser.add_argument("--heads-json", type=str, default="stage1_outputs_n500_d1_wrong_heads/stage1_d1_wrong_heads.json")
    parser.add_argument("--head-key", type=str, default="wrong_heads")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_n500_all_head_step_drift")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=None)
    parser.add_argument("--target-name", type=str, default="actual")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--early-frac", type=float, default=0.25)
    parser.add_argument("--late-frac", type=float, default=0.25)
    parser.add_argument(
        "--label-conditional-frac",
        type=float,
        default=0.25,
        help="Within-label early/late fraction for H-vs-G drift specificity.",
    )
    parser.add_argument("--token-step-bins", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-group-count", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_d1_heads(path, key):
    if not path or not os.path.exists(path):
        return {}, set()
    with open(path) as f:
        payload = json.load(f)
    by_head = {}
    selected = set()
    for item in payload.get(key, []):
        layer = int(item["layer"])
        head = int(item["head"])
        selected.add((layer, head))
        by_head[(layer, head)] = {
            "d1_selection_rank": int(item.get("selection_rank", len(selected))),
            "d1_selection_score": float(item.get("selection_score", item.get("h_minus_g", 0.0))),
            "d1_h_minus_g": float(item.get("h_minus_g", 0.0)),
            "d1_auc_hallucinated_high": (
                float(item["auc_hallucinated_high"])
                if item.get("auc_hallucinated_high") is not None
                else None
            ),
            "d1_activation_threshold": float(item.get("activation_threshold", 0.0)),
        }
    return by_head, selected


def target_index(record, target_name, fallback_index):
    names = list(record.get("target_names", []))
    if target_name:
        for idx, name in enumerate(names):
            if name == target_name:
                return idx
    return int(fallback_index)


def contribution_cube(records, target_name, fallback_index):
    matrices = []
    for record in records:
        contrib = record["head_logit_contrib"].detach().cpu()
        idx = target_index(record, target_name, fallback_index)
        if idx < 0 or idx >= contrib.shape[2]:
            raise IndexError(
                f"target index {idx} out of range for record {record.get('object_index')} "
                f"with target_count={contrib.shape[2]}"
            )
        matrices.append(contrib[:, :, idx].numpy())
    return np.stack(matrices, axis=0)


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    if pooled == 0:
        return None
    return float((b.mean() - a.mean()) / pooled)


def mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    return float(values.mean())


def gap_stats(values, early_mask, late_mask, min_count):
    early = values[early_mask]
    late = values[late_mask]
    out = {
        "early_n": int(len(early)),
        "late_n": int(len(late)),
        "early_mean": mean_or_none(early),
        "late_mean": mean_or_none(late),
        "late_minus_early": None,
        "cohen_d_late_vs_early": None,
    }
    if len(early) >= min_count and len(late) >= min_count:
        out["late_minus_early"] = float(late.mean() - early.mean())
        out["cohen_d_late_vs_early"] = cohen_d(early, late)
    return out


def metric_values(raw_values, metric):
    if metric == "contrib":
        return raw_values
    if metric == "positive_contrib":
        return np.maximum(raw_values, 0.0)
    if metric == "negative_contrib":
        return np.minimum(raw_values, 0.0)
    if metric == "abs_contrib":
        return np.abs(raw_values)
    raise KeyError(metric)


def make_early_late_masks(steps, early_frac, late_frac):
    early_cut = float(np.quantile(steps, early_frac))
    late_cut = float(np.quantile(steps, 1.0 - late_frac))
    early_mask = steps <= early_cut
    late_mask = steps >= late_cut
    return early_cut, late_cut, early_mask, late_mask


def quantile_masks_for_subset(steps, subset_mask, frac):
    subset_steps = steps[subset_mask]
    if len(subset_steps) == 0:
        empty = np.zeros_like(subset_mask, dtype=bool)
        return None, None, empty, empty
    early_cut = float(np.quantile(subset_steps, frac))
    late_cut = float(np.quantile(subset_steps, 1.0 - frac))
    early_mask = subset_mask & (steps <= early_cut)
    late_mask = subset_mask & (steps >= late_cut)
    return early_cut, late_cut, early_mask, late_mask


def make_label_conditional_masks(steps, labels, frac):
    grounded_mask = labels == 1
    hallucinated_mask = labels == 0
    g_early_cut, g_late_cut, g_early_mask, g_late_mask = quantile_masks_for_subset(
        steps,
        grounded_mask,
        frac,
    )
    h_early_cut, h_late_cut, h_early_mask, h_late_mask = quantile_masks_for_subset(
        steps,
        hallucinated_mask,
        frac,
    )
    return {
        "grounded_early_cut": g_early_cut,
        "grounded_late_cut": g_late_cut,
        "grounded_early_mask": g_early_mask,
        "grounded_late_mask": g_late_mask,
        "hallucinated_early_cut": h_early_cut,
        "hallucinated_late_cut": h_late_cut,
        "hallucinated_early_mask": h_early_mask,
        "hallucinated_late_mask": h_late_mask,
    }


def score_or_neg_inf(row, key):
    value = row.get(key)
    if value is None or value == "":
        return -float("inf")
    return float(value)


def abs_score(row, key):
    value = row.get(key)
    if value is None or value == "":
        return -float("inf")
    return abs(float(value))


def finite_score_items(rows, key):
    return [row for row in rows if row.get(key) not in (None, "")]


def rank_rows(rows, score_name, key_fn, top_k):
    ranked = sorted(rows, key=key_fn, reverse=True)
    out = []
    for rank, row in enumerate(ranked[:top_k], start=1):
        item = compact_head_row(row)
        item["rank"] = rank
        item["score_name"] = score_name
        out.append(item)
    return out


def compact_head_row(row):
    keys = [
        "layer",
        "head",
        "metric",
        "is_d1_selected",
        "all_late_minus_early",
        "all_cohen_d_late_vs_early",
        "pearson_step_corr",
        "spearman_step_corr",
        "grounded_late_minus_early",
        "hallucinated_late_minus_early",
        "hall_specific_drift",
        "generic_positive_drift",
        "generic_abs_same_direction_drift",
        "grounded_risk",
        "lc_grounded_late_minus_early",
        "lc_hallucinated_late_minus_early",
        "lc_hall_specific_drift",
        "lc_generic_positive_drift",
        "lc_generic_abs_same_direction_drift",
        "lc_grounded_risk",
        "d1_selection_rank",
        "d1_selection_score",
    ]
    return {key: row.get(key) for key in keys if key in row}


def head_key(row):
    return (int(row["layer"]), int(row["head"]))


def overlap_summary(name, top_rows, d1_selected):
    top_set = {head_key(row) for row in top_rows}
    overlap = sorted(top_set & d1_selected)
    return {
        "list_name": name,
        "top_n": int(len(top_set)),
        "d1_n": int(len(d1_selected)),
        "overlap_n": int(len(overlap)),
        "overlap_fraction_of_top": float(len(overlap) / len(top_set)) if top_set else None,
        "overlap_fraction_of_d1": float(len(overlap) / len(d1_selected)) if d1_selected else None,
        "overlap_heads": [{"layer": layer, "head": head} for layer, head in overlap],
    }


def rank_distribution(rows, score_key, d1_selected, descending=True):
    valid = finite_score_items(rows, score_key)
    valid = sorted(valid, key=lambda row: float(row[score_key]), reverse=descending)
    ranks = {}
    for idx, row in enumerate(valid, start=1):
        ranks[head_key(row)] = idx
    d1_ranks = [ranks[key] for key in d1_selected if key in ranks]
    if not d1_ranks:
        return {
            "score": score_key,
            "num_ranked": len(valid),
            "num_d1_ranked": 0,
        }
    arr = np.array(d1_ranks, dtype=np.float64)
    return {
        "score": score_key,
        "num_ranked": len(valid),
        "num_d1_ranked": int(len(d1_ranks)),
        "d1_mean_rank": float(arr.mean()),
        "d1_median_rank": float(np.median(arr)),
        "d1_best_rank": int(arr.min()),
        "d1_top10pct_count": int((arr <= max(1, math.ceil(len(valid) * 0.10))).sum()),
        "d1_top30_count": int((arr <= 30).sum()),
    }


def rank_distribution_custom(rows, score_name, d1_selected, score_fn):
    scored = []
    for row in rows:
        score = score_fn(row)
        if score is None or not np.isfinite(float(score)):
            continue
        scored.append((float(score), row))
    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    ranks = {}
    for idx, (_, row) in enumerate(scored, start=1):
        ranks[head_key(row)] = idx
    d1_ranks = [ranks[key] for key in d1_selected if key in ranks]
    if not d1_ranks:
        return {
            "score": score_name,
            "num_ranked": len(scored),
            "num_d1_ranked": 0,
        }
    arr = np.array(d1_ranks, dtype=np.float64)
    return {
        "score": score_name,
        "num_ranked": len(scored),
        "num_d1_ranked": int(len(d1_ranks)),
        "d1_mean_rank": float(arr.mean()),
        "d1_median_rank": float(np.median(arr)),
        "d1_best_rank": int(arr.min()),
        "d1_top10pct_count": int((arr <= max(1, math.ceil(len(scored) * 0.10))).sum()),
        "d1_top30_count": int((arr <= 30).sum()),
    }


def token_step_bins(steps, num_bins):
    edges = np.quantile(steps, np.linspace(0, 1, num_bins + 1))
    bins = []
    for idx in range(num_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == num_bins - 1:
            mask = (steps >= lo) & (steps <= hi)
        else:
            mask = (steps >= lo) & (steps < hi)
        if mask.any():
            bins.append({
                "bucket": idx,
                "lo": lo,
                "hi": hi,
                "mid": float((lo + hi) / 2.0),
                "mask": mask,
            })
    return bins


def bin_profile_rows(cube, steps, labels, layer_range, bins, primary_metric):
    rows = []
    for layer in layer_range:
        for head in range(cube.shape[2]):
            values = metric_values(cube[:, layer, head], primary_metric)
            for bucket in bins:
                mask = bucket["mask"]
                base = {
                    "bucket": int(bucket["bucket"]),
                    "token_step_lo": float(bucket["lo"]),
                    "token_step_hi": float(bucket["hi"]),
                    "token_step_mid": float(bucket["mid"]),
                    "layer": int(layer),
                    "head": int(head),
                    "metric": primary_metric,
                    "n": int(mask.sum()),
                    "grounded_n": int(((labels == 1) & mask).sum()),
                    "hallucinated_n": int(((labels == 0) & mask).sum()),
                    "hallucination_rate": float(((labels == 0) & mask).sum() / mask.sum()) if mask.sum() else None,
                }
                for label_group, label_mask in [
                    ("all", mask),
                    ("grounded", mask & (labels == 1)),
                    ("hallucinated", mask & (labels == 0)),
                ]:
                    cur = values[label_mask]
                    rows.append({
                        **base,
                        "label_group": label_group,
                        "group_n": int(len(cur)),
                        "mean": mean_or_none(cur),
                        "std": float(cur.std()) if len(cur) else None,
                    })
    return rows


def sanitize(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def csv_safe_rows(rows):
    return [{key: sanitize(value) for key, value in row.items()} for row in rows]


def svg_scatter(points, x_key, y_key, title, x_label, y_label, path, width=920, height=680):
    usable = [
        point for point in points
        if point.get(x_key) not in (None, "") and point.get(y_key) not in (None, "")
    ]
    if not usable:
        return False

    x_vals = np.array([float(point[x_key]) for point in usable], dtype=np.float64)
    y_vals = np.array([float(point[y_key]) for point in usable], dtype=np.float64)
    x_min, x_max = float(x_vals.min()), float(x_vals.max())
    y_min, y_max = float(y_vals.min()), float(y_vals.max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    margin_left = 84
    margin_right = 42
    margin_top = 54
    margin_bottom = 74
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def sx(x):
        return margin_left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_top + (y_max - float(y)) / (y_max - y_min) * plot_h

    def fmt(value):
        return f"{value:.3g}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">{x_label}</text>',
        f'<text x="20" y="{height / 2}" transform="rotate(-90 20 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
    ]

    for tick in np.linspace(0, 1, 6):
        x = margin_left + tick * plot_w
        x_value = x_min + tick * (x_max - x_min)
        parts.append(f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" stroke="#eceff3" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{fmt(x_value)}</text>')
    for tick in np.linspace(0, 1, 6):
        y = margin_top + tick * plot_h
        y_value = y_max - tick * (y_max - y_min)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" stroke="#eceff3" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{fmt(y_value)}</text>')

    if x_min <= 0 <= x_max:
        x0 = sx(0)
        parts.append(f'<line x1="{x0:.1f}" y1="{margin_top}" x2="{x0:.1f}" y2="{margin_top + plot_h}" stroke="#9ca3af" stroke-width="1.4"/>')
    if y_min <= 0 <= y_max:
        y0 = sy(0)
        parts.append(f'<line x1="{margin_left}" y1="{y0:.1f}" x2="{margin_left + plot_w}" y2="{y0:.1f}" stroke="#9ca3af" stroke-width="1.4"/>')
    if x_min <= y_min and x_max >= y_max:
        pass

    for point in usable:
        selected = bool(point.get("is_d1_selected"))
        color = "#dc2626" if selected else "#6b7280"
        radius = 4.4 if selected else 2.2
        opacity = 0.9 if selected else 0.36
        parts.append(
            f'<circle cx="{sx(point[x_key]):.1f}" cy="{sy(point[y_key]):.1f}" '
            f'r="{radius}" fill="{color}" opacity="{opacity}"/>'
        )

    legend_x = margin_left + 14
    legend_y = margin_top + 18
    parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4.4" fill="#dc2626" opacity="0.9"/>')
    parts.append(f'<text x="{legend_x + 12}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#111827">D1 selected</text>')
    parts.append(f'<circle cx="{legend_x}" cy="{legend_y + 22}" r="2.6" fill="#6b7280" opacity="0.5"/>')
    parts.append(f'<text x="{legend_x + 12}" y="{legend_y + 26}" font-family="Arial" font-size="12" fill="#111827">other heads</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return True


def write_plots(primary_rows, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    p1 = plot_dir / "head_drift_grounded_vs_hallucinated.svg"
    if svg_scatter(
        primary_rows,
        "grounded_late_minus_early",
        "hallucinated_late_minus_early",
        "Head drift: grounded vs hallucinated",
        "D_G = grounded late - early contribution",
        "D_H = hallucinated late - early contribution",
        p1,
    ):
        paths.append(str(p1))

    p2 = plot_dir / "head_drift_vs_hall_specificity.svg"
    if svg_scatter(
        primary_rows,
        "all_late_minus_early",
        "hall_specific_drift",
        "Step drift vs hallucination specificity",
        "D_all = all late - early contribution",
        "D_H - D_G",
        p2,
    ):
        paths.append(str(p2))

    p3 = plot_dir / "head_label_cond_grounded_vs_hallucinated.svg"
    if svg_scatter(
        primary_rows,
        "lc_grounded_late_minus_early",
        "lc_hallucinated_late_minus_early",
        "Label-conditional drift: grounded vs hallucinated",
        "D_G within grounded = late - early",
        "D_H within hallucinated = late - early",
        p3,
    ):
        paths.append(str(p3))

    p4 = plot_dir / "head_step_drift_vs_label_cond_specificity.svg"
    if svg_scatter(
        primary_rows,
        "all_late_minus_early",
        "lc_hall_specific_drift",
        "Step drift vs label-conditional hallucination specificity",
        "D_all = global late - early contribution",
        "D_H(label-conditional) - D_G(label-conditional)",
        p4,
    ):
        paths.append(str(p4))
    return paths


def main():
    args = parse_args()
    if not (0 < args.early_frac < 1) or not (0 < args.late_frac < 1):
        raise ValueError("--early-frac and --late-frac must be in (0, 1).")
    if args.early_frac + args.late_frac >= 1:
        raise ValueError("--early-frac + --late-frac must be < 1.")
    if not (0 < args.label_conditional_frac < 0.5):
        raise ValueError("--label-conditional-frac must be in (0, 0.5).")
    if args.token_step_bins < 2:
        raise ValueError("--token-step-bins must be >= 2.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]
    if not records:
        raise ValueError("No records found in contribution trace.")

    d1_meta, d1_selected = load_d1_heads(args.heads_json, args.head_key)
    cube = contribution_cube(records, args.target_name, args.target_index)
    num_layers = cube.shape[1]
    num_heads = cube.shape[2]
    start_layer = int(args.start_layer)
    end_layer = num_layers - 1 if args.end_layer is None else int(args.end_layer)
    if start_layer < 0 or end_layer >= num_layers or start_layer > end_layer:
        raise ValueError(f"Invalid layer range [{start_layer}, {end_layer}] for {num_layers} layers.")
    layer_range = list(range(start_layer, end_layer + 1))

    steps = np.array([float(record["token_step"]) for record in records], dtype=np.float64)
    labels = np.array([int(record["label"]) for record in records], dtype=np.int64)
    early_cut, late_cut, early_mask, late_mask = make_early_late_masks(
        steps,
        args.early_frac,
        args.late_frac,
    )
    label_cond = make_label_conditional_masks(steps, labels, args.label_conditional_frac)

    metrics = ["contrib", "positive_contrib", "negative_contrib", "abs_contrib"]
    rows = []
    for layer in layer_range:
        for head in range(num_heads):
            raw = cube[:, layer, head].astype(np.float64)
            for metric in metrics:
                values = metric_values(raw, metric)
                all_gap = gap_stats(values, early_mask, late_mask, args.min_group_count)
                g_gap = gap_stats(values, early_mask & (labels == 1), late_mask & (labels == 1), args.min_group_count)
                h_gap = gap_stats(values, early_mask & (labels == 0), late_mask & (labels == 0), args.min_group_count)
                lc_g_gap = gap_stats(
                    values,
                    label_cond["grounded_early_mask"],
                    label_cond["grounded_late_mask"],
                    args.min_group_count,
                )
                lc_h_gap = gap_stats(
                    values,
                    label_cond["hallucinated_early_mask"],
                    label_cond["hallucinated_late_mask"],
                    args.min_group_count,
                )

                all_lme = all_gap["late_minus_early"]
                grounded_lme = g_gap["late_minus_early"]
                hallucinated_lme = h_gap["late_minus_early"]
                hall_specific = (
                    float(hallucinated_lme - grounded_lme)
                    if hallucinated_lme is not None and grounded_lme is not None
                    else None
                )
                lc_grounded_lme = lc_g_gap["late_minus_early"]
                lc_hallucinated_lme = lc_h_gap["late_minus_early"]
                lc_hall_specific = (
                    float(lc_hallucinated_lme - lc_grounded_lme)
                    if lc_hallucinated_lme is not None and lc_grounded_lme is not None
                    else None
                )

                generic_positive = None
                generic_abs_same = None
                if hallucinated_lme is not None and grounded_lme is not None:
                    if hallucinated_lme > 0 and grounded_lme > 0:
                        generic_positive = float(min(hallucinated_lme, grounded_lme))
                    else:
                        generic_positive = 0.0
                    if hallucinated_lme == 0 or grounded_lme == 0:
                        generic_abs_same = 0.0
                    elif math.copysign(1.0, hallucinated_lme) == math.copysign(1.0, grounded_lme):
                        generic_abs_same = float(min(abs(hallucinated_lme), abs(grounded_lme)))
                    else:
                        generic_abs_same = 0.0
                lc_generic_positive = None
                lc_generic_abs_same = None
                if lc_hallucinated_lme is not None and lc_grounded_lme is not None:
                    if lc_hallucinated_lme > 0 and lc_grounded_lme > 0:
                        lc_generic_positive = float(min(lc_hallucinated_lme, lc_grounded_lme))
                    else:
                        lc_generic_positive = 0.0
                    if lc_hallucinated_lme == 0 or lc_grounded_lme == 0:
                        lc_generic_abs_same = 0.0
                    elif math.copysign(1.0, lc_hallucinated_lme) == math.copysign(1.0, lc_grounded_lme):
                        lc_generic_abs_same = float(min(abs(lc_hallucinated_lme), abs(lc_grounded_lme)))
                    else:
                        lc_generic_abs_same = 0.0

                key = (layer, head)
                row = {
                    "layer": int(layer),
                    "head": int(head),
                    "metric": metric,
                    "target_name": args.target_name,
                    "is_d1_selected": int(key in d1_selected),
                    "pearson_step_corr": pearson(steps, values),
                    "spearman_step_corr": spearman(steps, values),
                    "all_early_n": all_gap["early_n"],
                    "all_late_n": all_gap["late_n"],
                    "all_early_mean": all_gap["early_mean"],
                    "all_late_mean": all_gap["late_mean"],
                    "all_late_minus_early": all_lme,
                    "all_cohen_d_late_vs_early": all_gap["cohen_d_late_vs_early"],
                    "grounded_early_n": g_gap["early_n"],
                    "grounded_late_n": g_gap["late_n"],
                    "grounded_early_mean": g_gap["early_mean"],
                    "grounded_late_mean": g_gap["late_mean"],
                    "grounded_late_minus_early": grounded_lme,
                    "grounded_cohen_d_late_vs_early": g_gap["cohen_d_late_vs_early"],
                    "hallucinated_early_n": h_gap["early_n"],
                    "hallucinated_late_n": h_gap["late_n"],
                    "hallucinated_early_mean": h_gap["early_mean"],
                    "hallucinated_late_mean": h_gap["late_mean"],
                    "hallucinated_late_minus_early": hallucinated_lme,
                    "hallucinated_cohen_d_late_vs_early": h_gap["cohen_d_late_vs_early"],
                    "hall_specific_drift": hall_specific,
                    "generic_positive_drift": generic_positive,
                    "generic_abs_same_direction_drift": generic_abs_same,
                    "grounded_risk": grounded_lme,
                    "lc_grounded_early_n": lc_g_gap["early_n"],
                    "lc_grounded_late_n": lc_g_gap["late_n"],
                    "lc_grounded_early_mean": lc_g_gap["early_mean"],
                    "lc_grounded_late_mean": lc_g_gap["late_mean"],
                    "lc_grounded_late_minus_early": lc_grounded_lme,
                    "lc_grounded_cohen_d_late_vs_early": lc_g_gap["cohen_d_late_vs_early"],
                    "lc_hallucinated_early_n": lc_h_gap["early_n"],
                    "lc_hallucinated_late_n": lc_h_gap["late_n"],
                    "lc_hallucinated_early_mean": lc_h_gap["early_mean"],
                    "lc_hallucinated_late_mean": lc_h_gap["late_mean"],
                    "lc_hallucinated_late_minus_early": lc_hallucinated_lme,
                    "lc_hallucinated_cohen_d_late_vs_early": lc_h_gap["cohen_d_late_vs_early"],
                    "lc_hall_specific_drift": lc_hall_specific,
                    "lc_generic_positive_drift": lc_generic_positive,
                    "lc_generic_abs_same_direction_drift": lc_generic_abs_same,
                    "lc_grounded_risk": lc_grounded_lme,
                }
                row.update(d1_meta.get(key, {}))
                rows.append(row)

    primary_rows = [row for row in rows if row["metric"] == "contrib"]
    top_lists = {
        "top_abs_contrib_cohen_d": rank_rows(
            primary_rows,
            "abs_contrib_cohen_d",
            lambda row: abs_score(row, "all_cohen_d_late_vs_early"),
            args.top_k,
        ),
        "top_abs_step_corr": rank_rows(
            primary_rows,
            "abs_step_corr",
            lambda row: abs_score(row, "pearson_step_corr"),
            args.top_k,
        ),
        "top_positive_contrib_late_minus_early": rank_rows(
            primary_rows,
            "positive_contrib_late_minus_early",
            lambda row: score_or_neg_inf(row, "all_late_minus_early"),
            args.top_k,
        ),
        "top_hall_specific_drift": rank_rows(
            primary_rows,
            "hall_specific_drift",
            lambda row: score_or_neg_inf(row, "hall_specific_drift"),
            args.top_k,
        ),
        "top_generic_positive_drift": rank_rows(
            primary_rows,
            "generic_positive_drift",
            lambda row: score_or_neg_inf(row, "generic_positive_drift"),
            args.top_k,
        ),
        "top_grounded_risk": rank_rows(
            primary_rows,
            "grounded_risk",
            lambda row: score_or_neg_inf(row, "grounded_risk"),
            args.top_k,
        ),
        "top_lc_hall_specific_drift": rank_rows(
            primary_rows,
            "lc_hall_specific_drift",
            lambda row: score_or_neg_inf(row, "lc_hall_specific_drift"),
            args.top_k,
        ),
        "top_lc_generic_positive_drift": rank_rows(
            primary_rows,
            "lc_generic_positive_drift",
            lambda row: score_or_neg_inf(row, "lc_generic_positive_drift"),
            args.top_k,
        ),
        "top_lc_grounded_risk": rank_rows(
            primary_rows,
            "lc_grounded_risk",
            lambda row: score_or_neg_inf(row, "lc_grounded_risk"),
            args.top_k,
        ),
    }

    overlap_rows = [
        overlap_summary(name, top_rows, d1_selected)
        for name, top_rows in top_lists.items()
    ]
    rank_summaries = [
        rank_distribution_custom(
            primary_rows,
            "abs_contrib_cohen_d",
            d1_selected,
            lambda row: abs(float(row["all_cohen_d_late_vs_early"]))
            if row.get("all_cohen_d_late_vs_early") not in (None, "")
            else None,
        ),
        rank_distribution_custom(
            primary_rows,
            "abs_step_corr",
            d1_selected,
            lambda row: abs(float(row["pearson_step_corr"]))
            if row.get("pearson_step_corr") not in (None, "")
            else None,
        ),
        rank_distribution(primary_rows, "all_late_minus_early", d1_selected, descending=True),
        rank_distribution(primary_rows, "hall_specific_drift", d1_selected, descending=True),
        rank_distribution(primary_rows, "generic_positive_drift", d1_selected, descending=True),
        rank_distribution(primary_rows, "grounded_risk", d1_selected, descending=True),
        rank_distribution(primary_rows, "lc_hall_specific_drift", d1_selected, descending=True),
        rank_distribution(primary_rows, "lc_generic_positive_drift", d1_selected, descending=True),
        rank_distribution(primary_rows, "lc_grounded_risk", d1_selected, descending=True),
    ]

    bins = token_step_bins(steps, args.token_step_bins)
    profile_rows = bin_profile_rows(cube, steps, labels, layer_range, bins, "contrib")
    plot_paths = [] if args.no_plots else write_plots(primary_rows, output_dir)

    summary = {
        "contrib_file": args.contrib_file,
        "heads_json": args.heads_json if d1_selected else None,
        "head_key": args.head_key if d1_selected else None,
        "num_records": int(len(records)),
        "num_grounded": int((labels == 1).sum()),
        "num_hallucinated": int((labels == 0).sum()),
        "num_layers_in_trace": int(num_layers),
        "num_heads_per_layer": int(num_heads),
        "layer_range": [start_layer, end_layer],
        "num_heads_analyzed": int(len(layer_range) * num_heads),
        "num_d1_heads": int(len(d1_selected)),
        "target_name": args.target_name,
        "target_index_fallback": args.target_index,
        "early_frac": args.early_frac,
        "late_frac": args.late_frac,
        "label_conditional_frac": args.label_conditional_frac,
        "early_token_step_cut": early_cut,
        "late_token_step_cut": late_cut,
        "early_n": int(early_mask.sum()),
        "late_n": int(late_mask.sum()),
        "early_grounded_n": int((early_mask & (labels == 1)).sum()),
        "early_hallucinated_n": int((early_mask & (labels == 0)).sum()),
        "late_grounded_n": int((late_mask & (labels == 1)).sum()),
        "late_hallucinated_n": int((late_mask & (labels == 0)).sum()),
        "label_conditional_windows": {
            "grounded": {
                "early_token_step_cut": label_cond["grounded_early_cut"],
                "late_token_step_cut": label_cond["grounded_late_cut"],
                "early_n": int(label_cond["grounded_early_mask"].sum()),
                "late_n": int(label_cond["grounded_late_mask"].sum()),
            },
            "hallucinated": {
                "early_token_step_cut": label_cond["hallucinated_early_cut"],
                "late_token_step_cut": label_cond["hallucinated_late_cut"],
                "early_n": int(label_cond["hallucinated_early_mask"].sum()),
                "late_n": int(label_cond["hallucinated_late_mask"].sum()),
            },
        },
        "top_lists": top_lists,
        "d1_overlap_summary": overlap_rows,
        "d1_rank_summaries": rank_summaries,
        "plots": plot_paths,
        "outputs": {
            "primary_contrib_inventory": str(output_dir / "stage1_all_head_step_drift_primary_contrib.csv"),
            "all_metric_inventory": str(output_dir / "stage1_all_head_step_drift_all_metrics.csv"),
            "bin_profile": str(output_dir / "stage1_all_head_step_drift_bin_profile.csv"),
            "top_heads": str(output_dir / "stage1_all_head_step_drift_top_heads.json"),
            "overlap": str(output_dir / "stage1_all_head_step_drift_d1_overlap.csv"),
            "summary": str(output_dir / "stage1_all_head_step_drift_summary.json"),
        },
    }

    write_csv(csv_safe_rows(primary_rows), output_dir / "stage1_all_head_step_drift_primary_contrib.csv")
    write_csv(csv_safe_rows(rows), output_dir / "stage1_all_head_step_drift_all_metrics.csv")
    write_csv(csv_safe_rows(profile_rows), output_dir / "stage1_all_head_step_drift_bin_profile.csv")
    write_csv(csv_safe_rows(overlap_rows), output_dir / "stage1_all_head_step_drift_d1_overlap.csv")
    write_csv(csv_safe_rows(rank_summaries), output_dir / "stage1_all_head_step_drift_d1_rank_summaries.csv")
    with open(output_dir / "stage1_all_head_step_drift_top_heads.json", "w") as f:
        json.dump(top_lists, f, indent=2)
    with open(output_dir / "stage1_all_head_step_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_grounded": summary["num_grounded"],
        "num_hallucinated": summary["num_hallucinated"],
        "num_heads_analyzed": summary["num_heads_analyzed"],
        "num_d1_heads": summary["num_d1_heads"],
        "early_token_step_cut": summary["early_token_step_cut"],
        "late_token_step_cut": summary["late_token_step_cut"],
        "early_counts": {
            "all": summary["early_n"],
            "grounded": summary["early_grounded_n"],
            "hallucinated": summary["early_hallucinated_n"],
        },
        "late_counts": {
            "all": summary["late_n"],
            "grounded": summary["late_grounded_n"],
            "hallucinated": summary["late_hallucinated_n"],
        },
        "label_conditional_windows": summary["label_conditional_windows"],
        "d1_overlap_summary": overlap_rows,
        "top_heads_preview": {
            name: heads[:5]
            for name, heads in top_lists.items()
        },
        "outputs": summary["outputs"],
        "plots": plot_paths,
    }, indent=2))


if __name__ == "__main__":
    main()
