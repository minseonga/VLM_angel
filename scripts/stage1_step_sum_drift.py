#!/usr/bin/env python
"""
Check whether head-contribution sum metrics drift with decoding step.

This is a forward-free diagnostic. It reads the existing Stage 1 head-logit
contribution trace and optionally a D.1 selected-head JSON, then summarizes how
sum-like metrics change across token_step bins.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from stage1_analyze_head_contrib import binary_auc, write_csv
from stage1_common import torch_load_compat


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze decoding-step drift in contribution sum metrics.")
    parser.add_argument("--contrib-file", type=str, default="stage1_outputs_n500/stage1_head_logit_contrib.pt")
    parser.add_argument("--heads-json", type=str, default="stage1_outputs_n500_d1_wrong_heads/stage1_d1_wrong_heads.json")
    parser.add_argument("--head-key", type=str, default="wrong_heads")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_n500_step_sum_drift")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument("--token-step-bins", type=int, default=20)
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument(
        "--plot-metrics",
        nargs="*",
        default=None,
        help="Metrics to visualize. Defaults to all available sum/count metrics.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_heads(path, key):
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        payload = json.load(f)
    heads = []
    seen = set()
    for item in payload.get(key, []):
        layer = int(item["layer"])
        head = int(item["head"])
        if (layer, head) in seen:
            continue
        seen.add((layer, head))
        heads.append({
            "layer": layer,
            "head": head,
            "activation_threshold": float(item.get("activation_threshold", 0.0)),
            "selection_rank": int(item.get("selection_rank", len(heads) + 1)),
        })
    return heads


def labels_hallucinated(rows):
    return np.array([1 if int(row["label"]) == 0 else 0 for row in rows], dtype=np.int64)


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


def linear_r2(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return None
    pooled = math.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    if pooled == 0:
        return None
    return float((b.mean() - a.mean()) / pooled)


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "q10": None,
            "q50": None,
            "q90": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
    }


def token_step_bins(rows, num_bins):
    steps = np.array([float(row["token_step"]) for row in rows], dtype=np.float64)
    edges = np.quantile(steps, np.linspace(0, 1, num_bins + 1))
    bins = []
    for idx in range(num_bins):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        if idx == num_bins - 1:
            bucket_rows = [row for row in rows if lo <= float(row["token_step"]) <= hi]
        else:
            bucket_rows = [row for row in rows if lo <= float(row["token_step"]) < hi]
        if bucket_rows:
            bins.append({"bucket": idx, "lo": lo, "hi": hi, "rows": bucket_rows})
    return bins


def bucket_overview_rows(bins):
    rows = []
    for bucket in bins:
        bucket_rows = bucket["rows"]
        grounded_n = int(sum(int(row["label"]) == 1 for row in bucket_rows))
        hallucinated_n = int(sum(int(row["label"]) == 0 for row in bucket_rows))
        rows.append({
            "bucket": bucket["bucket"],
            "token_step_lo": bucket["lo"],
            "token_step_hi": bucket["hi"],
            "token_step_mid": float((bucket["lo"] + bucket["hi"]) / 2.0),
            "n": int(len(bucket_rows)),
            "grounded_n": grounded_n,
            "hallucinated_n": hallucinated_n,
            "hallucination_rate": float(hallucinated_n / len(bucket_rows)) if bucket_rows else None,
        })
    return rows


def record_metrics(record, heads, start_layer, end_layer):
    contrib = record["head_logit_contrib"].detach().cpu().numpy()[:, :, 0]
    band = contrib[start_layer: end_layer + 1]
    actual = band.reshape(-1)

    row = {
        "object_index": int(record["object_index"]),
        "image_id": int(record["image_id"]),
        "label": int(record["label"]),
        "label_name": "hallucinated" if int(record["label"]) == 0 else "grounded",
        "surface_word": record.get("surface_word", ""),
        "node_word": record.get("node_word", ""),
        "token_step": int(record["token_step"]),
        "svar": float(record["svar"]),
        "all_head_signed_sum": float(actual.sum()),
        "all_head_pos_sum": float(actual[actual > 0].sum()) if np.any(actual > 0) else 0.0,
        "all_head_neg_sum": float(actual[actual < 0].sum()) if np.any(actual < 0) else 0.0,
        "all_head_abs_sum": float(np.abs(actual).sum()),
        "all_head_mean": float(actual.mean()),
    }

    if heads:
        vals = np.array([contrib[head["layer"], head["head"]] for head in heads], dtype=np.float64)
        thresholds = np.array([head["activation_threshold"] for head in heads], dtype=np.float64)
        active = vals > thresholds
        active_vals = vals * active
        row.update({
            "selected_head_count": int(len(heads)),
            "selected_contrib_sum": float(vals.sum()),
            "selected_contrib_mean": float(vals.mean()),
            "selected_abs_sum": float(np.abs(vals).sum()),
            "selected_pos_sum": float(vals[vals > 0].sum()) if np.any(vals > 0) else 0.0,
            "selected_active_contrib_sum": float(active_vals.sum()),
            "active_head_count": int(active.sum()),
            "any_active": int(active.any()),
            "active_head_fraction": float(active.mean()),
        })
    else:
        row.update({
            "selected_head_count": 0,
            "selected_contrib_sum": "",
            "selected_contrib_mean": "",
            "selected_abs_sum": "",
            "selected_pos_sum": "",
            "selected_active_contrib_sum": "",
            "active_head_count": "",
            "any_active": "",
            "active_head_fraction": "",
        })
    return row


def valid_metric_rows(rows, metric):
    return [row for row in rows if row.get(metric) not in ("", None)]


def metric_array(rows, metric):
    return np.array([float(row[metric]) for row in rows], dtype=np.float64)


def label_group_rows(rows, label_group):
    if label_group == "all":
        return rows
    target = 1 if label_group == "grounded" else 0
    return [row for row in rows if int(row["label"]) == target]


def bin_summary_rows(rows, bins, metrics, min_class_count):
    out = []
    for bucket in bins:
        bucket_rows = bucket["rows"]
        grounded_n = int(sum(int(row["label"]) == 1 for row in bucket_rows))
        hallucinated_n = int(sum(int(row["label"]) == 0 for row in bucket_rows))
        labels = labels_hallucinated(bucket_rows)
        base = {
            "bucket": bucket["bucket"],
            "token_step_lo": bucket["lo"],
            "token_step_hi": bucket["hi"],
            "n": int(len(bucket_rows)),
            "grounded_n": grounded_n,
            "hallucinated_n": hallucinated_n,
            "hallucination_rate": float(hallucinated_n / len(bucket_rows)) if bucket_rows else None,
        }
        for metric in metrics:
            cur_rows = valid_metric_rows(bucket_rows, metric)
            if not cur_rows:
                continue
            for label_group in ["all", "grounded", "hallucinated"]:
                group_rows = label_group_rows(cur_rows, label_group)
                stats = summarize_values(metric_array(group_rows, metric)) if group_rows else summarize_values([])
                out.append({
                    **base,
                    "metric": metric,
                    "label_group": label_group,
                    **stats,
                })

            if grounded_n >= min_class_count and hallucinated_n >= min_class_count:
                scores = metric_array(cur_rows, metric)
                auc = binary_auc(labels_hallucinated(cur_rows), scores)
                grounded = metric_array(label_group_rows(cur_rows, "grounded"), metric)
                hallucinated = metric_array(label_group_rows(cur_rows, "hallucinated"), metric)
                out.append({
                    **base,
                    "metric": metric,
                    "label_group": "hallucinated_minus_grounded",
                    "n": int(len(cur_rows)),
                    "mean": float(hallucinated.mean() - grounded.mean()),
                    "std": None,
                    "q10": None,
                    "q50": None,
                    "q90": None,
                    "auc_hallucinated_high": auc,
                    "oriented_auc": max(auc, 1.0 - auc) if auc is not None else None,
                })
    return out


def exact_step_summary_rows(rows, metrics, min_n):
    out = []
    steps = sorted({int(row["token_step"]) for row in rows})
    for step in steps:
        step_rows = [row for row in rows if int(row["token_step"]) == step]
        if len(step_rows) < min_n:
            continue
        grounded_n = int(sum(int(row["label"]) == 1 for row in step_rows))
        hallucinated_n = int(sum(int(row["label"]) == 0 for row in step_rows))
        for metric in metrics:
            cur_rows = valid_metric_rows(step_rows, metric)
            if not cur_rows:
                continue
            for label_group in ["all", "grounded", "hallucinated"]:
                group_rows = label_group_rows(cur_rows, label_group)
                if not group_rows:
                    continue
                stats = summarize_values(metric_array(group_rows, metric))
                out.append({
                    "token_step": int(step),
                    "metric": metric,
                    "label_group": label_group,
                    "n": int(len(group_rows)),
                    "grounded_n": grounded_n,
                    "hallucinated_n": hallucinated_n,
                    "hallucination_rate": float(hallucinated_n / len(step_rows)) if step_rows else None,
                    **stats,
                })
    return out


def metric_drift_summary(rows, bins, metrics):
    out = {}
    first_rows = bins[0]["rows"] if bins else []
    last_rows = bins[-1]["rows"] if bins else []
    for metric in metrics:
        cur_rows = valid_metric_rows(rows, metric)
        if not cur_rows:
            continue
        cur_steps = np.array([float(row["token_step"]) for row in cur_rows], dtype=np.float64)
        values = metric_array(cur_rows, metric)

        first_metric_rows = valid_metric_rows(first_rows, metric)
        last_metric_rows = valid_metric_rows(last_rows, metric)
        first_vals = metric_array(first_metric_rows, metric) if first_metric_rows else np.array([])
        last_vals = metric_array(last_metric_rows, metric) if last_metric_rows else np.array([])

        first_grounded = metric_array(label_group_rows(first_metric_rows, "grounded"), metric) if first_metric_rows else np.array([])
        last_grounded = metric_array(label_group_rows(last_metric_rows, "grounded"), metric) if last_metric_rows else np.array([])
        first_hallucinated = metric_array(label_group_rows(first_metric_rows, "hallucinated"), metric) if first_metric_rows else np.array([])
        last_hallucinated = metric_array(label_group_rows(last_metric_rows, "hallucinated"), metric) if last_metric_rows else np.array([])

        first_mean = float(first_vals.mean()) if len(first_vals) else None
        last_mean = float(last_vals.mean()) if len(last_vals) else None
        out[metric] = {
            "n": int(len(cur_rows)),
            "pearson_token_step": pearson(cur_steps, values),
            "spearman_token_step": spearman(cur_steps, values),
            "linear_r2_token_step": linear_r2(cur_steps, values),
            "first_bin_step_range": [bins[0]["lo"], bins[0]["hi"]] if bins else None,
            "last_bin_step_range": [bins[-1]["lo"], bins[-1]["hi"]] if bins else None,
            "first_bin_mean": first_mean,
            "last_bin_mean": last_mean,
            "last_minus_first": (
                float(last_mean - first_mean)
                if first_mean is not None and last_mean is not None
                else None
            ),
            "last_vs_first_cohens_d": cohen_d(first_vals, last_vals),
            "grounded_last_minus_first": (
                float(last_grounded.mean() - first_grounded.mean())
                if len(first_grounded) and len(last_grounded)
                else None
            ),
            "hallucinated_last_minus_first": (
                float(last_hallucinated.mean() - first_hallucinated.mean())
                if len(first_hallucinated) and len(last_hallucinated)
                else None
            ),
        }
    return out


def csv_safe_rows(rows):
    safe = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, (list, tuple)):
                item[key] = json.dumps(value)
            else:
                item[key] = value
        safe.append(item)
    return safe


def finite_points(points):
    return [
        (float(x), float(y))
        for x, y in points
        if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))
    ]


def svg_line_plot(series, title, y_label, path, width=920, height=520):
    margin_left = 78
    margin_right = 170
    margin_top = 48
    margin_bottom = 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_points = []
    for item in series:
        item["points"] = finite_points(item["points"])
        all_points.extend(item["points"])
    if not all_points:
        return False

    x_vals = np.array([x for x, _ in all_points], dtype=np.float64)
    y_vals = np.array([y for _, y in all_points], dtype=np.float64)
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    def sx(x):
        return margin_left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return margin_top + (y_max - float(y)) / (y_max - y_min) * plot_h

    def fmt(value):
        return f"{value:.3g}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="28" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">decoding token_step bin midpoint</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{y_label}</text>',
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

    parts.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.2"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111827" stroke-width="1.2"/>')

    for item in series:
        points = item["points"]
        if len(points) == 0:
            continue
        color = item["color"]
        path_points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        parts.append(f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.8" fill="{color}"/>')

    legend_x = margin_left + plot_w + 24
    legend_y = margin_top + 10
    for idx, item in enumerate(series):
        y = legend_y + idx * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{item["color"]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 30}" y="{y + 4}" font-family="Arial" font-size="12" fill="#111827">{item["name"]}</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return True


def write_step_plots(bin_rows, overview_rows, metrics, output_dir):
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    colors = {
        "all": "#4b5563",
        "grounded": "#2563eb",
        "hallucinated": "#dc2626",
    }
    for metric in metrics:
        series = []
        for label_group in ["all", "grounded", "hallucinated"]:
            points = []
            for row in bin_rows:
                if row.get("metric") != metric or row.get("label_group") != label_group:
                    continue
                if row.get("mean") is None:
                    continue
                mid = (float(row["token_step_lo"]) + float(row["token_step_hi"])) / 2.0
                points.append((mid, float(row["mean"])))
            series.append({
                "name": label_group,
                "color": colors[label_group],
                "points": points,
            })
        path = plot_dir / f"{metric}_by_token_step.svg"
        if svg_line_plot(series, f"{metric} by decoding step", metric, path):
            paths.append(str(path))

    rate_series = [{
        "name": "hallucination_rate",
        "color": "#7c3aed",
        "points": [
            (row["token_step_mid"], row["hallucination_rate"])
            for row in overview_rows
            if row["hallucination_rate"] is not None
        ],
    }]
    rate_path = plot_dir / "hallucination_rate_by_token_step.svg"
    if svg_line_plot(rate_series, "Hallucination rate by decoding step", "hallucination rate", rate_path):
        paths.append(str(rate_path))
    return paths


def main():
    args = parse_args()
    if args.token_step_bins < 2:
        raise ValueError("--token-step-bins must be >= 2.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heads = load_heads(args.heads_json, args.head_key)
    payload = torch_load_compat(args.contrib_file, map_location="cpu")
    records = payload["records"]
    rows = [
        record_metrics(record, heads, args.start_layer, args.end_layer)
        for record in records
    ]

    metrics = [
        "all_head_signed_sum",
        "all_head_pos_sum",
        "all_head_neg_sum",
        "all_head_abs_sum",
        "all_head_mean",
    ]
    if heads:
        metrics.extend([
            "selected_contrib_sum",
            "selected_contrib_mean",
            "selected_abs_sum",
            "selected_pos_sum",
            "selected_active_contrib_sum",
            "active_head_count",
            "any_active",
            "active_head_fraction",
        ])

    bins = token_step_bins(rows, args.token_step_bins)
    overview_rows = bucket_overview_rows(bins)
    bin_rows = bin_summary_rows(rows, bins, metrics, args.min_class_count)
    exact_rows = exact_step_summary_rows(rows, metrics, min_n=args.min_class_count)
    drift = metric_drift_summary(rows, bins, metrics)
    plot_metrics = metrics if args.plot_metrics is None else [metric for metric in args.plot_metrics if metric in metrics]
    plot_paths = [] if args.no_plots else write_step_plots(bin_rows, overview_rows, plot_metrics, output_dir)

    summary = {
        "contrib_file": args.contrib_file,
        "heads_json": args.heads_json if heads else None,
        "head_key": args.head_key if heads else None,
        "num_records": len(rows),
        "num_grounded": int(sum(int(row["label"]) == 1 for row in rows)),
        "num_hallucinated": int(sum(int(row["label"]) == 0 for row in rows)),
        "num_heads": len(heads),
        "layer_band": [args.start_layer, args.end_layer],
        "token_step_bins": args.token_step_bins,
        "metrics": metrics,
        "plot_metrics": plot_metrics,
        "drift_summary": drift,
        "top_abs_step_drift": sorted(
            [
                {
                    "metric": metric,
                    "pearson_token_step": stats.get("pearson_token_step"),
                    "spearman_token_step": stats.get("spearman_token_step"),
                    "linear_r2_token_step": stats.get("linear_r2_token_step"),
                    "first_bin_mean": stats.get("first_bin_mean"),
                    "last_bin_mean": stats.get("last_bin_mean"),
                    "last_minus_first": stats.get("last_minus_first"),
                    "last_vs_first_cohens_d": stats.get("last_vs_first_cohens_d"),
                    "grounded_last_minus_first": stats.get("grounded_last_minus_first"),
                    "hallucinated_last_minus_first": stats.get("hallucinated_last_minus_first"),
                }
                for metric, stats in drift.items()
                if stats.get("pearson_token_step") is not None
            ],
            key=lambda item: abs(item["pearson_token_step"]),
            reverse=True,
        ),
        "outputs": {
            "record_metrics": str(output_dir / "stage1_step_sum_drift_record_metrics.csv"),
            "bucket_overview": str(output_dir / "stage1_step_sum_drift_bucket_overview.csv"),
            "bin_summary": str(output_dir / "stage1_step_sum_drift_bin_summary.csv"),
            "exact_step_summary": str(output_dir / "stage1_step_sum_drift_exact_step_summary.csv"),
            "summary": str(output_dir / "stage1_step_sum_drift_summary.json"),
            "plots": plot_paths,
        },
    }

    write_csv(rows, output_dir / "stage1_step_sum_drift_record_metrics.csv")
    write_csv(csv_safe_rows(overview_rows), output_dir / "stage1_step_sum_drift_bucket_overview.csv")
    write_csv(csv_safe_rows(bin_rows), output_dir / "stage1_step_sum_drift_bin_summary.csv")
    write_csv(csv_safe_rows(exact_rows), output_dir / "stage1_step_sum_drift_exact_step_summary.csv")
    with open(output_dir / "stage1_step_sum_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "num_records": summary["num_records"],
        "num_grounded": summary["num_grounded"],
        "num_hallucinated": summary["num_hallucinated"],
        "num_heads": summary["num_heads"],
        "top_abs_step_drift": summary["top_abs_step_drift"][:8],
        "outputs": summary["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
