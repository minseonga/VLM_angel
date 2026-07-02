import csv
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


DEFAULT_IMAGE_ROOT = first_existing_path(
    "/mnt/pilab_nas/homes/mskang/data/val2014",
    "/home/kms/data/pope/val2014",
)
DEFAULT_ANNOTATION_DIR = first_existing_path(
    "/mnt/pilab_nas/homes/mskang/data/annotations",
    "/home/kms/data/images/mscoco/annotations",
)
DEFAULT_INSTRUCTION_PATH = str(REPO_ROOT / "examples" / "toy_img_query_list.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def coco_image_filename(image_id):
    return f"COCO_val2014_{int(image_id):012d}.jpg"


def coco_image_path(image_root, image_id):
    return os.path.join(image_root, coco_image_filename(image_id))


def load_chair_evaluator(cache_path, annotations_path):
    from chair import CHAIR

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    evaluator = CHAIR(annotations_path)
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(evaluator, f)
    return evaluator


def first_token_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        return None
    return int(ids[0])


def token_id_candidates(tokenizer, text):
    variants = []
    raw = str(text).strip()
    if raw:
        variants.extend([raw, " " + raw, raw.lower(), " " + raw.lower()])
        if raw.endswith("s"):
            variants.extend([raw[:-1], " " + raw[:-1]])

    out = []
    seen = set()
    for variant in variants:
        tok = first_token_id(tokenizer, variant)
        if tok is None or tok in seen:
            continue
        seen.add(tok)
        out.append({"variant": variant, "token_id": tok})
    return out


def split_generated_ids(sequence_ids, input_len, num_steps):
    if len(sequence_ids) >= input_len + num_steps:
        return sequence_ids[input_len: input_len + num_steps]
    return sequence_ids[-num_steps:]


def find_generated_token_step(tokenizer, text, sequence_ids, input_len, num_steps, start_step=0):
    generated_ids = split_generated_ids(sequence_ids, input_len, num_steps)
    candidates = token_id_candidates(tokenizer, text)

    for candidate in candidates:
        token_id = candidate["token_id"]
        for step in range(max(0, start_step), len(generated_ids)):
            if int(generated_ids[step]) == token_id:
                return {
                    "step": step,
                    "matched_text": candidate["variant"],
                    "matched_token_id": token_id,
                }

    # Fallback: allow matches in the full sequence and map them back to generation steps.
    gen_start = len(sequence_ids) - num_steps
    if len(sequence_ids) >= input_len + num_steps:
        gen_start = input_len
    for candidate in candidates:
        token_id = candidate["token_id"]
        for pos, seq_token_id in enumerate(sequence_ids):
            step = pos - gen_start
            if step < max(0, start_step) or step >= num_steps:
                continue
            if int(seq_token_id) == token_id:
                return {
                    "step": step,
                    "matched_text": candidate["variant"],
                    "matched_token_id": token_id,
                }

    raise ValueError(f"Could not align object text to generated token: {text!r}")


def visual_attention_matrix_for_step(outputs, step, vision_token_start, vision_token_end):
    step_attentions = outputs["attentions"][step]
    matrix = np.zeros((len(step_attentions), step_attentions[0].shape[1]), dtype=np.float32)
    for layer_idx, layer_attn in enumerate(step_attentions):
        visual_mass = layer_attn[0, :, -1, vision_token_start:vision_token_end].sum(dim=-1)
        matrix[layer_idx] = visual_mass.detach().float().cpu().numpy()
    return matrix


def svar_band_score(var_matrix, start_layer, end_layer):
    # Matches the notebook convention: mean over heads per layer, summed across layers.
    return float(var_matrix[start_layer: end_layer + 1].mean(axis=1).sum())


def object_mentions_from_caption(evaluator, gt_words, caption):
    words, node_words, idxs, raw_words = evaluator.caption_to_words(caption)
    mentions = []
    for mention_idx, (surface_word, node_word, word_idx) in enumerate(zip(words, node_words, idxs)):
        mentions.append({
            "mention_idx": int(mention_idx),
            "surface_word": surface_word,
            "node_word": node_word,
            "chair_word_idx": int(word_idx),
            "label": int(node_word in set(gt_words)),
        })
    return mentions, raw_words


def compute_overlap(records, score_key="svar"):
    grounded = np.array([float(r[score_key]) for r in records if int(r["label"]) == 1], dtype=np.float64)
    hallucinated = np.array([float(r[score_key]) for r in records if int(r["label"]) == 0], dtype=np.float64)

    summary = {
        "score_key": score_key,
        "num_records": len(records),
        "num_grounded": int(len(grounded)),
        "num_hallucinated": int(len(hallucinated)),
        "intervals": {},
    }
    if len(grounded) == 0 or len(hallucinated) == 0:
        return summary

    for name, quantiles in {
        "iqr": (0.25, 0.75),
        "q10_q90": (0.10, 0.90),
    }.items():
        g_lo, g_hi = np.quantile(grounded, quantiles)
        h_lo, h_hi = np.quantile(hallucinated, quantiles)
        lo = float(max(g_lo, h_lo))
        hi = float(min(g_hi, h_hi))
        count_g = count_h = 0
        if lo <= hi:
            count_g = int(((grounded >= lo) & (grounded <= hi)).sum())
            count_h = int(((hallucinated >= lo) & (hallucinated <= hi)).sum())
        summary["intervals"][name] = {
            "grounded_quantile_interval": [float(g_lo), float(g_hi)],
            "hallucinated_quantile_interval": [float(h_lo), float(h_hi)],
            "overlap_interval": [lo, hi],
            "valid": bool(lo <= hi),
            "count_grounded": count_g,
            "count_hallucinated": count_h,
            "count_total": int(count_g + count_h),
        }
    return summary


def annotate_overlap_flags(records, overlap_summary, score_key="svar"):
    for record in records:
        score = float(record[score_key])
        for name, interval in overlap_summary.get("intervals", {}).items():
            lo, hi = interval["overlap_interval"]
            record[f"overlap_{name}"] = bool(interval["valid"] and lo <= score <= hi)


def write_records_csv(records, path, exclude_keys=None):
    exclude = set(exclude_keys or [])
    keys = []
    for record in records:
        for key, value in record.items():
            if key in exclude:
                continue
            if isinstance(value, (list, tuple, dict)) or hasattr(value, "shape"):
                continue
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in keys})


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def torch_load_compat(path, map_location="cpu"):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
