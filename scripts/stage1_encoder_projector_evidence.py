#!/usr/bin/env python
"""
Attach encoder/projector object-evidence scores to generated object mentions.

Input is an object mention CSV from max-length sweep or Stage 1. For each
(image, object) pair, this computes:
  - encoder_clip_*: CLIP projected patch/text similarity
  - projector_llm_*: LLaVA projected visual token / LLM object embedding similarity
Then it merges the scores back onto every object mention row.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from stage1_common import DEFAULT_IMAGE_ROOT, coco_image_path


def parse_args():
    parser = argparse.ArgumentParser(description="Encoder/projector object evidence pilot.")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--object-csv", type=str, required=True)
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_encoder_projector_evidence")
    parser.add_argument("--output-file", type=str, default="encoder_projector_object_evidence.csv")
    parser.add_argument("--summary-file", type=str, default="encoder_projector_evidence_summary.json")
    parser.add_argument("--clip-model", type=str, default=None, help="Defaults to model config mm_vision_tower.")
    parser.add_argument("--skip-clip-encoder", action="store_true")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--batch-objects", type=int, default=64)
    return parser.parse_args()


def normalize(x):
    return F.normalize(x.float(), dim=-1)


def token_embedding_for_text(tokenizer, embed_tokens, text, device):
    variants = []
    raw = str(text).strip()
    if raw:
        variants.extend([raw, " " + raw, raw.lower(), " " + raw.lower()])
        if raw.endswith("s"):
            variants.extend([raw[:-1], " " + raw[:-1]])
    embs = []
    seen = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        ids = tuple(int(x) for x in ids if int(x) >= 0)
        if not ids or ids in seen:
            continue
        seen.add(ids)
        t = torch.tensor(ids, device=device, dtype=torch.long)
        emb = embed_tokens(t).float().mean(dim=0)
        embs.append(emb)
    if not embs:
        return None
    return normalize(torch.stack(embs, dim=0)).mean(dim=0)


def similarity_stats(patches, query):
    sims = normalize(patches) @ normalize(query.view(1, -1)).squeeze(0)
    topk = torch.topk(sims, k=min(5, sims.numel())).values
    return {
        "max": float(sims.max().item()),
        "mean": float(sims.mean().item()),
        "top5_mean": float(topk.mean().item()),
        "top5_sum": float(topk.sum().item()),
    }


def tensor_stats(prefix, x):
    norms = x.float().norm(dim=-1)
    return {
        f"{prefix}_patch_norm_mean": float(norms.mean().item()),
        f"{prefix}_patch_norm_std": float(norms.std().item()),
        f"{prefix}_patch_norm_max": float(norms.max().item()),
    }


def compute_auc_summary(df, metrics):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return {}
    out = {}
    if "label" not in df.columns:
        return out
    y = (df["label"].astype(int) == 0).astype(int)
    if y.nunique() < 2:
        return out
    for metric in metrics:
        if metric not in df.columns:
            continue
        x = pd.to_numeric(df[metric], errors="coerce")
        mask = x.notna()
        if mask.sum() == 0 or y[mask].nunique() < 2:
            continue
        auc = float(roc_auc_score(y[mask], x[mask]))
        out[metric] = {
            "auc_hallucinated_high": auc,
            "auc_direction_adjusted": max(auc, 1.0 - auc),
            "direction": "high_hall" if auc >= 0.5 else "low_hall",
        }
    return out


def main():
    args = parse_args()
    if not args.model_path:
        raise ValueError("Set --model-path or LLAVA_MODEL_PATH for LLaVA weights.")

    from llava.mm_utils import process_images
    from model_manager import ModelManager
    from utils import disable_torch_init, setup_seeds

    setup_seeds()
    disable_torch_init()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(args.object_csv)
    if args.num_images is not None:
        keep_images = list(rows["image_id"].drop_duplicates().astype(int))[: args.num_images]
        rows = rows[rows["image_id"].astype(int).isin(keep_images)].copy()

    objects = sorted(str(x) for x in rows["node_word"].dropna().unique())
    image_ids = list(rows["image_id"].drop_duplicates().astype(int))

    model_manager = ModelManager(args.model, model_path=args.model_path)
    model = model_manager.llm_model
    device = model.device
    vision_tower = model.get_vision_tower()
    mm_projector = model.get_model().mm_projector
    embed_tokens = model.get_model().embed_tokens

    clip_model = None
    clip_tokenizer = None
    if not args.skip_clip_encoder:
        from transformers import CLIPModel, CLIPTokenizer
        clip_name = args.clip_model or getattr(model.config, "mm_vision_tower", None) or "openai/clip-vit-large-patch14-336"
        clip_model = CLIPModel.from_pretrained(clip_name).to(device=device, dtype=torch.float16).eval()
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_name)

    llm_object_embs = {}
    for obj in objects:
        emb = token_embedding_for_text(model_manager.tokenizer, embed_tokens, obj, device)
        if emb is not None:
            llm_object_embs[obj] = emb

    clip_text_embs = {}
    if clip_model is not None:
        for start in range(0, len(objects), args.batch_objects):
            batch = objects[start: start + args.batch_objects]
            prompts = [f"a photo of a {obj}" for obj in batch]
            toks = clip_tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                txt = clip_model.get_text_features(**toks)
            txt = normalize(txt)
            for obj, emb in zip(batch, txt):
                clip_text_embs[obj] = emb.detach()

    evidence_by_image_object = {}
    for image_id in tqdm(image_ids, desc="encoder/projector evidence"):
        image = Image.open(coco_image_path(args.data_path, image_id)).convert("RGB")
        image_tensor = process_images([image], model_manager.image_processor, model.config).to(device, dtype=torch.float16)
        with torch.inference_mode():
            encoder_patches = vision_tower(image_tensor)[0].detach().float()
            projected_patches = mm_projector(encoder_patches.to(device=device, dtype=torch.float16)).detach().float()

            clip_projected_patches = None
            clip_global = None
            if clip_model is not None:
                image_forward = clip_model.vision_model(
                    pixel_values=image_tensor.to(device=device, dtype=torch.float16),
                    output_hidden_states=True,
                )
                select_layer = getattr(model.config, "mm_vision_select_layer", -2)
                patch_hidden = image_forward.hidden_states[select_layer][:, 1:, :][0].detach().float()
                clip_projected_patches = clip_model.visual_projection(patch_hidden.to(device=device, dtype=torch.float16)).detach().float()
                clip_global = clip_model.get_image_features(pixel_values=image_tensor.to(device=device, dtype=torch.float16))[0].detach().float()

        image_objects = sorted(str(x) for x in rows.loc[rows["image_id"].astype(int) == image_id, "node_word"].dropna().unique())
        for obj in image_objects:
            rec = {"image_id": int(image_id), "node_word": obj}
            rec.update(tensor_stats("encoder", encoder_patches))
            rec.update(tensor_stats("projector", projected_patches))

            llm_emb = llm_object_embs.get(obj)
            if llm_emb is not None:
                stats = similarity_stats(projected_patches, llm_emb)
                rec.update({f"projector_llm_{k}": v for k, v in stats.items()})

            if clip_projected_patches is not None and obj in clip_text_embs:
                text_emb = clip_text_embs[obj]
                stats = similarity_stats(clip_projected_patches, text_emb)
                rec.update({f"encoder_clip_{k}": v for k, v in stats.items()})
                rec["encoder_clip_global"] = float((normalize(clip_global.view(1, -1))[0] * text_emb).sum().item())

            evidence_by_image_object[(int(image_id), obj)] = rec

        del image_tensor, encoder_patches, projected_patches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evidence = pd.DataFrame(evidence_by_image_object.values())
    merged = rows.merge(evidence, on=["image_id", "node_word"], how="left")
    out_path = output_dir / args.output_file
    merged.to_csv(out_path, index=False)

    candidate_metrics = [
        "encoder_clip_max",
        "encoder_clip_top5_mean",
        "encoder_clip_global",
        "projector_llm_max",
        "projector_llm_top5_mean",
        "projector_llm_mean",
        "projector_patch_norm_mean",
        "encoder_patch_norm_mean",
    ]
    summary = {
        "config": vars(args),
        "num_rows": int(len(merged)),
        "num_images": int(merged["image_id"].nunique()),
        "num_objects": int(merged["node_word"].nunique()),
        "num_image_object_pairs": int(len(evidence)),
        "metric_means_by_label": merged.groupby("label")[[m for m in candidate_metrics if m in merged.columns]].mean().to_dict() if "label" in merged.columns else {},
        "auc_summary": compute_auc_summary(merged, candidate_metrics + ["token_step", "step_frac", "chair_word_idx", "mention_idx"]),
    }
    with open(output_dir / args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved merged evidence: {out_path}")


if __name__ == "__main__":
    main()
