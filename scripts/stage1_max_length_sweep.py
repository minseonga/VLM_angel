#!/usr/bin/env python
"""
Generate captions for the same images while sweeping max_new_tokens.

With --trace-metrics, this also records per-step confidence metrics,
visual-attention metrics, object-aligned metrics, and per-layer/per-head visual
attention mass at object-token steps.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from stage1_common import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_INSTRUCTION_PATH,
    coco_image_filename,
    coco_image_path,
    find_generated_token_step,
    load_chair_evaluator,
    load_jsonl,
    object_mentions_from_caption,
    split_generated_ids,
    svar_band_score,
    visual_attention_matrix_for_step,
)


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def write_jsonl(records, path):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def write_csv(records, path):
    if not records:
        return
    keys = sorted({key for record in records for key in record.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def summarize(caption_records, object_records, max_tokens_list):
    out = {
        "num_captions": len(caption_records),
        "num_object_mentions": len(object_records),
        "by_max_tokens": {},
    }
    for max_tokens in max_tokens_list:
        caps = [r for r in caption_records if int(r["max_tokens"]) == max_tokens]
        objs = [r for r in object_records if int(r["max_tokens"]) == max_tokens]
        num_hall = sum(1 for r in objs if int(r["label"]) == 0)
        out["by_max_tokens"][str(max_tokens)] = {
            "num_captions": len(caps),
            "num_object_mentions": len(objs),
            "num_grounded_mentions": int(sum(1 for r in objs if int(r["label"]) == 1)),
            "num_hallucinated_mentions": int(num_hall),
            "object_hallucination_rate": float(num_hall / len(objs)) if objs else None,
            "mean_num_steps": float(sum(float(r["num_steps"]) for r in caps) / len(caps)) if caps else None,
            "mean_caption_chars": float(sum(len(r["caption"]) for r in caps) / len(caps)) if caps else None,
        }
    return out


def attention_summary(var_matrix, start_layer, end_layer):
    band = var_matrix[start_layer: end_layer + 1]
    flat = var_matrix.reshape(-1)
    band_flat = band.reshape(-1)
    return {
        "visual_attn_mean_all": float(var_matrix.mean()),
        "visual_attn_sum_all": float(var_matrix.sum()),
        "visual_attn_max_head_all": float(flat.max()) if flat.size else 0.0,
        "visual_attn_mean_band": float(band.mean()),
        "visual_attn_sum_band": float(band.sum()),
        "visual_attn_max_head_band": float(band_flat.max()) if band_flat.size else 0.0,
        "svar_band": svar_band_score(var_matrix, start_layer, end_layer),
    }


def confidence_summary(logits, actual_token_id, tokenizer):
    logits = logits.detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    vocab_size = int(probs.numel())
    top_vals, top_ids = torch.topk(probs, k=min(10, vocab_size))
    actual_token_id = int(actual_token_id)
    actual_prob = float(probs[actual_token_id].item())
    actual_logprob = float(log_probs[actual_token_id].item())
    top1_prob = float(top_vals[0].item())
    top2_prob = float(top_vals[1].item()) if len(top_vals) > 1 else 0.0
    entropy = float(-(probs * log_probs).sum().item())
    return {
        "vocab_entropy": entropy,
        "vocab_normalized_entropy": float(entropy / np.log(vocab_size)),
        "top1_prob": top1_prob,
        "top1_token_id": int(top_ids[0].item()),
        "top1_token": tokenizer.decode([int(top_ids[0].item())], skip_special_tokens=False),
        "top2_prob": top2_prob,
        "top1_top2_margin": float(top1_prob - top2_prob),
        "top5_mass": float(top_vals[:5].sum().item()),
        "top10_mass": float(top_vals[:10].sum().item()),
        "actual_token_id": actual_token_id,
        "actual_token": tokenizer.decode([actual_token_id], skip_special_tokens=False),
        "actual_prob": actual_prob,
        "actual_logprob": actual_logprob,
        "actual_surprisal": float(-actual_logprob),
        "actual_rank": int((probs > probs[actual_token_id]).sum().item()) + 1,
    }


def head_attention_records_from_matrix(var_matrix, base_record):
    rows = []
    for layer_idx in range(var_matrix.shape[0]):
        for head_idx in range(var_matrix.shape[1]):
            rows.append({
                **base_record,
                "layer": int(layer_idx),
                "head": int(head_idx),
                "visual_attn_mass": float(var_matrix[layer_idx, head_idx]),
            })
    return rows


def align_object(model_manager, mention, sequence_ids, input_len, num_steps, search_start):
    try:
        return find_generated_token_step(
            model_manager.tokenizer,
            mention["surface_word"],
            sequence_ids,
            input_len,
            num_steps,
            start_step=search_start,
        )
    except Exception:
        return find_generated_token_step(
            model_manager.tokenizer,
            mention["node_word"],
            sequence_ids,
            input_len,
            num_steps,
            start_step=search_start,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep max_new_tokens on the same images.")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--annotations-path", type=str, default=os.environ.get("ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR))
    parser.add_argument("--instruction-path", type=str, default=DEFAULT_INSTRUCTION_PATH)
    parser.add_argument("--cache", type=str, default="chair.pkl")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs_max_length_sweep")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens-list", type=str, default="16,32,64,128")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--trace-attention", action="store_true")
    parser.add_argument("--trace-confidence", action="store_true")
    parser.add_argument("--trace-metrics", action="store_true", help="Enable attention, confidence, and object-head metrics.")
    parser.add_argument("--save-object-heads", action="store_true", help="Save per-layer/per-head visual attention mass at object-token steps.")
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.trace_metrics:
        args.trace_attention = True
        args.trace_confidence = True
        args.save_object_heads = True
    if not args.model_path:
        raise ValueError("Set --model-path or LLAVA_MODEL_PATH for LLaVA weights.")

    from llava.mm_utils import process_images
    from model_manager import ModelManager
    from utils import disable_torch_init, setup_seeds

    setup_seeds()
    disable_torch_init()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_tokens_list = parse_int_list(args.max_tokens_list)
    instructions = load_jsonl(args.instruction_path)
    if args.num_samples is not None:
        instructions = instructions[: args.num_samples]

    evaluator = load_chair_evaluator(args.cache, args.annotations_path)
    model_manager = ModelManager(args.model, model_path=args.model_path)

    caption_records = []
    object_records = []
    step_metric_records = []
    object_head_records = []

    for image_index, item in tqdm(list(enumerate(instructions)), desc="images"):
        image_id = int(item["image_id"])
        instruction = item.get("instruction", "Please help me describe the image in detail.")
        image_path = coco_image_path(args.data_path, image_id)
        image = Image.open(image_path).convert("RGB")
        images_tensor = process_images(
            [image],
            model_manager.image_processor,
            model_manager.llm_model.config,
        ).to(model_manager.llm_model.device, dtype=torch.float16)

        _, input_ids, kwargs = model_manager.prepare_inputs_for_model(
            [instruction],
            images_tensor,
            use_dataloader=False,
        )
        input_len = int(input_ids.shape[1])
        image_filename = coco_image_filename(image_id)

        for max_tokens in max_tokens_list:
            with torch.inference_mode():
                outputs = model_manager.llm_model.generate(
                    input_ids,
                    do_sample=args.sample,
                    num_beams=args.beam,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    output_attentions=args.trace_attention,
                    output_scores=args.trace_confidence,
                    return_dict_in_generate=True,
                    **kwargs,
                )

            caption = model_manager.decode(outputs["sequences"])[0]
            if args.trace_confidence:
                num_steps = len(outputs["scores"])
            elif args.trace_attention:
                num_steps = len(outputs["attentions"])
            else:
                sequence_len = int(outputs["sequences"].shape[1])
                num_steps = sequence_len - input_len if sequence_len >= input_len else sequence_len

            sequence_ids = [int(x) for x in outputs["sequences"][0].detach().cpu().tolist()]
            generated_ids = [int(x) for x in split_generated_ids(sequence_ids, input_len, num_steps)]

            for step in range(num_steps):
                if not (args.trace_attention or args.trace_confidence):
                    break
                step_record = {
                    "image_index": int(image_index),
                    "image_id": image_id,
                    "max_tokens": int(max_tokens),
                    "token_step": int(step),
                    "num_steps": int(num_steps),
                    "step_frac": float(step / max(1, num_steps - 1)),
                    "generated_token_id": int(generated_ids[step]),
                }
                if args.trace_confidence:
                    step_record.update(confidence_summary(outputs["scores"][step][0], generated_ids[step], model_manager.tokenizer))
                if args.trace_attention:
                    var_matrix = visual_attention_matrix_for_step(
                        outputs,
                        step,
                        model_manager.img_start_idx,
                        model_manager.img_end_idx,
                    )
                    step_record.update(attention_summary(var_matrix, args.start_layer, args.end_layer))
                step_metric_records.append(step_record)

            chair_info = evaluator.compute_chair_token(image_filename, caption)
            gt_words = list(chair_info["mscoco_gt_words"])
            mentions, raw_words = object_mentions_from_caption(evaluator, gt_words, caption)

            caption_records.append({
                "image_index": int(image_index),
                "image_id": image_id,
                "image_filename": image_filename,
                "image_path": image_path,
                "instruction": instruction,
                "max_tokens": int(max_tokens),
                "num_steps": int(num_steps),
                "caption": caption,
                "raw_object_words": raw_words,
            })

            search_start = 0
            for mention in mentions:
                object_record = {
                    "image_index": int(image_index),
                    "image_id": image_id,
                    "max_tokens": int(max_tokens),
                    "num_steps": int(num_steps),
                    "caption": caption,
                    **mention,
                }
                if args.trace_attention or args.trace_confidence:
                    try:
                        alignment = align_object(model_manager, mention, sequence_ids, input_len, num_steps, search_start)
                    except Exception as exc:
                        object_record["alignment_error"] = str(exc)
                    else:
                        step = int(alignment["step"])
                        search_start = step + 1
                        object_record.update({
                            "token_step": step,
                            "step_frac": float(step / max(1, num_steps - 1)),
                            "matched_text": alignment["matched_text"],
                            "matched_token_id": int(alignment["matched_token_id"]),
                            "generated_token_id": int(generated_ids[step]),
                        })
                        if args.trace_confidence:
                            object_record.update(confidence_summary(outputs["scores"][step][0], generated_ids[step], model_manager.tokenizer))
                        if args.trace_attention:
                            var_matrix = visual_attention_matrix_for_step(
                                outputs,
                                step,
                                model_manager.img_start_idx,
                                model_manager.img_end_idx,
                            )
                            object_record.update(attention_summary(var_matrix, args.start_layer, args.end_layer))
                            if args.save_object_heads:
                                object_head_records.extend(head_attention_records_from_matrix(
                                    var_matrix,
                                    {
                                        "image_index": int(image_index),
                                        "image_id": image_id,
                                        "max_tokens": int(max_tokens),
                                        "mention_idx": int(mention["mention_idx"]),
                                        "node_word": mention["node_word"],
                                        "surface_word": mention["surface_word"],
                                        "label": int(mention["label"]),
                                        "token_step": step,
                                        "step_frac": float(step / max(1, num_steps - 1)),
                                    },
                                ))
                object_records.append(object_record)

            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_jsonl(caption_records, output_dir / "max_length_sweep_captions.jsonl")
    write_csv(object_records, output_dir / "max_length_sweep_object_mentions.csv")
    write_csv(step_metric_records, output_dir / "max_length_sweep_step_metrics.csv")
    write_csv(object_head_records, output_dir / "max_length_sweep_object_head_attention.csv")
    summary = summarize(caption_records, object_records, max_tokens_list)
    with open(output_dir / "max_length_sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
