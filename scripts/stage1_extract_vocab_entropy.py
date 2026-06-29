#!/usr/bin/env python
"""
Extract per-decoding-step vocabulary uncertainty metrics.

For each generated step this saves:
  - full-vocab entropy H(p)
  - normalized entropy H(p) / log(|V|)
  - min-entropy -log(max p)
  - top-k mass and entropy over the renormalized top-k distribution
  - bottom-k mass and entropy over the renormalized bottom-k distribution
  - actual generated token probability/rank

It also labels generated COCO object mentions with CHAIR and joins the step-level
metrics back to object-token records for grounded vs hallucinated analysis.
"""

import argparse
import json
import math
import os
from collections import Counter
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
    write_records_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-step vocab entropy/min-entropy/top-k metrics.")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--annotations-path", type=str, default=os.environ.get("ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR))
    parser.add_argument("--instruction-path", type=str, default=DEFAULT_INSTRUCTION_PATH)
    parser.add_argument("--cache", type=str, default="chair.pkl")
    parser.add_argument("--output-dir", type=str, default="stage1_vocab_entropy")
    parser.add_argument("--output-file", type=str, default="stage1_vocab_entropy.pt")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-k", nargs="+", type=int, default=[5, 10, 50, 100])
    parser.add_argument("--bottom-k", nargs="+", type=int, default=[5, 10, 50, 100])
    parser.add_argument("--save-top-tokens", type=int, default=10)
    return parser.parse_args()


def entropy_metrics_from_logits(logits, actual_token_id, tokenizer, top_ks, bottom_ks, save_top_tokens):
    logits = logits.detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    vocab_size = int(probs.numel())

    entropy = float(-(probs * log_probs).sum().item())
    max_prob, top1_id = torch.max(probs, dim=-1)
    max_prob = float(max_prob.item())
    top1_id = int(top1_id.item())
    min_entropy = float(-math.log(max(max_prob, 1e-45)))

    actual_prob = float(probs[int(actual_token_id)].item())
    actual_logprob = float(log_probs[int(actual_token_id)].item())
    # Rank is 1-based. It is the number of vocab items with prob strictly greater, plus one.
    actual_rank = int((probs > probs[int(actual_token_id)]).sum().item()) + 1

    record = {
        "vocab_size": vocab_size,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(vocab_size),
        "effective_vocab_size": float(math.exp(min(entropy, 80.0))),
        "min_entropy": min_entropy,
        "top1_prob": max_prob,
        "top1_token_id": top1_id,
        "top1_token": tokenizer.decode([top1_id], skip_special_tokens=False),
        "actual_token_id": int(actual_token_id),
        "actual_token": tokenizer.decode([int(actual_token_id)], skip_special_tokens=False),
        "actual_prob": actual_prob,
        "actual_logprob": actual_logprob,
        "actual_surprisal": float(-actual_logprob),
        "actual_rank": actual_rank,
    }

    max_k = max(max(top_ks), save_top_tokens)
    top_vals, top_ids = torch.topk(probs, k=min(max_k, vocab_size))
    top_vals = top_vals.numpy()
    top_ids = top_ids.numpy()

    for k in top_ks:
        kk = min(int(k), vocab_size)
        vals = top_vals[:kk]
        mass = float(vals.sum())
        if mass > 0:
            q = vals / mass
            top_entropy = float(-(q * np.log(np.maximum(q, 1e-45))).sum())
        else:
            top_entropy = 0.0
        record[f"top{k}_mass"] = mass
        record[f"top{k}_entropy"] = top_entropy
        record[f"top{k}_normalized_entropy"] = top_entropy / math.log(kk) if kk > 1 else 0.0

    if bottom_ks:
        bottom_max_k = min(max(bottom_ks), vocab_size)
        bottom_logits, bottom_ids = torch.topk(logits, k=bottom_max_k, largest=False)
        bottom_probs = probs[bottom_ids].numpy()

        for k in bottom_ks:
            kk = min(int(k), vocab_size)
            vals = bottom_probs[:kk]
            mass = float(vals.sum())
            q = torch.softmax(bottom_logits[:kk], dim=-1).numpy()
            bottom_entropy = float(-(q * np.log(np.maximum(q, 1e-45))).sum())
            record[f"bottom{k}_mass"] = mass
            record[f"bottom{k}_entropy"] = bottom_entropy
            record[f"bottom{k}_normalized_entropy"] = bottom_entropy / math.log(kk) if kk > 1 else 0.0

    if save_top_tokens > 0:
        tops = []
        for prob, token_id in zip(top_vals[:save_top_tokens], top_ids[:save_top_tokens]):
            tops.append({
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "prob": float(prob),
            })
        record["top_tokens"] = tops

    return record


def summarize_object_entropy(records):
    summary = {
        "num_object_records": len(records),
        "num_grounded": int(sum(int(r["label"]) == 1 for r in records)),
        "num_hallucinated": int(sum(int(r["label"]) == 0 for r in records)),
        "metrics": {},
    }
    metrics = [
        "entropy",
        "normalized_entropy",
        "effective_vocab_size",
        "min_entropy",
        "top1_prob",
        "actual_prob",
        "actual_surprisal",
        "actual_rank",
    ]
    subset_metric_names = []
    for record in records:
        for key in record:
            if (
                (key.startswith("top") or key.startswith("bottom"))
                and (key.endswith("_mass") or key.endswith("_entropy"))
            ):
                subset_metric_names.append(key)
    metrics.extend(sorted(set(subset_metric_names)))

    for metric in metrics:
        vals_by_label = {}
        for label, name in [(1, "grounded"), (0, "hallucinated")]:
            vals = [float(r[metric]) for r in records if int(r["label"]) == label and metric in r]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                vals_by_label[name] = {
                    "n": int(len(arr)),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "q10": float(np.quantile(arr, 0.10)),
                    "q50": float(np.quantile(arr, 0.50)),
                    "q90": float(np.quantile(arr, 0.90)),
                }
        if vals_by_label:
            summary["metrics"][metric] = vals_by_label
    return summary


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

    instructions = load_jsonl(args.instruction_path)
    if args.num_samples is not None:
        instructions = instructions[: args.num_samples]

    evaluator = load_chair_evaluator(args.cache, args.annotations_path)
    model_manager = ModelManager(args.model, model_path=args.model_path)

    image_records = []
    step_records = []
    object_records = []

    for image_index, item in tqdm(list(enumerate(instructions)), desc="vocab entropy"):
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

        with torch.inference_mode():
            outputs = model_manager.llm_model.generate(
                input_ids,
                do_sample=False,
                num_beams=args.beam,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs,
            )

        caption = model_manager.decode(outputs["sequences"])[0]
        sequence_ids = [int(x) for x in outputs["sequences"][0].detach().cpu().tolist()]
        num_steps = len(outputs["scores"])
        input_len = int(input_ids.shape[1])
        generated_ids = [int(x) for x in split_generated_ids(sequence_ids, input_len, num_steps)]

        image_record = {
            "image_index": image_index,
            "image_id": image_id,
            "image_filename": coco_image_filename(image_id),
            "image_path": image_path,
            "instruction": instruction,
            "caption": caption,
            "input_len": input_len,
            "num_steps": num_steps,
            "sequence_ids": sequence_ids,
            "generated_ids": generated_ids,
        }
        image_records.append(image_record)

        for step, score in enumerate(outputs["scores"]):
            metrics = entropy_metrics_from_logits(
                score[0],
                generated_ids[step],
                model_manager.tokenizer,
                args.top_k,
                args.bottom_k,
                args.save_top_tokens,
            )
            step_record = {
                "image_index": image_index,
                "image_id": image_id,
                "token_step": step,
                **metrics,
            }
            step_records.append(step_record)

        chair_info = evaluator.compute_chair_token(coco_image_filename(image_id), caption)
        gt_words = list(chair_info["mscoco_gt_words"])
        mentions, _ = object_mentions_from_caption(evaluator, gt_words, caption)

        search_start = 0
        for mention in mentions:
            try:
                alignment = find_generated_token_step(
                    model_manager.tokenizer,
                    mention["surface_word"],
                    sequence_ids,
                    input_len,
                    num_steps,
                    start_step=search_start,
                )
            except Exception:
                try:
                    alignment = find_generated_token_step(
                        model_manager.tokenizer,
                        mention["node_word"],
                        sequence_ids,
                        input_len,
                        num_steps,
                        start_step=search_start,
                    )
                except Exception as exc:
                    print(
                        f"[warn] skip unaligned object entropy: image={image_id} "
                        f"surface={mention['surface_word']!r} node={mention['node_word']!r}: {exc}"
                    )
                    continue

            search_start = int(alignment["step"]) + 1
            step_record = step_records[-num_steps + int(alignment["step"])]
            object_record = {
                "object_index": len(object_records),
                "image_index": image_index,
                "image_id": image_id,
                "caption": caption,
                "surface_word": mention["surface_word"],
                "node_word": mention["node_word"],
                "chair_word_idx": mention["chair_word_idx"],
                "mention_idx": mention["mention_idx"],
                "label": int(mention["label"]),
                "token_step": int(alignment["step"]),
                "matched_text": alignment["matched_text"],
                "matched_token_id": int(alignment["matched_token_id"]),
                "generated_token_id": int(generated_ids[int(alignment["step"])]),
            }
            for key, value in step_record.items():
                if key in {"image_index", "image_id", "token_step", "top_tokens"}:
                    continue
                object_record[key] = value
            object_records.append(object_record)

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "config": vars(args),
        "num_images": len(image_records),
        "num_step_records": len(step_records),
        "num_object_records": len(object_records),
        "object_entropy_summary": summarize_object_entropy(object_records),
    }

    payload = {
        "config": vars(args),
        "image_records": image_records,
        "step_records": step_records,
        "object_records": object_records,
        "summary": summary,
    }

    output_path = output_dir / args.output_file
    torch.save(payload, output_path)
    with open(output_dir / "stage1_vocab_entropy_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_records_csv(step_records, output_dir / "stage1_vocab_entropy_step_records.csv", exclude_keys={"top_tokens"})
    write_records_csv(object_records, output_dir / "stage1_vocab_entropy_object_records.csv", exclude_keys={"caption"})

    print(json.dumps({
        "output": str(output_path),
        "num_images": len(image_records),
        "num_step_records": len(step_records),
        "num_object_records": len(object_records),
        "num_grounded_objects": int(sum(int(r["label"]) == 1 for r in object_records)),
        "num_hallucinated_objects": int(sum(int(r["label"]) == 0 for r in object_records)),
    }, indent=2))


if __name__ == "__main__":
    main()
