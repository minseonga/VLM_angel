#!/usr/bin/env python
"""
Extract object-token SVAR traces with enough metadata to replay overlap cases.

This is Stage 1 / D1:
  - generate captions
  - label generated COCO object mentions with CHAIR
  - compute per-layer/per-head visual attention mass for each object token step
  - identify SVAR overlap intervals between grounded and hallucinated mentions
"""

import argparse
from collections import Counter
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
    annotate_overlap_flags,
    coco_image_filename,
    coco_image_path,
    compute_overlap,
    find_generated_token_step,
    load_chair_evaluator,
    load_jsonl,
    object_mentions_from_caption,
    split_generated_ids,
    svar_band_score,
    visual_attention_matrix_for_step,
    write_records_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 D1: extract SVAR overlap traces.")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--annotations-path", type=str, default=os.environ.get("ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR))
    parser.add_argument("--instruction-path", type=str, default=DEFAULT_INSTRUCTION_PATH)
    parser.add_argument("--cache", type=str, default="chair.pkl")
    parser.add_argument("--output-dir", type=str, default="stage1_outputs")
    parser.add_argument("--output-file", type=str, default="stage1_svar_object_traces.pt")
    parser.add_argument("--captions-file", type=str, default="stage1_generated_captions.jsonl")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def characterize_overlap(records):
    out = {}
    for mode in ["iqr", "q10_q90"]:
        key = f"overlap_{mode}"
        subset = [record for record in records if bool(record.get(key))]
        mode_summary = {
            "count_total": len(subset),
            "count_grounded": int(sum(int(r["label"]) == 1 for r in subset)),
            "count_hallucinated": int(sum(int(r["label"]) == 0 for r in subset)),
            "top_grounded_objects": [],
            "top_hallucinated_objects": [],
            "svar": {},
            "token_step": {},
        }
        for label, name in [(1, "grounded"), (0, "hallucinated")]:
            label_subset = [r for r in subset if int(r["label"]) == label]
            counter = Counter(r["node_word"] for r in label_subset)
            top_key = f"top_{name}_objects"
            mode_summary[top_key] = [
                {"node_word": node_word, "count": int(count)}
                for node_word, count in counter.most_common(20)
            ]
            if label_subset:
                svar = np.array([float(r["svar"]) for r in label_subset], dtype=np.float64)
                steps = np.array([int(r["token_step"]) for r in label_subset], dtype=np.float64)
                mode_summary["svar"][name] = {
                    "mean": float(svar.mean()),
                    "std": float(svar.std()),
                    "q10": float(np.quantile(svar, 0.10)),
                    "q50": float(np.quantile(svar, 0.50)),
                    "q90": float(np.quantile(svar, 0.90)),
                }
                mode_summary["token_step"][name] = {
                    "mean": float(steps.mean()),
                    "q50": float(np.quantile(steps, 0.50)),
                }
        out[mode] = mode_summary
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

    instructions = load_jsonl(args.instruction_path)
    if args.num_samples is not None:
        instructions = instructions[: args.num_samples]

    evaluator = load_chair_evaluator(args.cache, args.annotations_path)
    model_manager = ModelManager(args.model, model_path=args.model_path)

    image_records = []
    object_records = []
    captions_path = output_dir / args.captions_file

    with open(captions_path, "w") as captions_f:
        for image_index, item in tqdm(list(enumerate(instructions)), desc="SVAR traces"):
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
                    output_attentions=True,
                    return_dict_in_generate=True,
                    **kwargs,
                )

            caption = model_manager.decode(outputs["sequences"])[0]
            sequence_ids = [int(x) for x in outputs["sequences"][0].detach().cpu().tolist()]
            num_steps = len(outputs["attentions"])
            input_len = int(input_ids.shape[1])
            generated_ids = [int(x) for x in split_generated_ids(sequence_ids, input_len, num_steps)]

            image_filename = coco_image_filename(image_id)
            chair_info = evaluator.compute_chair_token(image_filename, caption)
            gt_words = list(chair_info["mscoco_gt_words"])
            mentions, raw_words = object_mentions_from_caption(evaluator, gt_words, caption)

            image_record = {
                "image_index": image_index,
                "image_id": image_id,
                "image_filename": image_filename,
                "image_path": image_path,
                "instruction": instruction,
                "caption": caption,
                "input_len": input_len,
                "num_steps": num_steps,
                "sequence_ids": sequence_ids,
                "generated_ids": generated_ids,
                "gt_words": gt_words,
                "raw_words": raw_words,
            }
            image_records.append(image_record)
            captions_f.write(json.dumps({"image_id": image_id, "caption": caption}) + "\n")
            captions_f.flush()

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
                            f"[warn] skip unaligned object: image={image_id} "
                            f"surface={mention['surface_word']!r} node={mention['node_word']!r}: {exc}"
                        )
                        continue

                search_start = int(alignment["step"]) + 1
                var_matrix = visual_attention_matrix_for_step(
                    outputs,
                    int(alignment["step"]),
                    model_manager.img_start_idx,
                    model_manager.img_end_idx,
                )
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
                    "svar": svar_band_score(var_matrix, args.start_layer, args.end_layer),
                    "var_matrix": var_matrix.astype(np.float32),
                }
                object_records.append(object_record)

            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overlap_summary = compute_overlap(object_records, score_key="svar")
    annotate_overlap_flags(object_records, overlap_summary, score_key="svar")
    overlap_characterization = characterize_overlap(object_records)

    payload = {
        "config": vars(args),
        "image_records": image_records,
        "object_records": object_records,
        "overlap_summary": overlap_summary,
        "overlap_characterization": overlap_characterization,
    }

    trace_path = output_dir / args.output_file
    torch.save(payload, trace_path)

    with open(output_dir / "stage1_svar_overlap_summary.json", "w") as f:
        json.dump({
            "overlap_summary": overlap_summary,
            "overlap_characterization": overlap_characterization,
        }, f, indent=2)
    write_records_csv(
        object_records,
        output_dir / "stage1_svar_object_records.csv",
        exclude_keys={"var_matrix", "caption"},
    )

    print(f"Saved trace: {trace_path}")
    print(f"Saved captions: {captions_path}")
    print(json.dumps({
        "overlap_summary": overlap_summary,
        "overlap_characterization": overlap_characterization,
    }, indent=2))


if __name__ == "__main__":
    main()
