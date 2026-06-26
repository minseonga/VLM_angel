#!/usr/bin/env python
"""
Replay SVAR-overlap object steps and capture per-head direct logit contribution.

This is Stage 1 / D2:
  - read D1 object traces
  - select overlap records
  - replay the caption prefix before each object token
  - capture per-layer/per-head direct contribution to target object-token logits
"""

import argparse
import csv
import json
import math
import os
import types
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from stage1_common import (
    DEFAULT_IMAGE_ROOT,
    coco_image_path,
    first_token_id,
    token_id_candidates,
    torch_load_compat,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 D2: trace per-head logit contributions.")
    parser.add_argument("--trace-file", type=str, default="stage1_outputs/stage1_svar_object_traces.pt")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output-dir", type=str, default="stage1_outputs")
    parser.add_argument("--output-file", type=str, default="stage1_head_logit_contrib.pt")
    parser.add_argument("--overlap-mode", choices=["iqr", "q10_q90", "all"], default="iqr")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--start-layer", type=int, default=5)
    parser.add_argument("--end-layer", type=int, default=18)
    return parser.parse_args()


def traced_llama_forward_factory(layer_idx, trace_store, target_token_ids, lm_head_weight):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    def traced_llama_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError("LlamaAttention layer_idx is required when using cache.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be {(bsz, 1, q_len, kv_seq_len)}, got {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device),
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        head_outputs = torch.matmul(attn_weights, value_states)

        with torch.no_grad():
            last_head_outputs = head_outputs[0, :, -1, :].float()
            o_weight = self.o_proj.weight.detach().float()
            o_weight_by_head = o_weight.reshape(o_weight.shape[0], self.num_heads, self.head_dim)
            per_head_hidden = torch.einsum("hd,mhd->hm", last_head_outputs, o_weight_by_head)

            target_ids = torch.tensor(target_token_ids, dtype=torch.long, device=hidden_states.device)
            target_rows = lm_head_weight.detach().float().to(hidden_states.device).index_select(0, target_ids)
            contrib = torch.matmul(per_head_hidden, target_rows.t())
            trace_store[layer_idx] = contrib.detach().cpu()

        attn_output = head_outputs.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return traced_llama_forward


def install_head_logit_tracer(model, target_token_ids):
    trace_store = {}
    originals = []
    lm_head_weight = model.lm_head.weight

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        attn.forward = types.MethodType(
            traced_llama_forward_factory(layer_idx, trace_store, target_token_ids, lm_head_weight),
            attn,
        )

    def restore():
        for attn, original_forward in originals:
            attn.forward = original_forward

    return trace_store, restore


def add_target(targets, names, token_id, name):
    if token_id is None:
        return
    token_id = int(token_id)
    if token_id in targets:
        return
    targets.append(token_id)
    names.append(name)


def build_targets(tokenizer, object_record, image_record):
    targets = []
    names = []

    add_target(targets, names, object_record["generated_token_id"], "actual")

    for field_name, target_name in [
        ("matched_token_id", "matched"),
        ("surface_word", "surface"),
        ("node_word", "node"),
    ]:
        value = object_record.get(field_name)
        if isinstance(value, int):
            token_id = value
        else:
            token_id = first_token_id(tokenizer, value)
        add_target(targets, names, token_id, target_name)

    for candidate in token_id_candidates(tokenizer, object_record.get("node_word", "")):
        add_target(targets, names, candidate["token_id"], f"node_variant:{candidate['variant']}")

    for gt_word in image_record.get("gt_words", []):
        for candidate in token_id_candidates(tokenizer, gt_word):
            add_target(targets, names, candidate["token_id"], f"gt:{gt_word}")
            break

    return targets, names


def selected_records(payload, overlap_mode):
    records = list(payload["object_records"])
    if overlap_mode == "all":
        return records
    key = f"overlap_{overlap_mode}"
    return [record for record in records if bool(record.get(key))]


def write_contrib_csv(records, path, start_layer, end_layer):
    fields = [
        "object_index",
        "image_id",
        "label",
        "surface_word",
        "node_word",
        "token_step",
        "svar",
        "target_count",
        "actual_band_mean",
        "actual_band_std",
        "actual_band_min",
        "actual_band_max",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            contrib = record["head_logit_contrib"]
            actual = contrib[start_layer: end_layer + 1, :, 0]
            writer.writerow({
                "object_index": record["object_index"],
                "image_id": record["image_id"],
                "label": record["label"],
                "surface_word": record["surface_word"],
                "node_word": record["node_word"],
                "token_step": record["token_step"],
                "svar": record["svar"],
                "target_count": len(record["target_token_ids"]),
                "actual_band_mean": float(actual.mean()),
                "actual_band_std": float(actual.std()),
                "actual_band_min": float(actual.min()),
                "actual_band_max": float(actual.max()),
            })


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

    payload = torch_load_compat(args.trace_file, map_location="cpu")
    image_records = {int(r["image_index"]): r for r in payload["image_records"]}
    objects = selected_records(payload, args.overlap_mode)
    objects = sorted(objects, key=lambda r: (int(r["image_index"]), int(r["token_step"]), int(r["object_index"])))
    if args.max_records is not None:
        objects = objects[: args.max_records]

    model_manager = ModelManager(args.model, model_path=args.model_path)
    contrib_records = []

    for object_record in tqdm(objects, desc="head logit replay"):
        image_record = image_records[int(object_record["image_index"])]
        image_id = int(image_record["image_id"])
        image_path = coco_image_path(args.data_path, image_id)
        image = Image.open(image_path).convert("RGB")
        images_tensor = process_images(
            [image],
            model_manager.image_processor,
            model_manager.llm_model.config,
        ).to(model_manager.llm_model.device, dtype=torch.float16)

        _, input_ids, kwargs = model_manager.prepare_inputs_for_model(
            [image_record["instruction"]],
            images_tensor,
            use_dataloader=False,
        )

        step = int(object_record["token_step"])
        generated_ids = [int(x) for x in image_record["generated_ids"]]
        if step >= len(generated_ids):
            print(f"[warn] skip object {object_record['object_index']}: step exceeds generated ids")
            continue

        if step > 0:
            prefix_tokens = torch.tensor(
                [generated_ids[:step]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            replay_input_ids = torch.cat([input_ids, prefix_tokens], dim=1)
        else:
            replay_input_ids = input_ids

        target_token_ids, target_names = build_targets(model_manager.tokenizer, object_record, image_record)
        if not target_token_ids:
            print(f"[warn] skip object {object_record['object_index']}: no target ids")
            continue

        trace_store, restore = install_head_logit_tracer(model_manager.llm_model, target_token_ids)
        try:
            with torch.inference_mode():
                _ = model_manager.llm_model(
                    input_ids=replay_input_ids,
                    images=kwargs["images"],
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
        finally:
            restore()

        if len(trace_store) != model_manager.llm_model.config.num_hidden_layers:
            print(
                f"[warn] object {object_record['object_index']}: captured "
                f"{len(trace_store)} layers"
            )

        matrices = []
        for layer_idx in range(model_manager.llm_model.config.num_hidden_layers):
            if layer_idx not in trace_store:
                raise RuntimeError(f"missing traced layer {layer_idx}")
            matrices.append(trace_store[layer_idx])
        head_logit_contrib = torch.stack(matrices, dim=0).to(torch.float32)

        contrib_record = {
            "object_index": int(object_record["object_index"]),
            "image_index": int(object_record["image_index"]),
            "image_id": image_id,
            "caption": image_record["caption"],
            "surface_word": object_record["surface_word"],
            "node_word": object_record["node_word"],
            "label": int(object_record["label"]),
            "token_step": step,
            "svar": float(object_record["svar"]),
            "target_token_ids": target_token_ids,
            "target_names": target_names,
            "head_logit_contrib": head_logit_contrib,
            "gt_words": image_record.get("gt_words", []),
            "generated_token_id": int(object_record["generated_token_id"]),
        }
        contrib_records.append(contrib_record)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_payload = {
        "config": vars(args),
        "source_trace_file": args.trace_file,
        "overlap_mode": args.overlap_mode,
        "records": contrib_records,
    }

    output_path = output_dir / args.output_file
    torch.save(output_payload, output_path)
    write_contrib_csv(contrib_records, output_dir / "stage1_head_logit_contrib_records.csv", args.start_layer, args.end_layer)

    with open(output_dir / "stage1_head_logit_contrib_summary.json", "w") as f:
        json.dump({
            "source_trace_file": args.trace_file,
            "overlap_mode": args.overlap_mode,
            "num_records": len(contrib_records),
            "num_grounded": int(sum(r["label"] == 1 for r in contrib_records)),
            "num_hallucinated": int(sum(r["label"] == 0 for r in contrib_records)),
        }, f, indent=2)

    print(f"Saved head logit contribution trace: {output_path}")
    print(f"records={len(contrib_records)} grounded={sum(r['label'] == 1 for r in contrib_records)} hallucinated={sum(r['label'] == 0 for r in contrib_records)}")


if __name__ == "__main__":
    main()
