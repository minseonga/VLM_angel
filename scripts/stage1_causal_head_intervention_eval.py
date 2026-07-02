#!/usr/bin/env python
"""
Causal head intervention screen for Stage 1D.

Runs open-ended caption generation with selected-head interventions and evaluates
CHAIR on the generated captions. Use this for quick go/no-go tests of whether
heads identified in step-conditioned analysis have practical causal effect.
"""

import argparse
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
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_INSTRUCTION_PATH,
    coco_image_path,
    load_chair_evaluator,
    load_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate selected-head causal interventions with CHAIR.")
    parser.add_argument("--model", type=str, default="llava-1.5")
    parser.add_argument("--model-path", type=str, default=os.environ.get("LLAVA_MODEL_PATH"))
    parser.add_argument("--data-path", type=str, default=os.environ.get("IMAGE_FOLDER", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--annotations-path", type=str, default=os.environ.get("ANNOTATION_DIR", DEFAULT_ANNOTATION_DIR))
    parser.add_argument("--instruction-path", type=str, default=DEFAULT_INSTRUCTION_PATH)
    parser.add_argument("--cache", type=str, default="chair.pkl")
    parser.add_argument("--output-dir", type=str, default="stage1_causal_head_eval")
    parser.add_argument("--variant-name", type=str, default=None)
    parser.add_argument("--heads-json", type=str, default=None)
    parser.add_argument(
        "--head-set",
        choices=["support_heads", "prior_heads", "random_heads"],
        default="support_heads",
    )
    parser.add_argument(
        "--mode",
        choices=["none", "visual_boost", "visual_suppress", "head_scale"],
        default="none",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Attention-logit shift for visual boost/suppress.")
    parser.add_argument("--head-scale", type=float, default=0.0, help="Scale for selected head outputs in head_scale mode.")
    parser.add_argument("--min-new-token-step", type=int, default=0)
    parser.add_argument("--max-new-token-step", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def load_selected_heads(path, head_set):
    if not path:
        return []
    with open(path) as f:
        payload = json.load(f)
    heads = payload.get(head_set, [])
    out = []
    seen = set()
    for item in heads:
        layer = int(item["layer"])
        head = int(item["head"])
        if (layer, head) in seen:
            continue
        seen.add((layer, head))
        out.append({"layer": layer, "head": head})
    return out


def group_heads_by_layer(heads):
    grouped = {}
    for item in heads:
        grouped.setdefault(int(item["layer"]), []).append(int(item["head"]))
    return {layer: sorted(set(heads)) for layer, heads in grouped.items()}


def step_is_active(kv_seq_len, prompt_len, min_step, max_step):
    current_step = int(kv_seq_len) - int(prompt_len)
    if current_step < int(min_step):
        return False
    if max_step is not None and current_step > int(max_step):
        return False
    return True


def patched_llama_forward_factory(layer_idx, config):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    selected_heads = config["heads_by_layer"].get(layer_idx, [])

    def patched_llama_forward(
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

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

        active = bool(selected_heads) and step_is_active(
            kv_seq_len,
            config["prompt_len"],
            config["min_new_token_step"],
            config["max_new_token_step"],
        )
        if active and config["mode"] in {"visual_boost", "visual_suppress"}:
            head_idx = torch.tensor(selected_heads, dtype=torch.long, device=attn_weights.device)
            sign = 1.0 if config["mode"] == "visual_boost" else -1.0
            attn_weights[:, head_idx, -1, config["img_start_idx"]: config["img_end_idx"]] = (
                attn_weights[:, head_idx, -1, config["img_start_idx"]: config["img_end_idx"]]
                + sign * float(config["alpha"])
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        head_outputs = torch.matmul(attn_weights, value_states)

        if active and config["mode"] == "head_scale":
            head_idx = torch.tensor(selected_heads, dtype=torch.long, device=head_outputs.device)
            head_outputs[:, head_idx, -1, :] = head_outputs[:, head_idx, -1, :] * float(config["head_scale"])

        attn_output = head_outputs.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return patched_llama_forward


def install_head_intervention(model, heads, mode, alpha, head_scale, img_start_idx, img_end_idx, prompt_len, min_step, max_step):
    if mode == "none" or not heads:
        return lambda: None

    heads_by_layer = group_heads_by_layer(heads)
    originals = []
    config = {
        "heads_by_layer": heads_by_layer,
        "mode": mode,
        "alpha": alpha,
        "head_scale": head_scale,
        "img_start_idx": int(img_start_idx),
        "img_end_idx": int(img_end_idx),
        "prompt_len": int(prompt_len),
        "min_new_token_step": int(min_step),
        "max_new_token_step": None if max_step is None else int(max_step),
    }

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx not in heads_by_layer:
            continue
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        attn.forward = types.MethodType(patched_llama_forward_factory(layer_idx, config), attn)

    def restore():
        for attn, original in originals:
            attn.forward = original

    return restore


def variant_name(args):
    if args.variant_name:
        return args.variant_name
    if args.mode == "none":
        return "baseline"
    step_part = f"steps{args.min_new_token_step}"
    if args.max_new_token_step is not None:
        step_part += f"-{args.max_new_token_step}"
    else:
        step_part += "plus"
    return f"{args.head_set}_{args.mode}_{step_part}"


def write_jsonl(records, path):
    with open(path, "w") as f:
        for record in records:
            json.dump(record, f)
            f.write("\n")


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
    name = variant_name(args)

    instructions = load_jsonl(args.instruction_path)
    if args.num_samples is not None:
        instructions = instructions[: args.num_samples]

    selected_heads = load_selected_heads(args.heads_json, args.head_set)
    if args.mode != "none" and not selected_heads:
        raise ValueError(f"No heads loaded for {args.head_set}; pass --heads-json or use --mode none.")

    evaluator = load_chair_evaluator(args.cache, args.annotations_path)
    model_manager = ModelManager(args.model, model_path=args.model_path)

    caption_records = []
    caption_path = output_dir / f"{name}_captions.jsonl"

    for idx, item in tqdm(list(enumerate(instructions)), desc=name):
        image_id = int(item["image_id"])
        instruction = item.get("instruction", "Please help me describe the image in detail.")
        image = Image.open(coco_image_path(args.data_path, image_id)).convert("RGB")
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

        restore = install_head_intervention(
            model_manager.llm_model,
            selected_heads,
            args.mode,
            args.alpha,
            args.head_scale,
            model_manager.img_start_idx,
            model_manager.img_end_idx,
            input_ids.shape[1],
            args.min_new_token_step,
            args.max_new_token_step,
        )
        try:
            with torch.inference_mode():
                output_ids = model_manager.llm_model.generate(
                    input_ids,
                    do_sample=args.sample,
                    num_beams=args.beam,
                    max_new_tokens=args.max_tokens,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict_in_generate=False,
                    **kwargs,
                )
        finally:
            restore()

        caption = model_manager.decode(output_ids)[0]
        caption_records.append({
            "image_id": image_id,
            "caption": caption,
            "variant": name,
            "mode": args.mode,
            "head_set": args.head_set if args.mode != "none" else "",
        })

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            write_jsonl(caption_records, caption_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(caption_records, caption_path)

    chair_output = evaluator.compute_chair(str(caption_path), "image_id", "caption")
    detail_path = output_dir / f"{name}_chair_details.json"
    summary_path = output_dir / f"{name}_summary.json"
    with open(detail_path, "w") as f:
        json.dump(chair_output, f, indent=2)

    summary = {
        "variant": name,
        "config": vars(args),
        "num_captions": len(caption_records),
        "num_heads": len(selected_heads) if args.mode != "none" else 0,
        "selected_heads": selected_heads,
        "overall_metrics": chair_output["overall_metrics"],
        "outputs": {
            "captions": str(caption_path),
            "chair_details": str(detail_path),
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({
        "variant": name,
        "num_captions": len(caption_records),
        "num_heads": summary["num_heads"],
        "overall_metrics": summary["overall_metrics"],
        "outputs": {
            "summary": str(summary_path),
            "captions": str(caption_path),
            "chair_details": str(detail_path),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
