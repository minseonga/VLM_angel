#!/usr/bin/env bash
set -euo pipefail

: "${LLAVA_MODEL_PATH:?Set LLAVA_MODEL_PATH to the LLaVA-1.5 checkpoint path or HF id.}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
INSTRUCTION_PATH="${INSTRUCTION_PATH:-examples/toy_img_query_list.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-stage1_outputs}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
OVERLAP_MODE="${OVERLAP_MODE:-iqr}"
MAX_RECORDS="${MAX_RECORDS:-}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
TOKEN_STEP_BINS="${TOKEN_STEP_BINS:-4}"
SVAR_MATCH_THRESHOLD="${SVAR_MATCH_THRESHOLD:-0.05}"
TOKEN_STEP_MATCH_THRESHOLD="${TOKEN_STEP_MATCH_THRESHOLD:-}"

COMMON_ARGS=(
  --model llava-1.5
  --model-path "${LLAVA_MODEL_PATH}"
  --data-path "${IMAGE_FOLDER}"
)

python scripts/stage1_extract_svar_object_traces.py \
  "${COMMON_ARGS[@]}" \
  --annotations-path "${ANNOTATION_DIR}" \
  --instruction-path "${INSTRUCTION_PATH}" \
  --num-samples "${NUM_SAMPLES}" \
  --output-dir "${OUTPUT_DIR}"

TRACE_ARGS=(
  "${COMMON_ARGS[@]}"
  --trace-file "${OUTPUT_DIR}/stage1_svar_object_traces.pt"
  --overlap-mode "${OVERLAP_MODE}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${MAX_RECORDS}" ]]; then
  TRACE_ARGS+=(--max-records "${MAX_RECORDS}")
fi

python scripts/stage1_trace_head_logit_contrib.py "${TRACE_ARGS[@]}"

python scripts/stage1_analyze_head_contrib.py \
  --contrib-file "${OUTPUT_DIR}/stage1_head_logit_contrib.pt" \
  --output-dir "${OUTPUT_DIR}"

CONTROL_ARGS=(
  --contrib-file "${OUTPUT_DIR}/stage1_head_logit_contrib.pt"
  --output-dir "${OUTPUT_DIR}"
  --bootstrap-iters "${BOOTSTRAP_ITERS}"
  --token-step-bins "${TOKEN_STEP_BINS}"
  --svar-match-threshold "${SVAR_MATCH_THRESHOLD}"
)

if [[ -n "${TOKEN_STEP_MATCH_THRESHOLD}" ]]; then
  CONTROL_ARGS+=(--token-step-match-threshold "${TOKEN_STEP_MATCH_THRESHOLD}")
fi

python scripts/stage1_controlled_analysis.py "${CONTROL_ARGS[@]}"
