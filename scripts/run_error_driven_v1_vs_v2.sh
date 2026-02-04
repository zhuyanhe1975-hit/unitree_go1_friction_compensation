#!/usr/bin/env bash
set -euo pipefail

# Error-driven config: higher freq + softer gains
AMP=0.2
FREQ=1.0
DUR=20
KP=0.5
KD=0.005
TAU_FF_LIMIT=0.15
TAU_FF_SCALE=1.0
FF_UPDATE_DIV=1

DS_V1="runs/torque_delta_dataset_cov_v1.npz"
M_V1="runs/torque_delta_model_cov_v1.pt"
DS_V2="runs/torque_delta_dataset_cov_v2.npz"
M_V2="runs/torque_delta_model_cov_v2.pt"

echo "=== real_log source ==="
PYTHONPATH=. python3 scripts/inspect_real_log_source.py --npz runs/real_log.npz || true
echo

echo "=== RUN v1 (error-driven) ==="
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both --ff_type torque_delta \
  --amp "$AMP" --freq "$FREQ" --duration "$DUR" \
  --kp "$KP" --kd "$KD" \
  --tau_ff_limit "$TAU_FF_LIMIT" --tau_ff_scale "$TAU_FF_SCALE" \
  --ff_update_div "$FF_UPDATE_DIV" \
  --torque_delta_dataset "$DS_V1" \
  --torque_delta_model "$M_V1"

echo
echo "=== RUN v2 (error-driven) ==="
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both --ff_type torque_delta \
  --amp "$AMP" --freq "$FREQ" --duration "$DUR" \
  --kp "$KP" --kd "$KD" \
  --tau_ff_limit "$TAU_FF_LIMIT" --tau_ff_scale "$TAU_FF_SCALE" \
  --ff_update_div "$FF_UPDATE_DIV" \
  --torque_delta_dataset "$DS_V2" \
  --torque_delta_model "$M_V2"

echo
echo "=== SUMMARY (torque_delta only) ==="
PYTHONPATH=. python3 scripts/summarize_ff_demo_batch.py \
  --glob 'runs/ff_demo_report_*.md' \
  --window_s 0.5 \
  --reversal_signal auto \
  --align_s 0.25
