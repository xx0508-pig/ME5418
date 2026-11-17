#!/usr/bin/env bash
# Formal conventional-controller baseline evaluation.
# Protocol: 2 controller variants x 5 friction conditions x 200 episodes,
# environment seeds 10000-10199, no video.
set -euo pipefail

PROJECT="/home/saga/Documents/Codex/2026-10-01/files-mentioned-by-the-user-proposal/outputs/ME5418"
SUITE="$PROJECT/outputs/evaluation_suites/formal_baseline_n200_envseeds_10000_10199"
ROOT="$PROJECT/outputs/runs/formal_baseline_n200_envseeds_10000_10199"
LOG_DIR="$SUITE/logs"
PY="$PROJECT/scripts/local.sh"

mkdir -p "$ROOT" "$LOG_DIR"

run_one() {
  local variant="$1"
  local condition="$2"
  shift 2
  local run_name="${variant}_${condition}"
  local run_dir="$ROOT/$run_name"
  local log_path="$LOG_DIR/${run_name}.log"

  if [[ -e "$run_dir" ]]; then
    echo "[$(date '+%F %T')] REFUSE existing run directory: $run_dir" >&2
    exit 1
  fi

  echo "[$(date '+%F %T')] START $run_name"
  /usr/bin/bash "$PY" -m scripts.evaluate_feedback_baseline \
    --controller-variant "$variant" \
    --output-root "outputs/runs/formal_baseline_n200_envseeds_10000_10199" \
    --run-name "$run_name" \
    --episodes 200 \
    --video-episodes 0 \
    --seed 10000 \
    "$@" \
    --velocity-kp 0.20 \
    --velocity-ki 0.05 \
    --max-object-speed 0.06 \
    --slowdown-distance 0.14 \
    > "$log_path" 2>&1

  if [[ ! -f "$run_dir/summary.json" ]]; then
    echo "[$(date '+%F %T')] MISSING summary: $run_dir/summary.json" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] DONE $run_name"
}

run_one pi_no_release random_mu_0p3_0p9 --friction-min 0.3 --friction-max 0.9
run_one pi_no_release fixed_mu_0p3 --friction 0.3
run_one pi_no_release fixed_mu_0p6 --friction 0.6
run_one pi_no_release fixed_mu_0p9 --friction 0.9
run_one pi_no_release fixed_mu_1 --friction 1.0

run_one hybrid_pi_settle random_mu_0p3_0p9 --friction-min 0.3 --friction-max 0.9
run_one hybrid_pi_settle fixed_mu_0p3 --friction 0.3
run_one hybrid_pi_settle fixed_mu_0p6 --friction 0.6
run_one hybrid_pi_settle fixed_mu_0p9 --friction 0.9
run_one hybrid_pi_settle fixed_mu_1 --friction 1.0

echo "[$(date '+%F %T')] ALL_10_JOBS_DONE"
