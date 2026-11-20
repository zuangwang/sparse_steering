#!/usr/bin/env bash
# Set up the environment and run one case (model, task, seed) from start to finish.
#
#   bash run_experiment.sh <model> <task> [seed]
#   bash run_experiment.sh gemma truthfulqa 0
#
# Models: gemma qwen llama qwen7 qwen32b      Tasks: truthfulqa halueval paradetox rtp
# Seeds:  0 1 2 3 (default 0)
#
# Settings are environment variables:
#   SPARSE_STEERING_RUN_ROOT  folder for weights, data and results, outside this folder
#                             (default: ~/.local/share/sparse_steering)
#   HF_TOKEN                  Hugging Face token, needed for gated models such as Gemma
#   PYTHON                    interpreter that creates .venv (default: python3.11)
#   CAPABILITY=0              skip the general-capability evaluation
#   DRY_RUN=1                 print the commands without running them
#
# Finished steps are skipped, so an interrupted run can simply be started again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
}
[ $# -ge 2 ] && [ $# -le 3 ] || usage
model=$1 task=$2 seed=${3:-0}
case "$model" in gemma | qwen | llama | qwen7 | qwen32b) ;; *) usage ;; esac
case "$task" in truthfulqa | halueval | paradetox | rtp) ;; *) usage ;; esac
case "$seed" in 0 | 1 | 2 | 3) ;; *) usage ;; esac

dry=${DRY_RUN:-0}
py=.venv/bin/python
run() {
  echo "+ $*"
  [ "$dry" = 1 ] || "$@"
}

echo "== 1/5 Python environment"
[ -x "$py" ] || run "${PYTHON:-python3.11}" -m venv .venv
if [ ! -f .venv/installed ] || [ requirements.txt -nt .venv/installed ]; then
  run "$py" -m pip install -r requirements.txt
  [ "$dry" = 1 ] || touch .venv/installed
fi
echo "+ source launch/environment.sh"
if [ "$dry" != 1 ]; then
  source launch/environment.sh
  "$py" -c 'import sys, torch
ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
sys.exit(0 if ok else "A CUDA GPU with native bf16 support is required.")'
fi

echo "== 2/5 Model weights and task data"
run "$py" -m reproduce.download_models --model "$model"
run "$py" -m reproduce.prepare

# RealToxicityPrompts reuses the ParaDetox direction and mask of the same seed.
fit_task=$task
[ "$task" = rtp ] && fit_task=paradetox
echo "== 3/5 Fit the direction, the comparison methods and the coordinate mask"
run "$py" -m reproduce.fit --seed "$seed" --model "$model" --task "$fit_task"

echo "== 4/5 Evaluate every method on the strength grid"
run "$py" -m reproduce.grid --seed "$seed" --model "$model" --task "$task"

echo "== 5/5 General capability"
if [ "${CAPABILITY:-1}" = 0 ] || [ "$task" = rtp ]; then
  echo "skipped"
else
  [ -f "${DATA_DIR:-}/capability_panels/MANIFEST.json" ] || run "$py" -m reproduce.capability_panels
  run "$py" -m reproduce.capability --seed "$seed" --model "$model" --task "$task"
fi

[ "$dry" = 1 ] && exit 0
result="$OUTPUT_DIR/seed$seed/$model/$task"
echo
echo "Results: $result"
echo "Best feasible setting of each method:"
"$py" - "$result/test.json" <<'EOF'
import json
import sys

data = json.load(open(sys.argv[1]))
print(f"  {'method':28s}{'score':>8s}{'NLL ratio':>11s}")
print(f"  {'unsteered':28s}{data['unsteered']['score']:8.2f}{1.0:11.3f}")
for method, point in data["selected"].items():
    if not point or point.get("score") is None:
        print(f"  {method:28s}{'none':>8s}")
    else:
        print(f"  {method:28s}{point['score']:8.2f}{point['nll_ratio']:11.3f}")
EOF
