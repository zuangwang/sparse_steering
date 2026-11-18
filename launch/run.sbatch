#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --gpus=1
set -euo pipefail
source launch/environment.sh
python -m reproduce.runner "$@"
