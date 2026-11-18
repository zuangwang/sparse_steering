#!/usr/bin/env bash
set -euo pipefail
source launch/environment.sh
python -m reproduce.runner "$@"
