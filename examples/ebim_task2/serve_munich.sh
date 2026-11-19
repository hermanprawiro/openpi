#!/usr/bin/env bash
# Serve a trained Munich checkpoint through Docker.
#
#   examples/ebim_task2/serve_munich.sh <config> <ckpt_dir> [port]
#
#   <config>    training config name, e.g. pi05_ebim_task2_munich_real
#   <ckpt_dir>  checkpoint step dir containing params/ and assets/
#   [port]      default 8000
#
# Extra args are passed to `docker compose up` via COMPOSE_UP_ARGS (default: --build).
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
fi

config="$1"
ckpt_dir="$2"
port="${3:-8000}"

if [[ ! -d "$ckpt_dir" ]]; then
  echo "error: checkpoint dir not found: $ckpt_dir" >&2
  exit 1
fi
ckpt_dir="$(cd "$ckpt_dir" && pwd -P)"
for sub in params assets; do
  if [[ ! -d "$ckpt_dir/$sub" ]]; then
    echo "error: $ckpt_dir/$sub is missing (point at the step dir, e.g. .../<exp>/30000)" >&2
    exit 1
  fi
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# The checkpoint id the server reports to clients: <exp>/<step>, the last two components of the step dir.
ckpt_name="$(basename "$(dirname "$ckpt_dir")")/$(basename "$ckpt_dir")"
export POLICY_CONFIG="$config" CKPT_DIR="$ckpt_dir" CKPT_NAME="$ckpt_name" PORT="$port"
echo "serving config=$config ckpt=$ckpt_dir (id $ckpt_name) port=$port"
# shellcheck disable=SC2086
exec docker compose -f "$here/compose.policy_server.yml" up ${COMPOSE_UP_ARGS:---build}
