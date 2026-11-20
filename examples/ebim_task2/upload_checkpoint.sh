#!/usr/bin/env bash
# Upload one trained step to the Hugging Face checkpoint repo, mirroring openpi's on-disk layout
# <config>/<exp>/<step>/{params,assets,holdout_eval.json}. Never uploads train_state/.
#
#   examples/ebim_task2/upload_checkpoint.sh <config> <exp> <step> [--repo hermanprawiro/ebim-task2-real] [--dry-run]
#
# Refuses unless params/, assets/ and holdout_eval.json exist (run eval_real_holdout.py first). A missing repo is
# created as a private model repo (an existing repo keeps its visibility; hermanprawiro/ebim-task2-real is public).
# Needs HF_TOKEN (or `hf auth login`) with write access.
# CHECKPOINTS_DIR overrides the checkpoint root (default: <repo>/checkpoints).
set -euo pipefail

repo="hermanprawiro/ebim-task2-real"
dry_run=0
pos=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="${2:?--repo needs a value}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "error: unknown option $1" >&2; exit 2 ;;
    *) pos+=("$1"); shift ;;
  esac
done
if [[ ${#pos[@]} -ne 3 ]]; then
  echo "usage: $0 <config> <exp> <step> [--repo $repo] [--dry-run]" >&2
  exit 2
fi
config="${pos[0]}"; exp="${pos[1]}"; step="${pos[2]}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ckpt="${CHECKPOINTS_DIR:-$root/checkpoints}/$config/$exp/$step"
dest="$config/$exp/$step"

if [[ ! -d "$ckpt" ]]; then
  echo "error: checkpoint dir not found: $ckpt" >&2; exit 1
fi
for sub in params assets; do
  if [[ ! -d "$ckpt/$sub" ]]; then
    echo "error: $ckpt/$sub is missing; refusing to upload an incomplete checkpoint" >&2; exit 1
  fi
done
if [[ -z "$(find "$ckpt/assets" -name norm_stats.json -print -quit)" ]]; then
  echo "error: no norm_stats.json under $ckpt/assets; the server cannot de-normalise without it" >&2; exit 1
fi
if [[ ! -f "$ckpt/holdout_eval.json" ]]; then
  echo "error: $ckpt/holdout_eval.json is missing; run the hold-out eval first:" >&2
  echo "  uv run examples/ebim_task2/eval_real_holdout.py --config-name $config --checkpoint-dir $ckpt" >&2
  exit 1
fi

run() {
  echo "+ $*"
  if [[ $dry_run -eq 0 ]]; then "$@"; fi
}
# The Hugging Face CLI: `hf` (huggingface_hub >= 0.34; the only name in 1.x and later), else the older
# `huggingface-cli` (same arguments), else the one in the repo's uv environment.
if command -v hf >/dev/null; then hf=(hf)
elif command -v huggingface-cli >/dev/null; then hf=(huggingface-cli)
else hf=(uv run huggingface-cli); fi

# --private applies only when the upload creates the repo; an existing repo keeps its visibility.
up=(upload --repo-type model --private "$repo")
run "${hf[@]}" "${up[@]}" "$ckpt/params" "$dest/params"
run "${hf[@]}" "${up[@]}" "$ckpt/assets" "$dest/assets"
run "${hf[@]}" "${up[@]}" "$ckpt/holdout_eval.json" "$dest/holdout_eval.json"

if [[ $dry_run -eq 1 ]]; then verb="Would upload"; else verb="Uploaded"; fi
cat <<MSG

$verb $dest to $repo. To serve it on another machine:
  hf download $repo --include "$dest/*" --local-dir checkpoints
  examples/ebim_task2/serve_munich.sh $config checkpoints/$dest
MSG
