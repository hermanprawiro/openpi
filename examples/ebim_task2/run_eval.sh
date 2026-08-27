#!/usr/bin/env bash
# One-command task2 eval: starts the ebim eval_recorder, runs the eval_client
# to completion, then stops the recorder (discarding the trailing episode its
# final reset opened) and writes eval_results/<name>/summary.md.
#
# Both artifact sets share one name:
#   client:   examples/ebim_task2/eval_results/<name>/
#   recorder: $EBIM_ROOT/task2_isaacsim/services/eval_recording/eval_raw_out/<name>/
#
# Prerequisites (HOW_TO_RUN_EVAL.md sections 1-3): policy server, Isaac Sim
# scene with --record, official evaluator container, EBIM_ROOT in .env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME_RE='^[A-Za-z0-9][A-Za-z0-9._-]*$'
RECORDER_READY_TIMEOUT="${RECORDER_READY_TIMEOUT:-180}"
RECORDER_QUIT_TIMEOUT="${RECORDER_QUIT_TIMEOUT:-300}"

usage() {
  cat <<EOF
Usage: run_eval.sh [--eval-name NAME] [--dry-run] [-- CLIENT_ARGS...]

  --eval-name NAME  shared run name (default: eval_<YYYYmmdd_HHMMSS>);
                    single path component: ${NAME_RE}
  --dry-run         print the resolved name, artifact dirs and commands, then exit
  -- CLIENT_ARGS    extra eval_client flags, e.g. -- --num-episodes 1

Environment: EBIM_ROOT (or set it in .env), RECORDER_READY_TIMEOUT,
RECORDER_QUIT_TIMEOUT, EVAL_RAW_OUT (recorder out dir, repo-relative).
EOF
}

NAME="eval_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=0
CLIENT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-name) NAME="${2:?--eval-name needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; CLIENT_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$NAME" =~ $NAME_RE ]]; then
  echo "Invalid --eval-name '$NAME' (must match $NAME_RE)" >&2
  exit 2
fi

# EBIM_ROOT: environment wins, then .env (same precedence as the client's dotenv).
if [[ -z "${EBIM_ROOT:-}" && -f "$SCRIPT_DIR/.env" ]]; then
  EBIM_ROOT="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\+\)\?EBIM_ROOT=//p' "$SCRIPT_DIR/.env" | tail -1 | tr -d "\"'")"
fi
EBIM_ROOT="${EBIM_ROOT/#\~/$HOME}"
if [[ -z "${EBIM_ROOT:-}" ]]; then
  echo "EBIM_ROOT is not set; export it or add it to $SCRIPT_DIR/.env (see .env.example)" >&2
  exit 2
fi
RECORDER_SH="$EBIM_ROOT/task2_isaacsim/scripts/run_eval_recorder.sh"
if [[ ! -x "$RECORDER_SH" ]]; then
  echo "Recorder launcher not found or not executable: $RECORDER_SH (is EBIM_ROOT correct?)" >&2
  exit 2
fi
OUT_REL="${EVAL_RAW_OUT:-services/eval_recording/eval_raw_out}"
if [[ "$OUT_REL" = /* ]]; then
  echo "EVAL_RAW_OUT is an absolute container path; run_eval.sh needs the repo-relative form to find the session on the host" >&2
  exit 2
fi
SESSION_DIR="$EBIM_ROOT/task2_isaacsim/$OUT_REL/$NAME"
RUN_DIR="$SCRIPT_DIR/eval_results/$NAME"

# Fail before the recorder launcher mkdir -p's the session dir.
for dir in "$RUN_DIR" "$SESSION_DIR"; do
  if [[ -e "$dir" ]]; then
    echo "Run dir already exists: $dir -- pick a new --eval-name or remove the old run" >&2
    exit 2
  fi
done

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "eval name:        $NAME"
  echo "client artifacts: $RUN_DIR"
  echo "recorder session: $SESSION_DIR"
  echo "recorder command: $RECORDER_SH record --eval-name $NAME  (output: $SESSION_DIR/recorder.log)"
  echo "client command:   EVAL_ARGS='--run-name $NAME ${CLIENT_ARGS[*]:-}' docker compose -f $SCRIPT_DIR/compose.eval_client.yml run --rm -T eval_client"
  echo "summary command:  python3 $SCRIPT_DIR/summarize_eval.py --session-dir $SESSION_DIR --run-dir $RUN_DIR"
  exit 0
fi

tmpdir="$(mktemp -d)"
touch "$tmpdir/start_stamp"
mkfifo "$tmpdir/recorder_ctl"
# Read-write open never blocks and keeps the fifo open between writes; if the
# wrapper dies uncleanly the resulting EOF is read by the recorder as 'q'.
exec 3<>"$tmpdir/recorder_ctl"

rec_pid=""
stopped=0
stop_recorder() {
  [[ "$stopped" -eq 1 || -z "$rec_pid" ]] && return 0
  stopped=1
  if kill -0 "$rec_pid" 2>/dev/null; then
    # '0' stops+discards the trailing episode the final reset opened (and
    # suppresses evaluate-on-quit); 'q' quits after in-flight evaluator
    # calls are joined and the encoders drained.
    echo "[run_eval] stopping recorder (discard trailing episode, quit)"
    printf '0\n' >&3 || true
    sleep 1
    printf 'q\n' >&3 || true
    local waited=0
    while kill -0 "$rec_pid" 2>/dev/null; do
      if (( waited >= RECORDER_QUIT_TIMEOUT )); then
        echo "[run_eval] WARNING: recorder still running after ${RECORDER_QUIT_TIMEOUT}s; killing it -- check 'docker ps' for a leftover eval_recording container and ${RECORDER_LOG:-its log}" >&2
        kill "$rec_pid" 2>/dev/null || true
        break
      fi
      sleep 1
      waited=$((waited + 1))
    done
  fi
}
cleanup() {
  stop_recorder
  exec 3>&- || true
  rm -rf "$tmpdir"
}
trap cleanup EXIT
trap 'echo "[run_eval] interrupted" >&2; exit 130' INT TERM

# Recorder output goes to a log file inside its own session dir: it would
# otherwise interleave with the client's output (and staircase while the
# client holds the terminal). Pre-create the dir so the redirect can open;
# the pre-flight above ensured it did not exist, and the recorder launcher
# mkdir -p's it anyway.
RECORDER_LOG="$SESSION_DIR/recorder.log"
mkdir -p "$SESSION_DIR"
echo "[run_eval] starting recorder: $NAME"
echo "[run_eval]   (recorder output: $RECORDER_LOG -- tail -f it to watch)"
"$RECORDER_SH" record --eval-name "$NAME" <&3 >"$RECORDER_LOG" 2>&1 &
rec_pid=$!

echo "[run_eval] waiting for recorder to become ready (${RECORDER_READY_TIMEOUT}s timeout)"
waited=0
while ! [[ -f "$SESSION_DIR/.status.json" && "$SESSION_DIR/.status.json" -nt "$tmpdir/start_stamp" ]]; do
  if ! kill -0 "$rec_pid" 2>/dev/null; then
    echo "[run_eval] recorder exited before becoming ready (image built? see 'run_eval_recorder.sh build'); last recorder output:" >&2
    tail -n 20 "$RECORDER_LOG" >&2 || true
    exit 1
  fi
  if (( waited >= RECORDER_READY_TIMEOUT )); then
    echo "[run_eval] recorder not ready after ${RECORDER_READY_TIMEOUT}s; last recorder output:" >&2
    tail -n 20 "$RECORDER_LOG" >&2 || true
    exit 1
  fi
  sleep 2
  waited=$((waited + 2))
done

echo "[run_eval] recorder ready; starting eval client"
set +e
# -T: no pseudo-TTY. The client never reads its TTY, and an attached TTY
# would put this terminal into raw mode; Ctrl-C is instead proxied as
# SIGINT, which the client handles gracefully ("Interrupted; stopping robot").
EVAL_ARGS="--run-name $NAME${CLIENT_ARGS[*]:+ ${CLIENT_ARGS[*]}}" \
  docker compose -f "$SCRIPT_DIR/compose.eval_client.yml" run --rm -T eval_client
client_rc=$?
set -e
echo "[run_eval] eval client exited with code $client_rc"

# Stop the recorder before summarizing so manifests and evaluator outputs are final.
stop_recorder

if [[ -d "$RUN_DIR" ]]; then
  echo "[run_eval] writing summary"
  python3 "$SCRIPT_DIR/summarize_eval.py" --session-dir "$SESSION_DIR" --run-dir "$RUN_DIR" \
    || echo "[run_eval] WARNING: summary generation failed; re-run summarize_eval.py manually" >&2
fi

exit "$client_rc"
