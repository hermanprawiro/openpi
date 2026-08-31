# EBiM Task 2 — Autonomous Policy Eval Client

Closes the loop between an openpi policy server (pi0/pi0.5 fine-tuned on
task2, see [README.md](README.md) and [POLICY_SERVER.md](POLICY_SERVER.md))
and the Isaac Sim benchmark scene (`ebim/benchmark/task2_isaacsim`). Fully
autonomous: runs N episodes, requesting a scene reset between episodes, with
a sim-time episode timeout and LeRobot-style **async inference** (real-time
chunking) as the primary mode.

Videos and official IoU scoring are **not** done here — run the benchmark's
`eval_recorder` service alongside. It records per-episode videos and
auto-triggers the official evaluation whenever this client publishes a
scene-reset request (i.e. at each episode boundary, plus one `final_reset`
after the last episode). Nothing uploads to Hugging Face.

## Architecture

```
eval_client/
├── constants.py   contract constants + topic names (transcribed from
│                  task2_isaacsim/config/topics.yaml — mirror changes!)
├── config.py      EvalConfig: defaults < eval_config.yaml < CLI flags
├── obs.py         joint-name fallbacks, 20-dim state, image decode, obs dict
├── actions.py     pedal tokens, gripper rad, 20-dim action split
├── chunking.py    ChunkExecutor (sync) + AsyncChunkBuffer + merge strategies
├── inference.py   reconnecting PolicyClient + InferenceWorker thread
├── ros_node.py    the only ROS module: obs cache + command publishers
├── episode.py     guards, 30 Hz sim-time grid, reset handshake, rollouts
├── results.py     result.jsonl + config.yaml per run
└── main.py        episode loop orchestration
```

All modules except `ros_node.py`/`main.py` are ROS-free and unit-tested:

```bash
uv run pytest examples/ebim_task2/eval_client/tests   # from the openpi root
```

## Prerequisites

1. **Isaac Sim scene** (benchmark repo, on the sim machine) — must be
   launched with `--record` (publishes `/isaac/joint_states_full`, odom and
   the cameras) and **without** `--arm-keyboard-teleop`:

   ```bash
   ./task2_isaacsim/scripts/run_isaacsim_teleop.sh --scene room --no-browser \
       --no-republisher --controller-mode none -- --record --randomize-objects
   ```

2. **Policy server** (any machine reachable from the sim machine), per
   [POLICY_SERVER.md](POLICY_SERVER.md). Ready when it logs
   `server listening on 0.0.0.0:8000`.

3. **eval_recorder** (benchmark repo, for videos + official scores):

   ```bash
   task2_isaacsim/scripts/run_eval_recorder.sh record --eval-name my_submission
   ```

## Running

Server connection settings live in `examples/ebim_task2/.env` (gitignored;
loaded automatically by docker compose and by the client itself — real
environment variables take precedence):

```bash
cp examples/ebim_task2/.env.example examples/ebim_task2/.env
# edit POLICY_SERVER_HOST / POLICY_SERVER_PORT / POLICY_SERVER_API_KEY
```

```bash
cd /path/to/openpi
export HOST_UID=$(id -u) HOST_GID=$(id -g)
mkdir -p examples/ebim_task2/eval_results  # pre-create so the bind mount is user-owned
docker compose -f examples/ebim_task2/compose.eval_client.yml build
docker compose -f examples/ebim_task2/compose.eval_client.yml run --rm eval_client
```

Cross-machine: the sim and the client share the machine (host-network DDS);
only the policy server is remote, addressed via `.env` (websocket). The
connection settings are deliberately excluded from the `config.yaml`
written into `eval_results/` so the artifact can be shared.

Extra flags go through `EVAL_ARGS` (CLI overrides `eval_config.yaml`):

```bash
EVAL_ARGS="--num-episodes 1 --no-async-inference" docker compose ... run --rm eval_client
```

## Inference modes

- **Async (default)**: a worker thread calls the server while the control
  loop keeps publishing queued actions on the 30 Hz sim-time grid. A new
  chunk is requested when the queue drops to
  `chunk_size_threshold × execute_horizon`; arriving chunks skip the rows
  made stale by inference latency and merge via `chunk_merge`:
  - `replace` — drop the old plan (jumps at every hand-off);
  - `blend_uniform` / `blend_favor_new` / `blend_favor_old` — fixed-weight
    average of the step-aligned overlap (0.5/0.5, 0.3/0.7, 0.7/0.3 old/new);
  - `crossfade` — old→new weight ramp over `chunk_blend_steps` rows;
  - `offset_decay` — adopt the fresh plan, shifted to be continuous with
    the next queued action, shift decayed over `chunk_blend_steps` rows
    (recommended).

  Base twist and gripper dims are never interpolated. Each episode record
  reports `switch_jump_mean/max` (arm-joint jump at the first action after
  a merge, rad) and `step_jump_mean/max` (between consecutive actions) for
  comparing strategies. Background, formulas and the strategies surveyed
  but not implemented: [docs/chunk_merge_strategies.md](docs/chunk_merge_strategies.md).
- **Sync** (`--no-async-inference`): blocking infer every `replan_steps`
  steps; the control grid re-anchors after each inference gap.

**Success rates are not comparable across modes** — async replans more
often at the same threshold.

## Smoke test without the sim

```bash
# against a live policy server; exercises resize, websocket, async buffer
python3 examples/ebim_task2/fake_obs_client.py --host <server-ip> --port 8000
```

Kill and restart the server mid-run to exercise the reconnect path.

## Outputs

`eval_results/<run-name>/` (default `eval_<unix-ts>` when `--run-name` is
not set; `run_eval.sh` passes one so client and recorder artifacts share it):
- `config.yaml` — resolved config + the server's checkpoint metadata
- `result.jsonl` — one line per episode: steps, duration, infer latency,
  async stats (chunk latency, dry steps, queue depth, merges dropped),
  abort reason if any
- `summary.md` + `images/` — merged report (evaluator scores + client
  timing), written by `summarize_eval.py` / `run_eval.sh`

Scores and videos live in the eval_recorder's output
(`services/eval_recording/eval_raw_out/<eval-name>/`).

## Abort reasons

| reason | meaning |
|---|---|
| `external scene reset` | someone pressed hotkey 5 mid-episode; the scene was wiped, episode unscorable, run continues |
| `sim clock stalled` | no clock progress for 10 s (sim paused/crashed) |
| `sim clock jumped backward` | unexpected reset/rebase mid-episode |
| `stale observation streams` | a camera/joint/odom stream went silent > 2 s |
| `action queue dry for N steps` | async: no fresh chunk for ~3 sim-s (server too slow/gone) |
| `incomplete observation snapshot` | missing image or non-finite state |
| `inference failed N times in a row` | websocket reconnect budget exhausted |

## Contract provenance

Topic names, joint names, pedal tokens and the 20-dim layout are
transcribed from the benchmark repo (`task2_isaacsim/config/topics.yaml`,
`services/recording/record_task2.py`) and match the training-side converter
[convert_task2_data_to_lerobot.py](convert_task2_data_to_lerobot.py). The
eval camera and ground-truth object poses are deliberately never
subscribed — competition rule: the policy sees only the three robot
cameras and the 20-dim state.
