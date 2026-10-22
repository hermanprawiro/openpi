# How to Run the Full 10-Episode Eval (EBiM Task 2)

End-to-end procedure for an official-style eval run: Isaac Sim scene +
remote policy server + eval_recorder (videos & IoU scores) + the
autonomous eval client. See [README_EVAL_CLIENT.md](README_EVAL_CLIENT.md)
for architecture details and the abort-reason glossary.

Throughout, two repo roots are used — adjust to your machines:

```bash
export OPENPI=~/codes/openpi                 # this repo (sim machine)
export EBIM=~/codes/ebim/benchmark           # benchmark repo (sim machine)
```

The **sim, recorder, and client all run on the sim machine** (they share
host-network DDS). Only the **policy server** may be on another machine.

---

## 0. One-time setup

```bash
# Client image (build context = openpi root)
cd $OPENPI
docker compose -f examples/ebim_task2/compose.eval_client.yml build

# Recorder image (benchmark repo)
$EBIM/task2_isaacsim/scripts/run_eval_recorder.sh build

# Server connection -> .env (gitignored; never lands in eval artifacts)
cp examples/ebim_task2/.env.example examples/ebim_task2/.env
# edit: POLICY_SERVER_HOST=<server-ip>, POLICY_SERVER_PORT=8000
#       POLICY_SERVER_API_KEY=...        (only if the server uses one)
#       EBIM_ROOT=<path to the benchmark repo root, e.g. ~/codes/ebim/benchmark>
#       (used by run_eval.sh to launch the eval_recorder)

# Results dir must exist user-owned before the first compose run
mkdir -p $OPENPI/examples/ebim_task2/eval_results
export HOST_UID=$(id -u) HOST_GID=$(id -g)

# REQUIRED on the sim machine: enlarge kernel UDP buffers. The 720p head
# camera is ~2.7 MB/frame over best-effort UDPv4; with the recorder and
# client both subscribed, the stock 208 KB buffer drops frame bursts and
# episodes abort with "stale observation streams: ['image_head']".
# (Verified: 7/10 episodes aborted with stock buffers, 10/10 clean after.)
echo -e "net.core.rmem_max=67108864\nnet.core.rmem_default=67108864" | \
  sudo tee /etc/sysctl.d/99-ros2-dds.conf && sudo sysctl --system
```

Note: compose bind-mounts `eval_config.yaml` over the copy baked into the
image, so edits to it apply on the next `run` — no rebuild needed. Rebuild
only after changing `eval_client/` code. Per-run overrides go via `EVAL_ARGS`.

## 1. Policy server (any machine with the checkpoint)

```bash
cd <openpi-on-server-machine>
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_ebim_task2_fixpos_200_finetune \
    --policy.dir=checkpoints/pi05_ebim_task2_fixpos_200_finetune/pi05_fixpos_200/29999
```

If the server shares a GPU with Isaac Sim (same machine), cap JAX first:
`export XLA_PYTHON_CLIENT_MEM_FRACTION=0.35`.

Ready when it logs `server listening on 0.0.0.0:8000` (~1 min weights +
~40 s warmup). Verify from the sim machine:

```bash
source examples/ebim_task2/.env
curl -s -o /dev/null -w "%{http_code}\n" http://$POLICY_SERVER_HOST:$POLICY_SERVER_PORT/healthz   # expect 200
```

## 2. Isaac Sim scene (sim machine)

The Isaac Sim 5.1.0 container must already be running (see the benchmark
repo README). Then:

```bash
cd $EBIM
./task2_isaacsim/scripts/run_isaacsim_teleop.sh --scene room \
    --no-browser --no-republisher --controller-mode none \
    -- --record --randomize-objects
```

- `--record` is **required** (publishes cameras, `/isaac/joint_states_full`, odom).
- Do **not** pass `--arm-keyboard-teleop` (it bypasses ROS arm commands).
- Add `--headless` (before the `--`) to run without a Kit window; note the
  hotkey-5 manual reset is unavailable headless, and expect a lower
  real-time factor (~0.2×: a 90 sim-second episode ≈ 7 wall minutes).
- Object randomization (`--randomize-objects`) re-jitters on **every**
  scene reset, i.e. per episode.
- For policy trained with fixpos dataset, add `--robot-x 2.1 --robot-y 3.05` to
  mimic the robot starting position during training.

Loading takes several minutes. Ready when the observation topics are live:

```bash
cd $OPENPI
docker run --rm --network host --ipc host \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e HOME=/tmp --user $(id -u):$(id -g) ebim-task2-eval-client \
  bash -c "source /opt/ros/jazzy/setup.bash && ros2 topic hz /isaac/head_camera/image_raw"
# expect ~10-24 Hz; Ctrl+C to stop
```

## 3. Official evaluator (sim machine)

The recorder auto-triggers scoring through the evaluator stack; make sure
it is up (it provides the `/isaac/eval_camera/evaluate` service):

```bash
docker ps --format '{{.Names}}' | grep -q eval_task2 \
  || $EBIM/scripts/evaluation/task2/run.sh up
```

## 4. Run the eval — recorder + client, one command (sim machine)

```bash
cd $OPENPI
./examples/ebim_task2/run_eval.sh --eval-name <submission-name>
# smoke test: ./examples/ebim_task2/run_eval.sh -- --num-episodes 1
```

The wrapper generates a name if `--eval-name` is omitted
(`eval_<YYYYmmdd_HHMMSS>`), then:

1. starts the eval_recorder (`$EBIM_ROOT` from the environment or `.env`)
   and waits for its session `.status.json` to appear — the recorder must
   be listening before the client's first reset, because it segments
   episodes on scene-reset *events* and scores on *requests*. Recorder
   console output goes to `<session-dir>/recorder.log` (`tail -f` it to
   watch); the terminal shows only the client;
2. runs the eval client with `--run-name <name>` so both artifact sets
   share the name;
3. after the client exits, sends the recorder `[0]` (stop+**discard** the
   trailing episode opened by the client's final reset) and `[q]` (quit —
   waits for in-flight evaluator calls and video encoders to finish);
4. writes a merged report `eval_results/<name>/summary.md` (see section 6)
   and exits with the client's exit code.

Everything after `--` goes to the client, e.g.
`-- --num-episodes 3 --no-async-inference`. `--dry-run` prints the
resolved name, artifact dirs and commands without starting anything.

What you should see, per episode: `Episode k/10: resetting scene` →
(recorder logs the reset + auto-evaluation of the previous episode) →
~90 sim-seconds of motion → `Episode k result: {...}` with steps ≈ 2700,
`queue_dry_steps` near 0, `aborted: false` (`switch_jump_max`, the largest
arm-joint jump at a chunk hand-off in rad, is the number to compare when
changing `chunk_merge`; see `docs/chunk_merge_strategies.md`). After
episode 10 the client
publishes one final reset request so the recorder scores the last episode.

Artifacts (paired by the shared name):

```
$OPENPI/examples/ebim_task2/eval_results/<name>/
  result.jsonl, config.yaml          # client: timing, abort reasons (no server address)
  summary.md, images/                # merged report (auto-generated)
$EBIM/task2_isaacsim/services/eval_recording/eval_raw_out/<name>/
  ep_NNN/videos/{head,wrist_left,wrist_right,eval_camera}.mp4   # PTS = sim time
  ep_NNN/evaluator/call_NNN/                                    # official IoU scores
  ep_NNN/{frames,dumps}/*.jsonl, manifest.json
```

### Manual mode (fallback)

Start the recorder yourself, **before** the client, then run the client
with the same name:

```bash
$EBIM/task2_isaacsim/scripts/run_eval_recorder.sh record --eval-name <name>
# other terminal:
cd $OPENPI && export HOST_UID=$(id -u) HOST_GID=$(id -g)
EVAL_ARGS="--run-name <name>" \
  docker compose -f examples/ebim_task2/compose.eval_client.yml run --rm eval_client
```

Recorder console keys: `[s]` status, `[e]` manual evaluate, `[0]`
stop+discard the current episode, `[q]` quit. After the client exits,
press `[0]` then `[q]` yourself (the final reset leaves a trailing
episode recording), and generate the report by hand:

```bash
python3 examples/ebim_task2/summarize_eval.py \
  --session-dir $EBIM/task2_isaacsim/services/eval_recording/eval_raw_out/<name> \
  --run-dir examples/ebim_task2/eval_results/<name>
```

Without `--run-name` the client falls back to `eval_results/eval_<unix-ts>/`.

## 6. After the run

- Read `eval_results/<name>/summary.md` first: aggregate stats, one row
  per episode (target slot, success, evaluator scores + client
  timing/aborts), a success-by-target-slot table (success = IoU > 0 and
  orientation correct; slots other than the home slot `B` measure position
  generalization), and the eval-camera frame per episode. Official protocol is best-of-10.
- Source of truth remains the raw recorder output
  (`ep_NNN/evaluator/call_NNN/`); `summarize_eval.py` can be re-run on any
  old session. The trailing recorder episode is intentionally marked
  *discarded* by the wrapper and excluded from the summary stats.
- Stop the sim (Ctrl+C in its terminal, or
  `docker exec isaac-sim-5-1-0-workshop pkill -f scene_room.py`), stop
  the server when done.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `observation streams not live` at start | Sim not ready yet, or launched without `--record`. |
| Mid-episode abort `stale observation streams: ['image_head']` | Best-effort UDP frame drops under recorder+client fan-out — apply the kernel-buffer step from section 0. If it persists, check the sim console for renderer stalls. |
| `no subscriber on the scene-reset request topic` | Sim not running the reset controller (`--record` missing) or DDS still discovering — retry after a few seconds. |
| `scene reset not confirmed` | Sim busy/stalled; check the sim console. Deformable re-init on reset can take a few seconds — the client already waits and retries once. |
| `action queue dry for N steps` abort | Server too slow or unreachable mid-episode; check server logs / network. The client holds the last action for ~3 sim-s before aborting. |
| `external scene reset` abort | Someone pressed hotkey 5 mid-episode. Episode is unscorable; the run continues. |
| Episodes take ~7 min wall each | Normal headless real-time factor; timeout is sim-time, results are unaffected. |
| Client `config.yaml` missing host/port | Intentional — connection settings are redacted from eval artifacts. |
| `run_eval.sh`: `Run dir already exists` | The name was used before (client or recorder side). Pick a new `--eval-name` or remove both old dirs — the client refuses to append to an existing `result.jsonl`. |
| `run_eval.sh`: recorder never becomes ready | Recorder image not built (`run_eval_recorder.sh build`), or `EBIM_ROOT` points at the wrong directory. Check the recorder's own output above the wrapper message. |
| Sim dies with `CUDA error 100: cudaErrorNoDevice` while host `nvidia-smi` works | The long-running Isaac container lost GPU access (common after a host NVIDIA driver update or systemd daemon-reload). `docker start isaac-sim-5-1-0-workshop`, verify with `docker exec isaac-sim-5-1-0-workshop nvidia-smi`, then relaunch the scene. |
