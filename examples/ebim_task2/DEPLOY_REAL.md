# Deploying the EBiM task 2 policy on the Munich robot

This is the end-to-end path from a trained checkpoint to the policy driving the real FR3 Duo. It is a short overview with
the exact commands; every step links to the detailed guide:

- [SERVER.md](SERVER.md) for whoever runs the **policy server** on the GPU machine;
- [README_ROBOT_CLIENT.md](README_ROBOT_CLIENT.md) for the **operator** at the robot laptop (every message the client
  prints is explained there).

Run every command from the **root of the openpi checkout** (the folder that contains `examples/`), at the commit we
give you, on both machines.

## 1. The setup

```
 GPU machine (NVIDIA, Docker)                       Robot laptop (Docker, same LAN as the robot)
┌────────────────────────────────┐   TCP 8000     ┌──────────────────────────────────────────┐
│ policy server (pi0.5)          │◄──────────────►│ robot client (ROS 2 Humble, Fast DDS)    │
│ serve_munich.sh <config> <ckpt>│   websocket    │ $RC check-obs ... $RC run                │
│ returns 50-step action chunks  │                │ 3 cameras + joint states in, 100 Hz      │
└────────────────────────────────┘                │ joint targets + gripper commands out     │
                                                  └───────────────────┬──────────────────────┘
                                                                      │ ROS 2 (domain 0)
                                                                      ▼
                                                     FR3 Duo: both arms, grippers, spine, cameras
```

- The **server** needs a GPU and no ROS. It loads one checkpoint and answers requests: three images, a 16-dim state and
  the prompt in, a chunk of 50 actions (20 Hz, 16-dim) out. It tells every client which checkpoint it serves.
- The **client** needs ROS and no GPU. It reads the robot, asks the server for a chunk, smooths it and streams joint
  targets to both arms at 100 Hz. It takes over the arms' `joint_impedance_controller` at the start of an episode and
  releases it at the end. Nothing moves without the operator pressing Enter.
- Both can run on the same machine (set `POLICY_SERVER_HOST=localhost`); that is how we tested it.

**Shipped checkpoint:** config `pi05_ebim_task2_munich_real`, step `munich_v1/29999` (full fine-tune, the last step
of a 30,000-step run). Alternatives are listed in [section 5](#5-swapping-in-another-checkpoint).

## 2. Before the session (needs internet)

Do all of this on a network with internet access. Afterwards neither machine needs internet.

### 2.1 Both machines: get the code

```bash
git clone --branch ebim/task2-real-robot-client https://github.com/hermanprawiro/openpi.git
cd openpi
# git checkout <commit>        # only if we name a specific commit; use the same one on both machines
```

### 2.2 GPU machine: checkpoint, tokenizer, server image

Needs Docker with `docker compose`, the NVIDIA driver and the NVIDIA Container Toolkit, and about 8 to 10 GB of free
GPU memory ([SERVER.md §1](SERVER.md#1-what-you-need)).

```bash
# 1. Checkpoint (about 12.4 GB) from the public repo; no account or token needed
pip install -U huggingface_hub
hf download hermanprawiro/ebim-task2-real \
    --include "pi05_ebim_task2_munich_real/munich_v1/29999/*" --local-dir checkpoints
ls checkpoints/pi05_ebim_task2_munich_real/munich_v1/29999/assets/ebim/task2_munich_real/norm_stats.json

# 2. Tokenizer (the server would otherwise download it on its first start)
mkdir -p ~/.cache/openpi/big_vision
curl -L -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model \
    https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

# 3. Server image (several minutes)
POLICY_CONFIG=x CKPT_DIR=/tmp docker compose -f examples/ebim_task2/compose.policy_server.yml build
```

The weights are at <https://huggingface.co/hermanprawiro/ebim-task2-real>. `hf` warns about unauthenticated requests;
that is harmless. With a `huggingface_hub` older than 0.34 the command is `huggingface-cli download` (same arguments).
Details: [SERVER.md §2](SERVER.md#2-get-the-checkpoint) and [§6](SERVER.md#6-offline-use).

### 2.3 Robot laptop: client image

```bash
docker compose -f examples/ebim_task2/compose.robot_client.yml build      # a few minutes, about 1.6 GB
```

The image is ROS 2 Humble with Fast DDS, the same as the robot PC. Do not substitute another ROS distribution or RMW.
Details: [README_ROBOT_CLIENT.md §4.1](README_ROBOT_CLIENT.md#41-build-the-image-needs-internet-a-few-minutes-about-16-gb).

### 2.4 Network

The robot laptop must reach the GPU machine on TCP port 8000. Open that port in the GPU machine's firewall. The laptop
must also be on the robot's ROS 2 network (domain 0, or set `ROS_DOMAIN_ID` and `ros.domain_id` together:
[README_ROBOT_CLIENT.md §2](README_ROBOT_CLIENT.md#2-what-must-be-true-before-you-start), item 5).

## 3. On the day: start the server

On the GPU machine. Restart the server shortly before the session rather than leaving it up for days (a long-running
container can lose the GPU after a system update).

```bash
examples/ebim_task2/serve_munich.sh pi05_ebim_task2_munich_real \
    checkpoints/pi05_ebim_task2_munich_real/munich_v1/29999
```

Wait for `Warmup done in ... s` (one to two minutes). Then, from the laptop:

```bash
curl http://<GPU machine ip>:8000/healthz        # prints: OK
```

The server runs in the foreground; Ctrl-C stops it. Log lines and problems: [SERVER.md §4](SERVER.md#4-expected-log-lines-and-the-health-check)
and [§7](SERVER.md#7-troubleshooting).

## 4. On the day: run the robot

On the robot laptop, after the organizers have brought the robot up. Before you start:

- GELLO teleoperation is **stopped** (nothing else publishes on `/<side>/gello/joint_states`).
- Each arm's `joint_impedance_controller` is loaded and **inactive**.
- The thermal pad and the RAM board are on the table as in the data collection.
- Your hand is at the **E-stop** whenever an arm can move. It is the primary safety; the client's checks are a second line.

Full list: [README_ROBOT_CLIENT.md §2](README_ROBOT_CLIENT.md#2-what-must-be-true-before-you-start).

```bash
script -f "$HOME/robot_session_$(date +%Y%m%d_%H%M).log"     # keep a transcript; type `exit` at the end

export POLICY_SERVER_HOST=<GPU machine ip>
RC="docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client"
```

Then run these in order. Do not go on until a step passes.

| # | Command | Moves? | Must show |
|---|---|---|---|
| 1 | `$RC check-obs` | no | `check_obs: PASS`; look at the three saved JPEGs |
| 2 | `$RC check-infer` | no | `check_infer: PASS`, `layout munich16`, `checkpoint munich_v1/29999`, chunk `(50, 16)` |
| 3 | `$RC check-motion --arm right` | **one arm, ±0.05 rad per joint** | `check_motion_right: PASS`, every joint `follows` |
| 4 | `$RC check-motion --arm left` | **one arm, ±0.05 rad per joint** | `check_motion_left: PASS`, every joint `follows` |
| 5 | `$RC prepare` | **both arms to the start pose, spine, grippers** | `prepare: done` |
| 6 | `$RC run --dry-run --episodes 1` | no (publishes only on preview topics) | the episode ends without `FREEZE` |
| 7 | `$RC run --episodes 1` | **yes: the policy drives both arms** | an `Outcome?` prompt after `Released.` |

What each step does and what to do when it fails: [README_ROBOT_CLIENT.md §5](README_ROBOT_CLIENT.md#5-step-by-step).
The keys, exit codes and the meaning of every message: [§3](README_ROBOT_CLIENT.md#3-quick-reference) and
[§8](README_ROBOT_CLIENT.md#8-what-the-messages-mean).

An episode of `run` goes like this:

1. `Reset the scene...`: place the pad and the board, step back, press Enter. Both arms move to the start pose.
2. The client takes over both arms while holding their current pose, then prints `Press Enter to start the policy`.
   **The controllers are active from here on.**
3. Enter starts the policy. While it runs: `p` pause, Enter resume, `q` end the episode, Ctrl-C emergency release.
   The default mode stops and goes: the arms hold still while the server computes (about 0.2 to 0.6 s), then execute
   1.5 s of the chunk.
4. The episode ends after 120 s, on `q`, or on a `FREEZE` (a stream went quiet or the server stalled). The client holds,
   deactivates the controllers, waits until they read inactive, prints `Released.` and asks for the outcome
   (`s` / `f` / `x`).

Run more episodes with `$RC run --run-name munich` (it continues until you answer `q` at the reset prompt; the same
`--run-name` after a `FREEZE` continues the numbering).

### After the session

```bash
sudo chown -R "$USER": examples/ebim_task2/robot_runs
tar czf munich_robot_runs.tar.gz -C examples/ebim_task2 robot_runs
```

Send us `munich_robot_runs.tar.gz`, the `script` transcript and your notes
([README_ROBOT_CLIENT.md §10](README_ROBOT_CLIENT.md#10-what-to-send-us)).

To look through the runs afterwards (for us; the operator does not need this), unpack the archive and start the
viewer from the host, no Docker needed (numpy required; pyyaml optional, for the settings summary), then open
<http://localhost:8765>:

```bash
tar xzf munich_robot_runs.tar.gz -C /some/dir
cd examples/ebim_task2 && python3 -m robot_client.viewer --runs /some/dir/robot_runs
```

It only reads the folder. The same page can run live next to the client:
[README_ROBOT_CLIENT.md §12](README_ROBOT_CLIENT.md#12-optional-watch-the-run-in-a-browser).

## 5. Swapping in another checkpoint

Only the server changes. The client image, its YAML and the commands above stay as they are.

1. Between episodes (never during one), stop the server with Ctrl-C.
2. Download the other step: `hf download hermanprawiro/ebim-task2-real --include "<config>/<exp>/<step>/*" --local-dir checkpoints`.
3. Start it with **the config it was trained with**: always the first directory level of the checkpoint path, e.g.
   `examples/ebim_task2/serve_munich.sh pi05_ebim_task2_munich_real_from_sim checkpoints/pi05_ebim_task2_munich_real_from_sim/munich_v1/20000`.
   A wrong pairing of the two full-FT configs stops the server at start with `ValueError: state norm stats are
   20-dim, but this config expects 16` (or `16-dim ... expects 20`): use the config from the path.
4. On the laptop, `$RC check-infer` must show the new checkpoint name. `run` records it in `server_metadata.json`.

Checkpoints in the public repo (hold-out scores and how they were chosen: [TRAINING_REAL.md §5](TRAINING_REAL.md#5-hold-out-eval-and-choosing-a-step)):

| Config (= first directory level) | Step | Note |
|---|---|---|
| `pi05_ebim_task2_munich_real` | **29999** | shipped; full fine-tune from `pi05_base` |
| `pi05_ebim_task2_munich_real` | 20000 | same run, best hold-out step by the selection rule |
| `pi05_ebim_task2_munich_real_from_sim` | 20000, 29999 | full fine-tune from the sim checkpoint: lower arm error and start offset on the hold-out, slightly worse gripper timing |
| `pi05_lora_ebim_task2_munich_real` | 12500 | the earlier LoRA stand-in (about 6.3 GB) |

Details: [SERVER.md §5](SERVER.md#5-swapping-the-checkpoint). How a checkpoint is trained, scored and uploaded:
[TRAINING_REAL.md](TRAINING_REAL.md).

## 6. If something goes wrong

| Situation | Do |
|---|---|
| Anything moves unexpectedly | E-stop |
| `REFUSED: ...` or `FAIL ...` (exit code 2) | Look the line up in [README_ROBOT_CLIENT.md §8](README_ROBOT_CLIENT.md#8-what-the-messages-mean); the client did not move anything it was refusing |
| `... is already active; run release --arm <side>` | `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client release --arm <side>`, then retry |
| `CONTROLLER STILL ACTIVE on ...` | Leave the client running. Deactivate that controller with your tooling (or the `release` command above, from a second terminal) or press the E-stop. Do **not** `docker stop` the client ([§11](README_ROBOT_CLIENT.md#11-if-the-container-was-killed-while-an-arm-was-active)) |
| The client container was killed while an arm was active | E-stop if anything moves, then `... run --rm robot-client release --arm left --arm right` |
| `cannot reach policy server` | Is the server past `Warmup done`? `curl http://<ip>:8000/healthz` from the laptop; firewall; `POLICY_SERVER_HOST` |
| Every `check-obs` stream says `no messages` | ROS domain or Fast DDS profile ([README_ROBOT_CLIENT.md §4.4](README_ROBOT_CLIENT.md#44-fast-dds-profile)) |
| Server out of memory, or `NVML: Unknown Error` | Stop other GPU jobs; restart the server container ([SERVER.md §7](SERVER.md#7-troubleshooting)) |
