# EBiM task 2 robot client: operator guide (Munich diagnostic run)

This guide is for the operator who runs our Docker image on the evaluation laptop next to the robot. You will not be
able to ask us questions during the session, so every refusal the client can print is explained in
[section 8](#8-what-the-messages-mean) with the setting or action that fixes it. Read sections 1 to 4 before the
session, then work through section 5 in order.

The person who starts the policy server on the GPU machine should read [SERVER.md](SERVER.md) instead.

## 1. What this is

- A **one-shot diagnostic run**, not an official submission. We trained a pi0.5 policy on the organizer-released Munich
  teleoperation data. The run answers one question: does that policy drive the real robot sensibly, and where does the
  real-robot path (cameras, latency, controller takeover, safety) break?
- The client reads the three cameras and both arms' joint states from ROS 2, asks the policy server for a chunk of
  50 actions (20 Hz, 2.5 s), and streams smoothed joint targets (100 Hz) to both arms plus gripper commands.
- The task prompt is fixed: `Pick up the thermal pad and place it on the target RAM board`.
- Everything you may need to change on site is a key in one YAML file (`robot_client/robot_site.munich.yaml`), a
  `--set key=value` flag, or an environment variable. You never edit source.
- **Safety model.** Nothing moves without you pressing Enter at a prompt. `check-obs`, `check-infer` and
  `run --dry-run` never move anything. The client refuses to start if anything looks wrong and ends an episode
  (`FREEZE`) if a camera or joint stream goes quiet, the policy server stalls, or its own command stream stops.
  **Your hand stays at the E-stop. The E-stop is the primary safety; the client's checks are a second line.**

## 2. What must be true before you start

1. The robot is brought up and enabled by the organizers: both arms, the spine, both grippers and all three cameras are
   running and publishing, and the robot is placed at the table where the training data was recorded. The client does no
   base navigation or docking.
2. **GELLO teleoperation is stopped.** Nothing else may publish on `/left/gello/joint_states` or
   `/right/gello/joint_states`. The client refuses to take over an arm whose command topic carries another publisher's
   messages.
3. On each arm's controller manager (`/left/controller_manager`, `/right/controller_manager`) the controller
   `joint_impedance_controller` is loaded and configured (state `inactive`). The client activates it and deactivates it
   again at the end. **If the controller is already `active` when the client starts, the client refuses** (`... is
   already active; run release --arm <side> ...`, nothing is published): run `$RC release --arm <side>` (or deactivate
   it with your tooling) and retry. If your tooling must do the activation instead, see
   [question 7](#7-site-facts-we-could-not-confirm-and-how-to-settle-them-on-site).
4. The thermal pad and the target RAM board are on the table, as in the data collection.
5. The laptop is on the same network and ROS domain as the robot (`ROS_DOMAIN_ID=0`, Humble, Fast DDS), and can reach the
   machine that runs the policy server on TCP port 8000 (or the port you chose). If the robot uses another domain,
   `export ROS_DOMAIN_ID=<n>` (the compose file passes it into the container) **and** set `ros.domain_id: <n>` in the
   site YAML; the client refuses to move anything while the two differ.
6. The laptop has Docker with the `docker compose` plugin and a keyboard in an interactive terminal. The client needs a
   real terminal (it reads single key presses only while the policy runs and at the start gate; every other prompt needs the answer typed, then Enter).

## 3. Quick reference

**Order of commands** (each is explained in [section 5](#5-step-by-step)):

```
check-obs  ->  check-infer  ->  check-motion --arm right  ->  check-motion --arm left
           ->  prepare  ->  run --dry-run  ->  run
```

**Keys**

**Type the letter, then press Enter** at every prompt that shows a list of answers in brackets: the `Reset the scene...`, `Workspace clear...`, `Outcome?` and `Note` prompts, including `q`, `k`, `s`, `f`, `x`. Only two places react to a single key press, without Enter: `Press Enter to start the policy` (Enter or `q`) and the running policy (`p`, Enter, `q`).

| Where | Key | Effect |
|---|---|---|
| Any prompt that says `[Enter] ... / [q] quit` | Enter | Continue (moves the robot when the prompt says so) |
| | `q` then Enter | Quit without moving |
| `Reset the scene...` prompt before an episode | Enter | Move both arms to the start pose (PTP), spine to its height, open grippers |
| | `k` then Enter | Keep the current pose (skip the move) |
| `Press Enter to start the policy` | Enter (single key) | Start the policy |
| | `q` (single key) | Release the controllers and quit |
| While the policy runs (single keys) | `p` | Pause: the arms hold still |
| | Enter | Resume after a pause |
| | `q` | End the episode (the arms hold, then the controllers are released) |
| | Ctrl-C | Emergency release: the client stops and releases what it activated |
| `Outcome?` prompt | `s` / `f` / `x`, then Enter | Success / fail / skip. Then an optional one-line note (type it, Enter; or just Enter) |
| `CONTROLLER STILL ACTIVE ...` message during a release | Enter (single key) | Only after you deactivated the controller with your tooling or pressed the E-stop: the client stops holding that arm and exits 5. The Enter answers only the arm(s) named in the messages shown so far; an arm still waiting for its deactivation keeps its hold and gets its own message, which needs its own Enter. Do **not** `docker stop` the client here (section 11) |

Keys you typed before a prompt appeared are thrown away, so an early Enter cannot start anything.
Pressing Ctrl-C twice does nothing extra: after the first one the client finishes its release
(`releasing — wait, or use the E-stop`).

**Exit codes**

| Code | Meaning |
|---|---|
| 0 | Done, or the check passed |
| 1 | Unexpected crash (a Python traceback). Not designed: send us the transcript |
| 2 | Refused or a check failed. The line starting with `REFUSED:` or `FAIL` says why |
| 3 | `FREEZE`: the episode was stopped for safety and the controllers were released |
| 4 | Configuration or usage error (`CONFIG ERROR:` or an argument error) |
| 5 | `CONTROLLER STILL ACTIVE`: a controller could not be confirmed inactive after the release and you confirmed with Enter that you handled it. Make sure it is deactivated (section 11) before anything else |
| 130 | You pressed Ctrl-C (or the container got SIGTERM, or the terminal was closed: SIGHUP). Everything the client activated was released first. During a release (including a `CONTROLLER STILL ACTIVE` hold) these signals are ignored: the release always finishes first |

**Facts worth knowing**

- The policy **never changes the left gripper**: in the training data it is always open. The left gripper's state stream
  is therefore **not required**, and its state value is fixed at 0.0 (`state.pin: {7: 0.0}` in the site YAML). A
  `check-obs` row `info  grip/left  no messages ... (optional; state dim 7 is pinned to 0.0)` is normal. The **right**
  gripper state is required.
- In the default `sync` mode the policy server is asked for a chunk, the arms **hold still while the client waits**
  (typically 0.5 to 0.6 s on an L4 GPU, measured by an evaluator with a different pi0.5), then 30 of the 50 actions
  (1.5 s) are executed, and so on. You will see a stop-and-go motion. That is intended.
- An episode ends after 120 s (`safety.episode_timeout_s`), when you press `q`, or on a `FREEZE`.

## 4. Build and run

Run all commands from the **root of the openpi checkout** (the folder that contains `examples/`), at the commit we gave
you.

### 4.1 Build the image (needs internet, a few minutes, about 1.6 GB)

```bash
docker compose -f examples/ebim_task2/compose.robot_client.yml build
```

Do this before the session, on a network with internet access. The build downloads the ROS 2 Humble base image,
apt packages, the Franka message packages (from GitHub) and Python packages. Running the client later needs no
internet. The image is `ros:humble-ros-base` with `rmw_fastrtps_cpp`, so it joins the robot's ROS graph the same way
the robot PC does (an earlier Jazzy-based container crashed the evaluation PC at Munich; do not substitute another ROS distribution or RMW).

### 4.2 Run any command

Set the policy server address once (use the IP of the GPU machine), and define a shortcut:

```bash
export POLICY_SERVER_HOST=<ip or hostname of the GPU machine>
# export POLICY_SERVER_PORT=8000          # only if the server uses another port
# export POLICY_SERVER_API_KEY=...        # leave unset unless the server operator gives you a key

RC="docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client"

$RC --help                     # list the commands
$RC check-obs --help           # options of one command
$RC check-obs
```

If `POLICY_SERVER_HOST` is not set the client prints a banner of `*` characters and tries `localhost:8000`.
Leave `POLICY_SERVER_API_KEY` unset unless the person running the policy server tells you otherwise (our server runs
without a key: SERVER.md sets none).

The container uses `--network host`, an interactive terminal, and the site file
`examples/ebim_task2/robot_client/robot_site.munich.yaml` mounted read-only at `/config/robot_site.yaml`. Edit that file
in place to change a setting for good (no rebuild needed), or add `--set key=value` after the command for a one-off
change, for example `$RC run --dry-run --set inference.execute_horizon=20`. Several `--set` flags are allowed. Values
are YAML, so lists are written like `--set 'joint_signs.right=[1,-1,1,1,1,1,1]'`.

Other commands that work through the same shortcut: `$RC bash` (a shell in the container), `$RC ros2 topic list`,
`$RC ros2 topic hz <topic>`.

### 4.3 Where the results go, and who owns them

Every command writes into `<OUT_DIR>/<run name>/`. The default `OUT_DIR` is `./robot_runs` **relative to the compose file**,
so it is `examples/ebim_task2/robot_runs/`; the run name is a timestamp in the container's time (UTC) unless you pass
`--run-name NAME` (letters, digits, `.`, `_`, `-`). Use another disk or folder with
`OUT_DIR=/some/absolute/path $RC ...` (export it like `POLICY_SERVER_HOST` to make it stick).

**The container runs as root, so everything in `robot_runs` is owned by root.** Hand it back before you copy or send it:

```bash
sudo chown -R "$USER": examples/ebim_task2/robot_runs      # or your OUT_DIR
```

Tip: keep a transcript of the terminal. The client's log lines and your prompts are not all saved in the run
directory. Start the session with

```bash
script -f "$HOME/robot_session_$(date +%Y%m%d_%H%M).log"
```

and type `exit` when you are done. Include that file when you send us the results.

### 4.4 Fast DDS profile

The Munich hosts use a UDP-only Fast DDS profile. The client therefore defaults to the profile we ship,
`/app/robot_client/fastdds_udp_only.xml` (UDPv4 only, no shared memory). You do not need to do anything. To change it:

| You want | Do |
|---|---|
| Fast DDS's built-in transports (UDP plus shared memory) | `FASTRTPS_DEFAULT_PROFILES_FILE= $RC check-obs` (set it to the empty string) |
| Your own profile file | `$RC` with extra flags: `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm -v /host/dir:/dds:ro -e FASTRTPS_DEFAULT_PROFILES_FILE=/dds/your_profile.xml robot-client check-obs` |

If the variable names a file that is not readable inside the container the client prints a `WARNING` at startup. If no
topic appears at all (every `check-obs` stream says `no messages`), the profile or the ROS domain is the first thing
to try; see the table in section 8.

## 5. Step by step

Use `--run-name` if you want the results of several steps in one folder, for example `--run-name munich`. Otherwise every
command makes its own timestamped folder, which is fine.

### Step 1: `check-obs` (nothing moves)

```bash
$RC check-obs
```

Listens for 8 s and prints one row per stream: the three cameras (expected about 20 / 30 / 30 Hz), both arms'
measured joint states (100 Hz or more), both grippers. It also reads the spine position and saves one preprocessed
frame per camera (`check_obs_head.jpg`, `check_obs_wrist_left.jpg`, `check_obs_wrist_right.jpg`) in the run folder.
Result `check_obs: PASS` and exit code 0 is required to go on.

What to look at beyond PASS:

- **Rates.** `warn rate ... is below 80 % of the expected ...` is a warning, not a failure, but note it. The arm rows
  matter most: the arm states and our 100 Hz command stream share one process with the camera callbacks, and heavy
  callback load can delay the command stream (see [section 9](#9-keeping-the-command-stream-at-100-hz)).
- **Image shapes.** Each camera row ends in `224x224x3`. The head camera row should read like
  `1280x720 bgr8 -> resize to 336x188 -> 224x224x3` (the live frame is resized to the size of the training frames, then
  padded to a square).
- **Look at the three JPEGs.** The head image should show the table scene with black bars at top and bottom; the wrist
  images should show the grippers' view. If the head view is framed very differently from the data collection, see
  [question 4](#7-site-facts-we-could-not-confirm-and-how-to-settle-them-on-site).
- **Spine.** `ok spine at X m, spine.height Y m` is good. A `warn spine` row means the configured spine height differs
  from where the spine is; see question 3.
- Optional: copy the checkpoint's `norm_stats.json` (in its `assets/ebim/task2_munich_real/` folder) into the output folder
  and add `--norm-stats /out/norm_stats.json` to get a warning for any state value outside the range the policy was
  trained on.

### Step 2: `check-infer` (nothing moves)

```bash
$RC check-infer
```

Connects to the policy server, checks that it serves our layout (`munich16`) and the exact prompt, sends 3 requests built
from the live streams (zeros if streams are missing), and prints the chunk shape (expect `(50, 16)`), the round-trip time
and the server's own inference time. Required: `check_infer: PASS`. The `server` row also shows the checkpoint name the
server loaded (`<exp>/<step>`); write it down. The checkpoint we ship for the session is the full fine-tune
`pi05_ebim_task2_munich_real`, step `munich_v1/29999` (see [SERVER.md](SERVER.md)), so expect
`config pi05_ebim_task2_munich_real, checkpoint munich_v1/29999` unless we told you otherwise. On our RTX 3090 a
round trip with the earlier LoRA stand-in was about 0.2 s (mean 193 ms, max 432 ms over 10 requests); the full
fine-tune was not timed on that GPU.

A `warn latency ... the arm will wait for the policy` row means a request took longer than the time one chunk lasts
(1.5 s in the default sync mode). The robot still runs, but slowly; tell us the numbers.

### Step 3: `check-motion --arm right`, then `--arm left` (small motion)

```bash
$RC check-motion --arm right
$RC check-motion --arm left
```

**This moves the arm.** Clear the workspace and keep your hand on the E-stop. After you press Enter the client takes over
that one arm (it activates the controller while holding the arm's current pose) and moves each of the 7 joints by
0.05 rad (about 3 degrees) and back, one joint at a time, about 4 s per joint. It then judges, per joint, whether the
measured joint followed the command.

| Verdict | Meaning |
|---|---|
| `follows` | The joint moved by roughly the commanded amount in the right direction (ratio 0.5 to 1.5) |
| `reversed` | It moved the opposite way (ratio -1.5 to -0.5) |
| `no-motion` | It did not move (ratio below 0.2 in size) |
| `unclear` | It moved too little or too much; rerun with `--delta 0.1` (the maximum is 0.2) |

Required: `check_motion_right: PASS` and `check_motion_left: PASS`, with every joint `follows`. The check aborts and
releases the arm immediately if any joint drifts more than `2 * delta + 0.02` rad from where the check started.

The controller must be `inactive` before the check: if it is already `active` (left on by bring-up, or by an earlier
external activation) the check fails with `FAIL takeover refused: ... is already active; run release --arm <side> ...`
without publishing anything. Run `$RC release --arm <side>` (or deactivate it with your tooling) and repeat the check.

If a joint is `reversed`, or all joints are `no-motion`, this tells you how the Munich controller reads the command
topic. Follow [question 1](#7-site-facts-we-could-not-confirm-and-how-to-settle-them-on-site) before continuing, and
re-run this step until it passes.

### Step 4: `prepare` (the robot moves a lot)

```bash
$RC prepare
```

**This moves both arms to the start pose and the spine to its working height.** Clear the workspace first. After Enter
the client: deactivates the impedance controllers (if active), moves the left arm, then the right arm, to
`start_pose` with the robot's own PTP action (slow: at most 0.3 to 0.4 rad/s), moves the spine to `spine.height`, and
opens both grippers. The start poses in the YAML are real start poses from the training data. Each step has a 60 s
timeout (`prepare.timeout_s`). **It always moves both arms**, also before a `run --arms right`: the arm that does not
run the policy must be at its start pose too.

`prepare: done` is the success line. Any failure prints `FAIL <step>: ...` and `REFUSED: prepare stopped at '<step>'`;
see the table in section 8. You can skip this command if the scene is already in a start-like pose: the `run` command
offers the same move at the start of every episode.

### Step 5: `run --dry-run` (nothing moves)

```bash
$RC run --dry-run --episodes 1
```

The complete run, with real cameras, real policy server and real recording, **but the client never activates or
deactivates a controller, never runs PTP or spine, sends no gripper commands, and publishes only on the preview topics**
`/left/robot_client_preview/joint_states` and `/right/robot_client_preview/joint_states`. The arm commands the policy
would send can be watched there. It prints `Press Enter to start the policy`; press Enter, watch it run (use `p` and `q`
as above), and answer the outcome prompt (`s` is fine).

Expected in a dry run:

- The robot does not follow the preview commands, so in a dry run the tracking leash (`safety.max_track_lag_rad`)
  compares the preview with its own last command instead of the measured pose: the preview follows the policy all the
  way and a `tracking lag > 0.50 rad` line should not appear. (The observations and the safety checks still use the
  real measured joints.)
- Stop it with `q` when you have seen enough, or let the 120 s timeout end it.

What counts as success: the episode ran without `FREEZE`, and the first chunk's start was accepted (no
`first action of the ... arm is ... rad from measured`).

### Step 6: `run` (the robot moves)

```bash
$RC run --episodes 1
```

Without `--episodes` the run continues episode after episode until you answer `q` at a reset prompt. For each episode:

1. `=== episode N ===` and `Reset the scene. Workspace clear? [Enter] move both arms to the start pose / [k] keep
   current pose / [q] quit`. Put the thermal pad and the RAM board in place, step back, then press Enter (`k` to skip the
   move). The robot runs the same PTP / spine / gripper sequence as `prepare`, for both arms (also with `--arms right`).
2. The client takes over both arms: it checks that no other publisher is active, that no controller is already active
   (refused otherwise, section 8.6) and that the measured joints are fresh, streams each arm's current pose, then
   activates the controller. Both arms are taken over even when only one is active, so GELLO must be stopped for both
   (`--arms right` makes the left arm hold its pose; with `--set takeover.hold_inactive_arms=false` the left arm is not
   taken over at all).
3. `Press Enter to start the policy (q to release and quit)`. **The controllers are active and holding the arms now**:
   your hand is on the E-stop from here on. Press Enter when you are ready.
4. `Policy running: p pause, Enter resume, q end the episode, Ctrl-C emergency release.`
5. The episode ends (120 s timeout, `q`, or `FREEZE`). The client holds the arms for 0.5 s, deactivates the controllers
   it activated and keeps streaming the hold until every controller reads inactive (`Released.`). With
   `takeover.mode: external` it prints `[<side>] deactivate ... now` and keeps holding until you have deactivated it.
   **The client never stops the command stream while a controller reads active**: if a deactivation keeps failing it
   prints `CONTROLLER STILL ACTIVE on ... — deactivate it now or press the E-stop` and holds until the controller reads
   inactive or you press Enter (then exit 5, section 8.8). After `Released.` it asks
   `Outcome? [s]uccess / [f]ail / [x] skip:` and an optional note. Be honest: this is our only label. `x` means "not a
   valid attempt" (for example you pressed `q` immediately).
6. Reset the scene by hand; the next episode starts at step 1.

A `FREEZE` ends the whole `run` command with exit code 3 (during an episode it first asks for a note). To go on, start `run` again with the
same `--run-name`; the episode numbering continues, nothing is overwritten.

### After the session

Section 10 tells you what to send us. If the container was ever killed while it had a controller active, read section 11
first.

## 6. Site settings at a glance

The settings you are most likely to touch. All are keys of `robot_site.munich.yaml`; `--set` takes the same dotted names.

| Key | Default | What it does |
|---|---|---|
| `command_mode` | `absolute` | `absolute`: the command topic carries target joint angles. `relative_to_activation`: targets relative to the pose at activation (see question 1) |
| `joint_signs.<side>` | all `1` | Per-joint direction flip, only used in `relative_to_activation` mode |
| `takeover.mode` | `switch` | `switch`: the client activates/deactivates the controller. `external`: you do it with your own tooling when the client asks (activate at the takeover, deactivate at the end of every episode; the client holds the arm until then; see question 7) |
| `takeover.accept_active` | `false` | `true` lets the client stream into a controller that is already active when it starts (normally refused). Only if we told you so |
| `takeover.max_measured_age_s` | `0.3` | The takeover refuses measured joint states older than this (a stopped state broadcaster) |
| `takeover.hold_inactive_arms` | `true` | With `--arms right`, still take over the left arm and hold it |
| `takeover.tol_rad` | `0.02` | How far the streamed hold may differ from the measured pose at activation |
| `takeover.service_timeout_s` | `5.0` | Timeout of each controller_manager service call |
| `state.pin` | `{7: 0.0}` | State dimensions replaced by a constant (dim 7 is the left gripper) |
| `arms.<side>.joint_names` | `<side>_fr3v2_joint1..7` | Joint names expected on the measured topic |
| `arms.<side>.command_joint_names` | same as `joint_names` | Joint names put on the command topic, if they differ |
| `arms.<side>.start_pose` | medoid of the data | Start pose used by `prepare` and the reset step |
| `arms.<side>.gripper.required` | right `true`, left `false` | Whether that gripper's state stream must exist |
| `cameras.<name>.topic` | see YAML | Camera topics. `type: raw` or `compressed` |
| `cameras.head.train_res` | `[336, 188]` | Head frames are resized to this (W, H) first if the live size differs |
| `cameras.<name>.crop` | none | `[x, y, w, h]` crop in live pixels, applied before everything else |
| `spine.height` | `0.434` | Spine target in metres |
| `spine.enabled` | `true` | `false` skips the spine move entirely |
| `spine.max_delta_m` | `0.2` | `prepare` refuses a spine move longer than this |
| `inference.mode` | `sync` | `sync` or `async` (see below) |
| `inference.execute_horizon` | `30` | Actions executed per chunk (1 to 50) |
| `safety.max_step_rad` | `0.10` | Largest change of one joint between two 20 Hz waypoints |
| `safety.max_track_lag_rad` | `0.5` | The waypoint ramp pauses while target and measured pose differ by more than this |
| `safety.obs_stale_hold_s` | `1.0` | A stream older than this: hold still until it recovers |
| `safety.obs_stale_freeze_s` | `5.0` | A stream older than this: `FREEZE` |
| `safety.measured_dead_s` | `1.5` | An arm's joint-state stream older than this: `FREEZE` |
| `safety.infer_dry_freeze_s` | `5.0` | No action to execute for this long: `FREEZE` |
| `safety.max_start_offset_rad` | `0.3` | The policy's first action may be at most this far from the measured pose |
| `safety.episode_timeout_s` | `120` | Episode length limit |
| `safety.q_margin_rad` | `0.05` | Targets stay this far inside the joint limits |
| `safety.stream_gap_freeze_s` | `0.1` | Command stream silent for this long: `FREEZE` |
| `smoother.velocity_scale` | `0.5` | Fraction of the FR3 velocity limits the smoother allows |
| `ros.rmw`, `ros.domain_id` | `rmw_fastrtps_cpp`, `0` | Must equal the container's `RMW_IMPLEMENTATION` and `ROS_DOMAIN_ID` |
| `server.connect_timeout_s`, `server.infer_timeout_s` | `10.0`, `10.0` | Policy server timeouts |
| `out_dir` | `/out` | Inside the container; leave it and use `OUT_DIR` on the host |

`inference.mode: async` asks for the next chunk in the background while the arms keep moving (merge `blend_favor_old`,
`trigger_threshold` 0.5, `blend_steps` 10, under `inference.async`). It avoids the stop-and-go but is the less tested
option; stay with `sync` unless `check-infer` shows round trips over about 1 s and we told you otherwise.

The policy server address is **not** a YAML key: it comes only from `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT` and
`POLICY_SERVER_API_KEY` (the client rejects them in the YAML).

## 7. Site facts we could not confirm, and how to settle them on site

We could not verify these before the session. Each has a check that tells you the answer and a setting that changes the
behaviour. Please note what you found.

**Q1. How does the arm controller read the command topic?** Absolute joint targets, or targets relative to a reference
pose captured at activation, possibly with per-joint direction signs? We assume absolute.
- Check: `$RC check-motion --arm right` (and left). With `command_mode: absolute`, every joint `follows` means the assumption is right.
- If joints come out `reversed` or `no-motion`: look at the `joint_signs:` line the check prints, then repeat with
  `$RC check-motion --arm right --set command_mode=relative_to_activation --set 'joint_signs.right=[1,-1,1,1,1,1,1]'`
  (use the signs it printed) until every joint says `follows`. If everything stays `no-motion`, the controller does not
  follow our topic at all: check `arms.<side>.command_topic`, `arms.<side>.controller`, and `takeover.mode`.
- Make it permanent for `prepare` and `run` by writing `command_mode` and `joint_signs` into the YAML. Do this for both arms.

**Q2. Is any GELLO or other teleop publisher running?** It must not be.
- Check: any takeover fails with `message(s) from another publisher on /right/gello/joint_states ... is GELLO teleop
  still running?`. A publisher that sends nothing only produces a log warning `other publisher(s) on ... idle`.
- Fix: stop that publisher. There is no setting that makes the client share the topic.

**Q3. What unit and value is the spine height?** The training data has the constant `434` for the spine; we read it as
millimetres, so the YAML says `spine.height: 0.434` (metres). The reference Munich profile uses metres.
- Check: `check-obs` prints the spine's current position (`spine at X m`) and warns `spine.max_delta_m` if it is more than 0.2 m from
  `spine.height`. `prepare` refuses with `spine.height ... m is ... m from the current ... m`.
- Fix: put the spine to the working height you use for the task (data collection height) and set `spine.height` to
  what `check-obs` reads there. Or skip the spine move with `--set spine.enabled=false` if the spine is already right.

**Q4. Head camera: live resolution and framing.** The training head frames are 336x188; the live ZED stream may be larger
(we expect 1280x720). The client resizes a live frame of another size to 336x188, then pads it to 224x224.
- Check: the head row of `check-obs` shows the live size and every step, and `check_obs_head.jpg` shows what the policy sees.
- Knobs: `cameras.head.train_res` (the resize target, only if the training frames were another size) and
  `cameras.head.crop: [x, y, w, h]` (crop of the live frame, in live pixels, if the dataset frames show a different part
  of the view). The default is no crop. Change them only if the saved image clearly differs from the camera's usual view
  in the scene.

**Q5. Joint names.** We expect `<side>_fr3v2_joint1..7` (the dataset itself says `fr3`).
- Check: `check-obs` row `joint(s) [...] not on <topic> (names seen: [...]); check arms.<side>.joint_names`.
- Fix: write the names that are listed after `names seen` into `arms.<side>.joint_names` in the YAML, in joint order 1 to 7.
  If the command topic uses other names than the measured topic, set `arms.<side>.command_joint_names` too. The gripper joint is
  `arms.<side>.gripper.joint`.

**Q6. Where does the policy server run, and can the laptop reach it?**
- Check: `$RC check-infer`. Also, from the laptop: `curl http://<host>:8000/healthz` prints `OK` when the server is ready.
- Fix: `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT`; open the port in the GPU machine's firewall.

**Q7. Can the laptop switch controllers on the robot, or must your own tooling do it?**
- Check: any takeover or `prepare` that fails with `list_controllers on /right/controller_manager failed ... not available
  within 5 s` or `switch_controller ... failed` means the controller_manager's services are not reachable from the
  container (check `arms.<side>.controller_manager`, the ROS domain and the DDS profile first).
- Knob: `takeover.mode: external` (`--set takeover.mode=external`). The client then never calls `switch_controller`.
  Instead it streams the hold pose, prints `[right] activate joint_impedance_controller on /right/controller_manager
  now (...; waiting up to 60 s ...)`, and you activate the controller with your own tooling within 60 s. The controller
  must be `inactive` (or not loaded yet) when the client starts; an already-active one is refused as in switch mode.
  `prepare` and the end-of-episode release work the same way: at the end of an episode the client prints
  `[right] deactivate joint_impedance_controller on /right/controller_manager now (the client keeps streaming the hold
  until it reads inactive; waiting up to 60 s)` and **keeps streaming the hold until the controller reads inactive**, so
  it is never active with nobody commanding it. If it is still active after 60 s the client prints `CONTROLLER STILL
  ACTIVE ...` and keeps holding until it reads inactive or you press Enter (exit 5). The client **still needs to read the
  controller state** (`list_controllers`), so external mode does not help if that service is unreachable too.

## 8. What the messages mean

Every line the client prints when something is wrong, and what to do. Text in `...` stands for values that change from
run to run. Severity: `REFUSED` = exit 2, `CONFIG ERROR` = exit 4, `FREEZE` = exit 3, `FAIL` (a row in a check) = the
check fails (exit 2), `warn` = does not fail anything, `info` = just information.

### 8.1 Startup, configuration and ROS environment

| Message | Meaning | What to do |
|---|---|---|
| `CONFIG ERROR: ... problem(s) in the site configuration:` followed by `  - ...` lines | The YAML or a `--set` value is invalid. Every problem is listed with its exact key. Nothing was started | Fix each listed key; see the rows below |
| `site config file not found: ...` | The `--site` file (default `/config/robot_site.yaml`) is not there | Use the default compose mount, or fix the path of the volume |
| `...: not valid YAML (...)` / `...: the top level must be a mapping, got ...` | The file has a syntax error or is not a key/value document | Fix the YAML (indentation, colons) |
| `override ...: expected dotted.key=yaml_value` | A `--set` argument has no `=` or an empty key | Write `--set section.key=value` |
| `override ...: value ... is not valid YAML (...)` | The value after `=` cannot be parsed | Quote it: `--set 'joint_signs.right=[1,-1,1,1,1,1,1]'` |
| `override ...: '...' is not a section (it is ...)` | The key path goes through a value that is not a section | Check the dotted key against the YAML |
| `...: unknown key (did you mean '...'?)` or `(valid keys: ...)` | A key in the YAML or `--set` does not exist (typo) | Use the suggested name |
| `...: required key is missing` | A mandatory key (for example `topic`, `controller`, `start_pose`) is absent | Add it |
| `...: expected int, got ...` (also `float`, `str`, `bool`) / `expected a list, got ...` / `expected a mapping, got ...` | Wrong type of value | Correct the value's type |
| `...: expected N values, got M` | A list has the wrong length (7 joints, 2 for `train_res`, 4 for `crop`) | Give exactly N values |
| `...: key must be int, got ...` | A key that must be a number is not (for example in `state.pin`) | Use a state dimension number such as `7` |
| `...: must be one of [...], got ...` | Value not in the allowed list (`type`: raw/compressed; `mode`; `impl`; `merge`; `command_mode`) | Use one of the listed values |
| `must be > 0, got ...` / `must be >= 0, got ...` / `must not be empty` / `all limits must be > 0` / `all velocities must be > 0` | A number or text is out of range | Use a positive value or fill the text |
| `must equal the trained prompt '...' (got ...); the policy server would refuse it` | `policy.prompt` was edited | Restore the exact prompt; it cannot be changed |
| `must equal the trained policy rate 20 Hz, got ...` | `policy.control_hz` was edited | Restore 20 |
| `(W, H) must be positive, got ...` / `(x, y, w, h) needs x, y >= 0 and w, h > 0, got ...` | `cameras.<name>.train_res` or `crop` is invalid | Fix the numbers |
| `...: state dim must be in [0, 15]` | `state.pin` names a dimension outside the 16-dim state | Use 0 to 15 |
| `joint names must be unique` | A joint name list repeats a name | Fix `joint_names` / `command_joint_names` |
| `must be in [0.0, 1.0] metres, got ... (a value in mm?)` | `spine.height` looks like millimetres | Give metres (`0.434`); see question 3 |
| `must be in [1, 50], got ...` | `inference.execute_horizon` outside 1 to 50 | Use 1 to 50 |
| `must be in [0.0, 1.0), got ...` / `must be >= 1, got ...` | `inference.async.trigger_threshold` / `blend_steps` out of range | Use values in range |
| `must be in (0.0, 1.0], got ...` | `smoother.velocity_scale` out of range | Use a value in (0, 1] |
| `must be in [0, 232], got ...` | `ros.domain_id` invalid | Use the robot's domain |
| `must be in [1, 65535], got ...` | `server.port` invalid | Use a valid port |
| `must be >= obs_stale_hold_s (...), got ...` | `safety.obs_stale_freeze_s` is smaller than `obs_stale_hold_s` | Make freeze >= hold |
| `keys must be exactly [...] (missing [...]; unexpected [...])` | `cameras`, `arms`, `joint_limits` or `joint_signs` must have exactly these names | Add or remove entries |
| `... not in arms [...]` / `duplicate entries in ...` / `active_arms: must not be empty` | `active_arms` (or `--arms`) names an unknown or repeated arm, or none | Use `left`, `right` or both once each |
| `...: every sign must be 1 or -1, got ...` | `joint_signs` has another value | Use 1 or -1 |
| `q_min and q_max need 7 values each, got ... and ...` / `every q_min must be below its q_max` | `safety.joint_limits` edited wrongly | Fix or remove the override (the FR3 limits are the default) |
| `... is outside the joint limits [...] (safety.joint_limits.<side>)` | A `start_pose` joint is outside the limits (typo, or degrees instead of radians) | Fix `arms.<side>.start_pose` (radians) |
| `stream_gap_freeze_s: ... must be >= 2 / stream_hz (= ... s at stream_hz ...)` | The gap limit is shorter than two command periods | Raise `safety.stream_gap_freeze_s` |
| `server.host: not allowed in the YAML or --set; set the environment variable POLICY_SERVER_HOST` (also `port`, `api_key`) | The server address was written into the YAML or a `--set` | Remove it; use the environment variable |
| `POLICY_SERVER_PORT: expected an integer, got ...` | Bad port value in the environment | Use a number |
| `--run-name must match ^[A-Za-z0-9][A-Za-z0-9._-]*$ (one path component), got ...` | Invalid run name | Use letters, digits, `.`, `_`, `-` only |
| `cannot create the run directory ... (...); mount out_dir or set --set out_dir=...` | The output folder cannot be created (permissions, missing mount) | Fix `OUT_DIR` on the host; do not change `out_dir` inside the container unless you also mount it |
| `python -m robot_client ...: error: ...`; `--joint: must be 'all' or a joint number 1..7, got ...`; `--delta: must be in (0, 0.2] rad, got ...`; `run: error: --episodes must be >= 1, got ...`; `check-infer: error: --n must be >= 1, got ...` | Wrong command-line arguments (exit 4) | Correct the argument; `$RC <command> --help` |
| `REFUSED: RMW_IMPLEMENTATION is ... but ros.rmw is ...; set the environment variable or the ros section of the site YAML` | The container's RMW differs from the YAML. `run`, `prepare` and `check-motion` refuse; the other commands only print `WARNING:` | Leave the image's `rmw_fastrtps_cpp` and the YAML `ros.rmw` equal; do not set `RMW_IMPLEMENTATION` |
| `REFUSED: ROS_DOMAIN_ID is ... but ros.domain_id is ...; set the environment variable (the compose file passes ROS_DOMAIN_ID through: ROS_DOMAIN_ID=<n> docker compose ... run --rm robot-client ...) or the ros section of the site YAML` / `ROS_DOMAIN_ID ... is not an integer (ros.domain_id is ...)` (same hint) | The domain of the container and the YAML differ | Make them equal. The robot is on domain 0. The container gets `ROS_DOMAIN_ID` from your shell (default 0): `export ROS_DOMAIN_ID=<n>` and set `ros.domain_id: <n>` in the YAML |
| `WARNING: FASTRTPS_DEFAULT_PROFILES_FILE=... is not a readable file in the container` | The profile path you set does not exist inside the container | Mount the file (section 4.4) or set it to empty |
| `REFUSED: ROS 2 Python packages are not importable (...); run the client in its container (or source /opt/ros/humble/setup.bash and the franka message workspace)` | The client was started outside its Docker image | Use `$RC ...` |
| a banner of `*` lines reading `POLICY_SERVER_HOST is not set: connecting to localhost:8000.` and `If the policy server runs on another machine, set POLICY_SERVER_HOST (and POLICY_SERVER_PORT / POLICY_SERVER_API_KEY) and rerun.` | The server address is not set | `export POLICY_SERVER_HOST=<ip>` and rerun |

### 8.2 The policy server

| Message | Meaning | What to do |
|---|---|---|
| `REFUSED: cannot reach policy server at <host:port> within 10s: ... Is the policy server running and reachable at <host:port>? Set POLICY_SERVER_HOST / POLICY_SERVER_PORT` (in `check-infer`: row `FAIL server cannot reach policy server at ...`) | No websocket connection, or the server did not answer within `server.connect_timeout_s`. The reason is at the end of the line: `Connection refused` = nothing listens there (server not started, wrong host/port); `timed out` or `No route to host` = network or firewall | Start the server ([SERVER.md](SERVER.md)); verify `curl http://<host>:<port>/healthz`; fix `POLICY_SERVER_HOST` / `POLICY_SERVER_PORT` |
| `server is not a munich16 policy (config=...)` | The server serves a policy without our layout (for example a sim checkpoint) | Ask the server operator to start a `*_munich_real` config |
| `server metadata does not match the munich16 layout (config=...): ...` with parts like `action_dim: expected 16, got ...`, `action_horizon: expected 50, ...`, `prompt: server has ..., client uses ...` | The server serves a policy trained with different dimensions, rate or prompt | Serve the right checkpoint/config. Never edit the prompt |
| `REFUSED: the policy server is not the munich16 policy this client needs: ... Serve a munich16 checkpoint (POLICY_CONFIG) and rerun` | Same, shown by `run` | Same |
| `FAIL server <host:port> refused: ...` | Same, shown by `check-infer` | Same |
| `Error in inference server:` followed by a traceback | The server crashed on a request (server-side error) | Look at the server's terminal; restart the server; send us the traceback |
| `server sent an error instead of its metadata:` | The server failed right after the connection opened | Same |
| `FAIL request i/n failed: ...` | One of the check requests failed (timeout after `server.infer_timeout_s`, connection lost) | Check the server log and network; rerun |
| `FAIL chunk request i: action chunk must be 2-D (T, 16), got ndim=...` / `action chunk width ... != 16: wrong-layout server?` / `action chunk has no rows` / `action chunk contains non-finite values` | The server answered with an unusable chunk | Wrong or broken checkpoint on the server; do not run. Send us the line |
| `warn chunk shape ...: fewer rows than inference.execute_horizon ...` | The chunk is shorter than the number of steps to execute | Lower `inference.execute_horizon` to at most the chunk length (normally 50) |
| `warn latency max round trip ... ms exceeds the budget ... ms (...): the arm will wait for the policy` | Inference plus network takes longer than one chunk lasts | The run still works, with long waits. Tell us the numbers; a GPU on a slower network or a busy GPU is the usual cause |
| `warn observation live snapshot unusable (...); sending zeros` / `ROS not available (...); sending zero observations` / `No live ... after 3 s; sending zero observations` | `check-infer` could not build a live observation and tested the server with zeros | Not a failure of the server check; run `check-obs` to find the missing stream |
| `unusable action chunk from the server: ...` (a `FREEZE`) | During a run, the server sent an unusable chunk | The episode is stopped; restart the server on the right checkpoint |
| log: `runner: inference failed (...); requesting again` | One request failed; the client tries again | If it repeats for 5 s the episode freezes (`inference dry for ...`) |
| log: `inference worker is dead (...); reconnecting to the policy server` | The connection to the server was lost; the client reconnects before the next episode | If the server is up again nothing to do; otherwise see the first row of this table |

### 8.3 `check-obs` rows

| Message | Meaning | What to do |
|---|---|---|
| `FAIL <key> no messages on <topic> in 8 s` (`<key>` is `cam/head`, `cam/wrist_left`, `cam/wrist_right`, `arm/left`, `arm/right`, `grip/right`) | Nothing arrived on that topic | Is the camera driver or state broadcaster running (`$RC ros2 topic list`, `$RC ros2 topic hz <topic>`)? Fix the topic in the YAML (`cameras.<name>.topic`, `arms.<side>.measured_topic`, `arms.<side>.gripper.state_topic`). If **every** stream is missing: same ROS domain? Try the other DDS profile (section 4.4) |
| `info grip/left no messages on ... (optional; state dim 7 is pinned to 0.0)` | The left gripper state is not required | Nothing; this is expected |
| `... (optional; but state dim 7 is NOT pinned: set state.pin.7)` | The left gripper stream is optional but its dimension is not pinned | Set `state.pin: {7: 0.0}` |
| `FAIL <key> N message(s) on <topic>, none usable: ...` | Messages arrive but cannot be decoded. For a camera `unsupported image encoding '...' (supported: rgb8, bgr8, rgba8, bgra8)` names the cause | Camera publishes an unsupported format (for example `mono8`, `16UC1`, depth): change the camera topic to a colour stream. Other texts (`image step ... is smaller than width ...`, `image has N bytes, expected height ... x step ... = ...`, `could not decode the compressed image payload`, `JointState has N names but M positions`) mean a malformed message from the driver: send us the line |
| `warn <key> N undecodable message(s) on <topic>: ...` | Some messages could not be decoded, most are fine | Note it |
| `FAIL <key> last usable sample on <topic> is ... s old (> obs_stale_hold_s 1.0): the stream stopped` | The stream was fine and then stopped during the check | A camera driver died (seen before in Munich). Restart it |
| `FAIL <key> joint(s) [...] not on <topic> (names seen: [...]); check arms.<side>.joint_names` (or `arms.<side>.gripper.joint`) | The joint names in the message differ from the YAML | Question 5 |
| `FAIL <key> cannot preprocess ...: ... (cameras.<name>.crop?)` | The image cannot be processed; typically the crop does not fit inside the live frame (`crop (x, y, w, h)=... does not fit inside the ...x... (WxH) frame`) | Fix or remove `cameras.<name>.crop` |
| `FAIL <key> ...: expected 224x224x3` | The processed image has the wrong shape | Send us the line |
| `FAIL <key> no image in the snapshot although ... delivers` / `no joint positions in the snapshot although ... delivers` / `no gripper value in the snapshot...` | The stream delivers but nothing usable is stored | Send us the transcript |
| `FAIL state ...` with `no <side> gripper sample and state dim ... is not pinned (set state.pin... or make the ... gripper stream required)`, `state dim(s) [...] are not finite: [...]`, `no measured joint positions for the ... arm`, `... arm has ... joint positions, expected (7,)` | The 16-dim state could not be built | Fix the stream above it; the dimension numbers are state positions 0 to 15 (left 0-6, left gripper 7, right 8-14, right gripper 15) |
| `info state not built (an arm stream is missing)` | Consequence of a failed arm row | Fix the arm row |
| `warn <key> rate ... Hz on <topic> is below 80 % of the expected ... Hz` | The stream is slower than expected | Section 9 for the arms; for cameras, check the driver and the network |
| `warn <key> could not save ...` | The check image could not be written | Check `OUT_DIR` permissions |
| `warn state range dim N: value ... outside the training range [..., ...]` / `dim N: value ... is not finite` / `cannot check against ...: ...` | Only with `--norm-stats`: the robot's current pose is outside what the policy saw in training, or the file is unusable | Informational; note which dimensions. If the file is for another layout (`norm stats cover ... state dims but the state has 16`) use the right file |
| `warn spine position unavailable (/franka_spine_node/get_position); prepare would refuse` / `reading the position failed: ...` (log: `spine position unavailable: service ... not available within 5 s` / `spine position unavailable: ... reported failure`) | The spine node's service did not answer | Is the spine node running? Otherwise set `spine.enabled: false` |
| `warn spine at ... m, spine.height ... m (delta ... m) > spine.max_delta_m ...: prepare would refuse; check the height and its unit` | Question 3 | Question 3 |
| `WARNING: prepare needs the franka_msgs and franka_spine_msgs Python packages ...; the spine position is not checked` | The Franka message packages are missing from the image | Rebuild the image; `prepare` will not work without them |

### 8.4 `check-motion`

| Message | Meaning | What to do |
|---|---|---|
| `joint N follows: commanded +0.050 rad, measured ... rad (ratio ...), other joints moved <= ... rad` | `ok` row; the joint followed | Nothing |
| `FAIL joint N reversed: ...` | It moved the opposite way | Question 1 |
| `FAIL joint N no-motion: ...` | The joint did not move | Question 1: is the controller following this command topic? |
| `FAIL joint N unclear: ...` | The ratio was between the accepted bands | `--delta 0.1` |
| `info joint_signs  joint_signs:  <side>: [...]   (or --set joint_signs.<side>=[...])` | The signs the measured motion suggests | Question 1 |
| `info hint joint_signs only apply with command_mode relative_to_activation; a reversed joint in absolute mode suggests the controller reads commands relative to activation: retry with --set command_mode=relative_to_activation and the signs above` | A reversed joint in absolute mode | Question 1: do what the hint says |
| `info hint no joint moved: is the controller really following the command topic? Check arms.<side>.command_topic, takeover.mode and command_mode` | Nothing moved at all | Check those three settings; is the controller really active? |
| `info hint unclear joints moved too little or too much; rerun with a larger --delta` | See `unclear` | `--delta 0.1` |
| `warn joint N no room to move inside the joint limits; skipped` | The joint is at a limit | Move the arm away from its limit (or accept the skip) |
| `FAIL aborted joint N is ... rad from where the check started (> 2 * delta + 0.02 = ... rad); the arm was released` | The arm moved far more than commanded. The check stopped and released | Do **not** continue. Question 1 (a wrong command mode can do this); send us the transcript |
| `FAIL aborted no measured joint states for <side>; the arm was released` | The state stream stopped | Section 8.3 |
| `FAIL aborted the command stream is unhealthy (...); the arm was released` | Our command stream stopped | Section 9 |
| `FAIL takeover refused: ...` | The takeover failed | Section 8.6 |

### 8.5 `prepare` and the reset step

| Message | Meaning | What to do |
|---|---|---|
| `REFUSED: prepare stopped at '<step>' (see above); fix the cause and rerun` preceded by `prepare: stopped at '<step>'` and `FAIL <step>: ...` | A step failed; the later steps did not run. `<step>` is `check command stream`, `deactivate left/right`, `ptp left/right`, `spine` or `grippers` | Read the `FAIL` line, see the rows below |
| `the joint command stream is running; prepare refuses to move the arms while it publishes` | Internal guard | Send us the transcript |
| `switch_controller refused to deactivate <controller> on <controller_manager>` / `<controller> still active after switch_controller` | The controller could not be deactivated before the PTP move | Deactivate it with your own tooling, or use `takeover.mode: external` (question 7) |
| `<controller> still active after 60 s; deactivate it and retry` and the prompt `[<side>] deactivate <controller> on <cm> now (prepare moves the arms with PTP; waiting up to 60 s, Ctrl-C aborts)` | `takeover.mode: external`: the client waits for you to deactivate the controller | Deactivate it within 60 s |
| `PTP to start_pose failed or timed out (/<side>/action_server/ptp_motion, timeout 60 s)` | The PTP action did not succeed. The log lines above show why: `<action>: action server not available within ... s`, `<action>: goal rejected`, `<action>: PTP failed (goal status ..., target status ...): <robot's message>`, `<action>: no result within ... s; canceling the goal`, `<action>: goal not accepted within ... s; canceling it if it is accepted later`, `<action>: goal cancel not acknowledged within ... s`, `<action>: canceling a goal accepted after we stopped waiting` | Check the action name `arms.<side>.ptp_action` and that the robot is enabled and not in an error state; check `start_pose` is reachable; try again; raise `prepare.timeout_s` if the move is just slow |
| `spine position unavailable (/franka_spine_node/get_position); not moving the spine. Check the spine node, or set spine.enabled: false` | The spine position service does not answer | Check the spine node, or `--set spine.enabled=false` |
| `spine.height ... m is ... m from the current ... m (> spine.max_delta_m 0.2); check the height and its unit (metres)` | Safety limit on the spine move length | Question 3 |
| `spine move to ... m failed or timed out (/franka_spine_node/move_absolute, timeout 60 s)` | The spine action failed. Log: `<action>: spine move failed (goal status ..., stop_by ...): ...` or the action-server lines of the PTP row | Check the spine node and `spine.action`; the robot may have refused the height |
| `REFUSED: prepare needs the franka_msgs and franka_spine_msgs Python packages (franka_ros2 b4164f5), which the robot_client image builds and sources; they are not importable here (...)` | The Franka message packages are not in the image | Rebuild the image |
| `Nothing was moved.` | You answered `q` at a prompt | Nothing |

### 8.6 Takeover (activating the controllers)

All of these appear as `REFUSED: takeover refused: <side>: ...` (exit 2) in `run`, as `FAIL takeover refused: ...` in
`check-motion`. Nothing is left activated: whatever the client activated was deactivated again (if that fails, see
`CONTROLLER STILL ACTIVE` in section 8.8). The foreign-publisher, already-active, not-loaded / wrong-state and measured-joint rows are checked before the
client publishes anything.

| Message | Meaning | What to do |
|---|---|---|
| `...message(s) from another publisher on <topic> in the last 1 s: is GELLO teleop still running? Stop it and retry.` | Someone else publishes on the arm's command topic | Question 2 |
| ``controller joint_impedance_controller on <cm> is already active; run `release --arm <side>` (or deactivate it with your tooling), then retry. The client never streams into a controller that was active before it started (set takeover.accept_active: true only if you know it follows absolute targets)`` | The controller was already active (left on by bring-up, by external activation, or after a failed release). Our first command would act on it immediately, so nothing was published | `$RC release --arm <side>` (or your tooling), then retry. Do not set `takeover.accept_active` unless we told you |
| `no measured joint states on <topic>; is the robot state broadcaster running?` / `no measured joint states for [...] (...); not starting the command stream` | The arm's state stream is silent | `check-obs`; `arms.<side>.measured_topic` |
| `the measured joint states on <topic> are ... s old (> takeover.max_measured_age_s 0.3); is the robot state broadcaster still running? Not taking over.` | The arm's state stream stopped some time after the client started (for example while the reset prompt was open) | Check the state broadcaster / robot (`$RC ros2 topic hz <topic>`), then retry |
| `list_controllers on <controller_manager> failed: ...` (for a timeout the end reads `service /right/controller_manager/list_controllers not available within 5 s; is the controller_manager running?`) | The controller_manager service did not answer within `takeover.service_timeout_s` | Question 7. Check `arms.<side>.controller_manager` (the namespace) and the DDS settings |
| `controller '<name>' is not loaded on <controller_manager> (available: ...). Load and configure it with your tooling, fix arms.<side>.controller, or set takeover.mode: external.` | The controller does not exist on that manager. `available:` lists what does | Use the right name in `arms.<side>.controller`, or have the robot side load it |
| `controller '<name>' is '<state>' on <cm>; it must be 'inactive' to be activated. Load and configure it with your tooling (e.g. ros2 control set_controller_state <controller> inactive -c <cm>), or set takeover.mode: external.` | The controller is loaded but not configured (state `unconfigured`) or in another state | Configure it with your tooling so that it is `inactive`, then retry |
| log: `<side>: <controller> is already active; takeover.accept_active: streaming into it` / `<side>: <controller> is already active (not activated by us; takeover.accept_active); streaming to it` | Only with `takeover.accept_active: true`: the client streams into a controller it did not activate | At the end of the episode the client asks you to deactivate it (`[<side>] deactivate ... now`) and holds the arm until you have |
| log: `<side>: <controller> is already active; streaming to it` | `takeover.mode: external`: you activated the controller while the client streamed the hold (normal) | Nothing |
| `the streamed hold differs from the measured joints by ... rad <when> (> takeover.tol_rad 0.02); the arm moved since the stream started. Let it settle and retry.` | At activation the arm must stand still at the pose we stream. It did not (somebody pushed it, it is vibrating, or a settling motion after `prepare`) | Wait a few seconds and retry. If it repeats on a still arm, raise `takeover.tol_rad` a little (not above about 0.05) and tell us |
| `the command stream is unhealthy (...); not activating` / `the command stream faulted while warming up: ...` / `only N/20 hold commands published after ... s; the command stream is not running` / `the command stream is unhealthy after takeover (...)` | Our own 100 Hz publisher is broken or too slow | Section 9; send the transcript |
| `switch_controller (activate <controller>) on <cm> failed: ...` / `switch_controller refused to activate <controller> on <cm>` | The controller manager rejected or did not answer the activation | Question 7 (`takeover.mode: external`); look at the robot side for why |
| `<controller> not confirmed active within 1.0 s after switch_controller (state: ...)` | The activation was accepted but the state did not become `active` | The client deactivates again; retry; use question 7 if it repeats |
| `[<side>] activate <controller> on <cm> now (the hold is streaming; waiting up to 60 s, Ctrl-C aborts)` | `takeover.mode: external`: waiting for you to activate | Activate the controller within 60 s |
| `<controller> not active after 60 s (state: ...); activate it with your tooling and retry` | Same, timed out | Retry and activate faster |
| `refusing to stream hold = measured: hold['<side>'] outside the joint limits: joint i = ... not in [..., ...]` | The arm stands outside the joint limits (the joint index here counts from 0) | Move the arm (hand-guide or your own tooling) inside the limits |
| log: `<side>: N other publisher(s) on <topic>, idle (no messages in the last 1 s); continuing. If GELLO teleop is running, stop it.` | A teleop process exists but sends nothing | Stop it anyway |
| log: `<side>: activation failed after switch_controller; deactivating <controller>` | An activation went wrong; the client is undoing it | Watch for `release:` errors below |

### 8.7 During a run: start gate, HOLD and FREEZE

Before the start, `Waiting for the streams: ...` (the arms keep holding) and `Not starting: ...` (you pressed Enter too
early) tell you a stream is stale; they clear on their own. During the episode, the same causes first make the arms
**hold** (`runner: HOLD (...)` in the log: the arms stop, the plan is dropped, the client re-plans when the stream is back);
if the problem lasts they **freeze**. A `FREEZE` stops the episode, holds the arm briefly, deactivates the controllers, and
prints `FREEZE: <reason>` (before the start: `FREEZE before the start: <reason>`). The `<reason>` is one of:

| Reason text | Meaning | What to do |
|---|---|---|
| `cam/<name> age ... s > obs_stale_hold_s 1.0` | Hold only: that camera had no new frame for more than 1 s | Wait; if it does not recover it becomes a freeze. Camera driver problem |
| `cam/<name> age ... s > obs_stale_freeze_s 5.0` | A camera was silent for over 5 s | Restart that camera driver; run `check-obs` |
| `arm/<side> age ... s > obs_stale_hold_s 1.0` | Hold only: the arm's joint states are late | See rates in `check-obs`; section 9 |
| `arm/<side> age ... s > measured_dead_s 1.5` | The arm's measured joint states stopped for over 1.5 s | The state broadcaster died or the network dropped; check the robot. Never ignore this one |
| `grip/<side> age ... s > ...` | The gripper state stream is late or stopped (only the required one, right) | Check the gripper driver |
| `<key> never seen` (in place of `age`) | That stream never delivered a usable sample | `check-obs` |
| `inference dry for ... s > infer_dry_freeze_s 5.0` | No action was available for over 5 s: the server answers too slowly or not at all | Check the server and network; `check-infer` |
| `fatal inference error (...)` / `fatal inference error (inference worker thread is not running)` | The inference thread died; the text in brackets says why. A wrong-layout server also ends here | Restart the server or fix the network; if the text says `munich16` the wrong checkpoint is served |
| `command streamer unhealthy: <fault or 'publishing stopped'>` (the fault is an error text such as `ValueError: right: invalid command [...]`, `publish failed for right: ...`, `margin ... leaves no room inside the joint limits` when `safety.q_margin_rad` is set far too large; the log then also shows `streamer tick failed; republishing the last commands until stopped`, `streamer loop failed; republishing the last commands until stopped` or `streamer: republishing the last command for <side> failed`, each with a traceback) | The 100 Hz command publisher raised an error or paused for more than `safety.stream_gap_freeze_s` (0.1 s). While faulted it keeps republishing the last command until the release | Section 9. Send us the transcript |
| `first action of the <side> arm is ... rad from measured > max_start_offset_rad 0.3` | The first chunk starts too far from where the arm is, so following it would make a jump | Move the arm closer to a typical start pose (let `prepare` do it), then retry; do not raise the limit without us |
| `cannot build the observation: ...` (for example `snapshot has no image for camera(s) [...]; have [...]`, or a state error as in `check-obs`) | An image or state could not be prepared during the run | Same causes as in `check-obs` |
| `unusable action chunk from the server: ...` | See section 8.2 | |

Other lines during a run:

| Message | Meaning | What to do |
|---|---|---|
| `Episode N ended by timeout / operator / freeze after M steps: <reason>` (the log also has `episode ended by ... after ... steps: ...`) | Why the episode ended. `timeout` reason: `episode_timeout_s 120.0 reached`; `operator`: `operator pressed q` | Nothing |
| log: `<side>: tracking lag > 0.50 rad, waypoint ramp paused` / `tracking lag back within 0.50 rad, ramp resumed` | The arm is not keeping up with the target; the client waits for it. Normal in `--dry-run` | If it happens in a real run all the time, the arm is not following (see question 1) or is slowed by the robot's own limits |
| log: `runner: tick N finished ... s late; re-anchoring` | The 20 Hz loop ran late | Occasional lines are fine; many lines mean the laptop is overloaded (section 9) |
| log: `<side>: smoother output ... outside the joint limits; clamped` | The smoother wanted to exceed a joint limit and was clamped | Tell us |
| log: `runner: paused by the operator (Enter resumes)` / `runner: resumed by the operator` | Your `p` and Enter | Nothing |
| `Waiting for the streams: ...` and `Not starting: ...` | See above | |
| `Quit.` | You pressed `q` at the start prompt | Nothing |

### 8.8 Releasing

| Message | Meaning | What to do |
|---|---|---|
| `Released.` | Every controller the client streamed to reads inactive (the client deactivated the ones it activated, you deactivated the others when asked) and the stream stopped | Nothing |
| `[<side>] deactivate <controller> on <cm> now (the client keeps streaming the hold until it reads inactive; waiting up to 60 s)` | The release reached a controller the client did not activate (`takeover.mode: external`, or `takeover.accept_active`). The arm keeps holding | Deactivate it with your tooling now |
| `[<side>] deactivate <controller> on <cm> now (the command stream is unhealthy (<reason>): the client may not be holding the arm; waiting up to 60 s for it to read inactive)` | Same, but our command stream has failed (the release follows a `FREEZE` for `command streamer unhealthy`): the arm may be holding its last command or nothing | Deactivate it with your tooling now; **keep your hand on the E-stop** |
| A block of `!` lines: `CONTROLLER STILL ACTIVE on <cm> (<controller>) — deactivate it now or press the E-stop.` / `The client keeps streaming the hold until it reads inactive. Press Enter once you have handled it.` | The client retried its own deactivation for 5 s (or waited 60 s for yours) and the controller still reads active (or its state cannot be read). The arm is still held by the stream | Deactivate it with your tooling (from another terminal: `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client release --arm <side>`) or press the E-stop. The client stops by itself once the state reads inactive; press Enter only when you have handled it. Do **not** `docker stop` this client: it ignores SIGTERM while it holds, so Docker kills it after 10 s and the stream stops with the controller still active |
| The same block with `The command stream is unhealthy (<reason>): the client may NOT be holding the arm. Press Enter once you have handled it.` as its second line | As above, but our command stream has failed, so the arm may not be held | Same; press the E-stop if the arm moves |
| ``CONTROLLER STILL ACTIVE: <controller> on <cm> may still be active after the release (not confirmed inactive); the command stream has stopped. Deactivate it with your tooling (or `release --arm <side>`) or press the E-stop`` (exit 5) | You pressed Enter at the message above while the controller did not read inactive | Make sure it is deactivated (section 11) before you run anything else |
| log: `reading the operator's confirmation failed` (with a traceback) | The terminal could not be read while the client waited for your Enter | It keeps holding until the controller reads inactive; deactivate it with your tooling |
| `releasing — wait, or use the E-stop` | You sent another interrupt during a release | Wait a few seconds. Use the E-stop if the arm is doing something wrong |
| `Interrupted. Everything this client activated has been released (see above).` | After Ctrl-C, SIGTERM or a closed terminal (exit 130) | Nothing. If `release:` lines appeared above, see below |
| `release: <side>: switch_controller refused to deactivate <controller> on <cm>` / `<controller> still active after deactivation (state: ...)` / `deactivating <controller> on <cm> failed: ...` / `deactivating <controller> interrupted (KeyboardInterrupt); retrying` / `hold interrupted (KeyboardInterrupt); continuing the release` / `hold failed: ...` / `stopping the command stream failed: ...` / `resetting the command adapter failed: ...` | Something went wrong in the release. The client keeps streaming the hold and retries while any controller still reads active (a `CONTROLLER STILL ACTIVE` block follows if that does not help); if `Released.` follows, every controller reads inactive | If no `Released.` follows: **run `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client release --arm left --arm right`** (from the openpi root), or deactivate with your own tooling, or use the E-stop |
| `REFUSED: release failed for [...] (see above); use the robot's own tooling or the E-stop` and the lines `<side>: cannot read the state of <controller> on <cm>: ...`, `<side>: deactivating <controller> on <cm> failed: ...`, `<side>: switch_controller did not deactivate ... (state: ...)` | The `release` command could not deactivate | Use the robot's own tooling or the E-stop |
| `<side>: <controller> on <cm> is <state>; nothing to deactivate` / `<side>: ... deactivated (now ...)` | The `release` command's normal output | Nothing |

### 8.9 Other

| Message | Meaning | What to do |
|---|---|---|
| A Python traceback (exit code 1) | A bug in the client, not designed behavior | If an arm may be active, run `docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client release --arm left --arm right` from the openpi root. Send us the transcript |
| `recorder: ... write failed (...); ... recording disabled` / `recorder: rejecting chunk of shape ... (expected ...)` / `recorder: events.log write failed; event logging disabled` / `recorder: final chunks.npz write failed` / `recorder: result.json write failed` | A log file could not be written (disk full or permissions). The run itself continues | Free disk space / fix `OUT_DIR`; the diagnostics for that episode are incomplete |
| `WARNING: could not save the report to ...: ...` | A check result file could not be written | Same |
| `Please answer Enter / k / q.` (or `Enter / q`, `s / f / x`) | You typed something that is not one of the choices; the prompt repeats | Type one of the listed answers |
| `<end of input>` | The terminal input was closed (stdin ended). Every prompt then answers `quit` or `skip` | Run the client in a normal interactive terminal |
| log: `runner: streamer.hold() at the end of the episode failed` / `runner: could not preprocess images for the recorder` / `streamer thread did not stop within 1 s` / `ROS executor stopped` / `recording event ... failed` (each with a traceback) | An internal error that does not stop the run by itself; `ROS executor stopped` means ROS message handling died, after which streams go stale and the episode freezes | Send us the transcript |
| `result.json` field `recorder_errors` listing `chunks: rejected shape ...`, `steps: close ...`, `chunks: final save ...` or `stream_stats: ...`; log `recorder: stream_stats.json write failed` | Diagnostics of that episode are incomplete | Free disk space / fix `OUT_DIR`; tell us |
| `sequence size exceeds remaining buffer` (printed by Fast DDS itself, not by the client) | Another participant on the DDS network sent a message that could not be decoded. We saw it on our lab network without any effect on the checks | Ignore unless streams are missing at the same time; then try the other DDS profile (section 4.4) |
| `service ... not available within ... s` / `service ... did not answer within ... s` / `service ... returned no response` (inside a longer message) | A ROS service on the robot did not answer in time | Is that node running, same ROS domain? |

### 8.10 Normal output (nothing to fix)

| Line | When |
|---|---|
| `Listening to the robot streams for 8 s ...` | `check-obs` is collecting |
| rows with `ok` or `info`: `7 joints matched, ..., q = [...]`, `shape (50, 16) (horizon x 16 munich16 actions)`, `live snapshot from the robot streams`, `zeros (no live observation)`, `every dim inside the training range`, `not checked (no norm stats given)`, `not checked (spine.enabled: false)`, `position not checked (no motion backend)`, `16 dims, pinned {7: 0.0}` | Checks |
| `check-motion takes over the right arm (joint_impedance_controller) and moves joint(s) 1-7 by 0.05 rad and back, one at a time (~4 s per joint).` followed by `Workspace clear, E-stop at hand? [Enter] start / [q] quit:` | Start of `check-motion` |
| `Workspace clear? [Enter] move both arms to the start pose / [q] quit:` | Start of `prepare` |
| `Policy server OK: munich16 / <config>` | `run` connected to the server |
| `=== episode N ===` / `=== dry-run episode N (nothing moves; preview topics only) ===` | Start of an episode |
| `Run finished: N episode(s) in <folder>` | End of `run` |
| `saved <path>` | A check wrote its JSON file |

## 9. Keeping the command stream at 100 Hz

The arms are driven by a 100 Hz command stream. If it stalls for more than 0.1 s (`safety.stream_gap_freeze_s`) the client
freezes with `command streamer unhealthy: publishing stopped`. The stream runs in the same process as the camera
decoding, so a busy laptop (other programs, a screen recorder, a heavy desktop) can delay it. What to do:

- Close other heavy programs on the laptop before `run`.
- In `check-obs`, look at the `arm/*` rows: the rate should be 100 Hz or more. Camera rows should be near 20 / 30 / 30 Hz.
- During a `run --dry-run` you can watch the stream rate directly. In a second terminal:

  ```bash
  $RC ros2 topic hz /right/robot_client_preview/joint_states
  ```

  It should report about 100 Hz with a small spread. (In a real run the same command on `/right/gello/joint_states`.)
- After a run, `episode_NNN/stream_stats.json` records `streamer.max_gap_s` per arm, the longest silence of the command
  stream. It should stay far below 0.1. A value above 0.05 means the laptop is struggling; tell us.
- Last resort, only if a freeze keeps happening with a gap just above 0.1 s: raise `safety.stream_gap_freeze_s` slightly
  (for example 0.2). This weakens a safety check, so only do it with the E-stop in hand and tell us.

## 10. What to send us

When the session is over:

```bash
sudo chown -R "$USER": examples/ebim_task2/robot_runs        # or your OUT_DIR
tar czf munich_robot_runs.tar.gz -C examples/ebim_task2 robot_runs
```

With a custom `OUT_DIR=/some/absolute/path`:

```bash
sudo chown -R "$USER": /some/absolute/path
tar czf munich_robot_runs.tar.gz -C "$(dirname /some/absolute/path)" "$(basename /some/absolute/path)"
```

Send `munich_robot_runs.tar.gz`, the `script` transcript (section 4.3), and a few notes: the answers to the questions in
section 7, which checkpoint the server ran (it is in `server_metadata.json` and the `check-infer` output), and anything
you saw that the numbers do not show (for example the arm hesitating, jerking, or a camera that looked wrong).

What is in a run folder `robot_runs/<run name>/`:

| File | Content |
|---|---|
| `check_obs.json`, `check_obs_<camera>.jpg`, `check_infer.json`, `check_motion_<side>.json` | Results of the checks (the three JPEGs are what the policy sees) |
| `config.yaml` | The exact settings used (the API key is masked) |
| `server_metadata.json` | What the policy server said it serves: config, checkpoint, layout |
| `events.log` | State changes, takeover/release events, refusals |
| `episode_NNN/steps.jsonl` | 20 Hz record of measured pose, actions, queue and safety mode |
| `episode_NNN/chunks.npz` | Every action chunk the server returned, with request/receive times |
| `episode_NNN/stream_stats.json` | Rates and largest gaps of every stream and of our command stream |
| `episode_NNN/images/<camera>/` | The policy's input images, 2 per second |
| `episode_NNN/result.json` | Your outcome (`s`/`f`/`x`, or `freeze`) and note |

## 11. If the container was killed while an arm was active

If the container died hard (SIGKILL, `docker kill`, a power cut, the laptop crashed) while a controller was active, the
client did not get to release it. The arm's impedance controller may still be active and nobody is commanding it. Press the
E-stop if anything moves, then run, from a new terminal in the openpi root (this does not depend on `$RC` or `POLICY_SERVER_HOST`; it never talks to the policy server):

```bash
docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client release --arm left --arm right
```

Run the same command after a `CONTROLLER STILL ACTIVE` exit (code 5), or before retrying when a takeover was refused with
`... is already active`.

It deactivates the configured controller of each arm if it is active (even if this client did not activate it) and prints
what it did per arm (`<side>: <controller> on <cm> deactivated (now inactive)`). Exit code 0 means both are
deactivated; exit code 2 prints why not. In that case use the robot's own tooling or the E-stop. A normal Ctrl-C or
`docker stop` does not need this: the client releases on its own, and prints `Interrupted. Everything this client
activated has been released`.

**Not during a `CONTROLLER STILL ACTIVE` hold.** While that block is on screen the client is in its release and ignores
Ctrl-C and SIGTERM (it prints `releasing — wait, or use the E-stop`). `docker stop` then waits 10 s and kills the client
with SIGKILL, so the hold stream stops while the controller is still active. Instead, leave the client running and, from
a second terminal, run the `release --arm <side>` command above for the arm it names (or deactivate it with your own
tooling, or press the E-stop). The client sees the controller inactive and finishes by itself; press Enter in its
terminal only if it keeps waiting after you handled it.
