# EBiM task 2 policy server: start, check, swap

For whoever starts the policy server on the GPU machine for the Munich diagnostic run. The robot client (see
[README_ROBOT_CLIENT.md](README_ROBOT_CLIENT.md)) connects to this server over the network and asks it for action chunks.
The server needs no robot and no ROS.

What it serves: one trained openpi pi0.5 checkpoint for the Munich data (layout `munich16`: 16-dim state, 16-dim action,
50-step chunks at 20 Hz). The server tells every client its layout, config name and checkpoint name when it connects, and
the client refuses a server whose layout or prompt does not match.

## 1. What you need

- A Linux machine with an NVIDIA GPU, Docker with the `docker compose` plugin, the NVIDIA driver and the NVIDIA Container
  Toolkit (so `docker run --gpus all ...` works).
- **GPU memory:** the model needs about 8 to 10 GB (bf16 inference); an L4 (24 GB) is fine. JAX reserves about 75 % of the
  GPU's memory when it starts, so `nvidia-smi` will show far more than 10 GB in use. That is expected. No other large
  GPU job should run on the same GPU.
- The openpi checkout at the commit we gave you (the same one as for the client). **The server container does not contain
  the source code: it bind-mounts the checkout at `/app` and runs `uv run scripts/serve_policy.py` there.**
- Internet access for the first build and the first start (see [section 6](#6-offline-use-not-yet-verified)).
- The checkpoint we name in the hand-over message, as three words: `<config>`, `<exp>`, `<step>`. The two configs we
  ship are `pi05_lora_ebim_task2_munich_real` (LoRA, the stand-in) and `pi05_ebim_task2_munich_real` (full fine-tune).

## 2. Get the checkpoint

The checkpoints are in the Hugging Face repository `Ching-Yeh/ebim-task2`, laid out as
`<config>/<exp>/<step>/{params,assets}` (the same layout as openpi's own `checkpoints/` folder; no `train_state/`).
From the root of the openpi checkout:

```bash
pip install -U "huggingface_hub[cli]"      # once; newer versions call the command `hf`
huggingface-cli download Ching-Yeh/ebim-task2 \
    --include "<config>/<exp>/<step>/*" --local-dir checkpoints
```

For example (replace with the values from the hand-over message):

```bash
huggingface-cli download Ching-Yeh/ebim-task2 \
    --include "pi05_ebim_task2_munich_real/munich_v1/30000/*" --local-dir checkpoints
```

The checkpoint directory is then `checkpoints/<config>/<exp>/<step>`. Check that it contains `params/` and `assets/`,
and that `assets/ebim/task2_munich_real/norm_stats.json` exists. The `assets` folder holds the normalisation statistics
the policy needs; without it the server fails or, worse, gives wrong actions. `holdout_eval.json` next to them is our
offline quality report, not needed for serving.

## 3. Start the server

```bash
examples/ebim_task2/serve_munich.sh <config> <ckpt_dir> [port]
```

for example

```bash
examples/ebim_task2/serve_munich.sh pi05_ebim_task2_munich_real checkpoints/pi05_ebim_task2_munich_real/munich_v1/30000
```

- `<config>` **must be the config the checkpoint was trained with** (the directory name of the checkpoint's first level).
  A LoRA checkpoint with the full-FT config name, or the reverse, fails to load.
- `<ckpt_dir>` is the step directory (it contains `params/` and `assets/`). The script checks that and stops with an
  error otherwise. It is mounted read-only at `/ckpt` in the container.
- `[port]` defaults to 8000. The server listens on all interfaces (host networking); open that TCP port for the robot
  laptop in the firewall.
- The first start builds the image (`scripts/docker/serve_policy.Dockerfile`, a CUDA 12.2 base image, uv and the Python
  dependencies), which takes several minutes and needs internet. The script runs `docker compose up --build` in the
  foreground, so the log is on your terminal and Ctrl-C stops the server. To run it in the background add `-d`:
  `COMPOSE_UP_ARGS="--build -d" examples/ebim_task2/serve_munich.sh ...`, and read the log with
  `docker logs -f <container>` (`docker ps` shows the name).

Without the wrapper script (this is what it runs):

```bash
POLICY_CONFIG=<config> CKPT_DIR=$PWD/checkpoints/<config>/<exp>/<step> PORT=8000 \
  docker compose -f examples/ebim_task2/compose.policy_server.yml up --build
```

Both `POLICY_CONFIG` and `CKPT_DIR` are required; docker compose stops with `required variable ... is missing a value`
otherwise. The compose file also mounts `~/.cache/openpi` (override with `OPENPI_DATA_HOME`) at `/openpi_assets` for the
downloaded tokenizer.

## 4. Expected log lines and the health check

The server logs, in this order (the exact numbers differ):

| Log line | Meaning |
|---|---|
| `Loading model...` | Reading `params/` (about a minute) |
| `Warming up policy (compiling the model, this can take a while)...` | One throwaway inference to compile the model (about 40 s) |
| `Warmup done in ... s` | **Ready.** The server now listens |
| `Creating server (host: ..., ip: ...)` | Followed by nothing until a client connects |
| `Connection from ('<client ip>', <port>) opened` | A client connected (also printed again for each reconnect) |
| `Policy warmup failed; the first inference request will be slow.` with a traceback | The model compiled badly. Look at the traceback (most often memory: see section 7) |

If the log stops after `Loading model...` with a long traceback, the checkpoint or config is wrong (section 7).

Check that it answers, from any machine that can reach it:

```bash
curl http://<server ip>:8000/healthz        # prints: OK
```

(the HTTP check answers only once the server is up, after the warm-up). Then run the full check from the robot laptop,
which is what matters:

```bash
$RC check-infer          # see README_ROBOT_CLIENT.md, section 4.2
```

`$RC` is the robot client shortcut defined in [README_ROBOT_CLIENT.md, section 4.2](README_ROBOT_CLIENT.md#42-run-any-command)
(`RC="docker compose -f examples/ebim_task2/compose.robot_client.yml run --rm robot-client"`, run from the openpi root on
the robot laptop, with `POLICY_SERVER_HOST` set to this machine).

Required: `check_infer: PASS`; the `server` row shows `layout munich16`, the config name and the checkpoint name
(`<exp>/<step>`); the chunk shape is `(50, 16)`. Typical inference time on an L4 was about 0.5 to 0.6 s per request in an
evaluator's measurement of a different pi0.5 (round trip 0.59 s); we expect similar or better after the warm-up.
`check-infer` warns if the slowest request is longer than the 1.5 s one executed chunk lasts.

## 5. Swapping the checkpoint

Swapping is purely a server-side change. **The robot client does not change**: no rebuild, no YAML edit, and it does not need
to be restarted between episodes (it reconnects, checks the layout again, and records the new checkpoint name in
`server_metadata.json` when `run` starts).

1. Stop the server: Ctrl-C in its terminal, or `docker stop <container>` (find it with `docker ps`). Note: `docker compose
   ... down` only works if `POLICY_CONFIG` and `CKPT_DIR` are set to anything, as compose checks them first.
2. Download the other step ([section 2](#2-get-the-checkpoint)) if you do not have it.
3. Start it again with the other `<config>` and `<ckpt_dir>` ([section 3](#3-start-the-server)). Wait for `Warmup done`.
4. From the laptop: `$RC check-infer` and look at the `checkpoint` name.

Do not swap while an episode is running: the client would reconnect at the next request, and an episode in progress would
freeze (`inference dry for ...`) while the server is down.

## 6. Offline use: not yet verified

What we know from the files, not from a test:

- The PaliGemma tokenizer is downloaded on the first start into `OPENPI_DATA_HOME` (default `~/.cache/openpi`, mounted at
  `/openpi_assets`) as `big_vision/paligemma_tokenizer.model`. **On a GPU machine without internet, put that file there
  in advance.** On any machine with internet, either fetch it directly:

  ```bash
  mkdir -p ~/.cache/openpi/big_vision
  curl -L -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model \
      https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
  ```

  or, inside the openpi checkout (after `uv sync`), run the one-liner that downloads it the same way the server does:

  ```bash
  uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://big_vision/paligemma_tokenizer.model', gs={'token': 'anon'}))"
  ```

  and copy the file to the same relative path on the server machine.
- The image build (`serve_policy.Dockerfile`) installs the dependencies with `uv sync --frozen --no-install-project
  --no-dev` and needs internet. The container's command is `uv run scripts/serve_policy.py ...`, and `uv run` re-syncs the
  environment with the project (which the build did not install) and the default dev dependency group (which the build
  excluded). **We therefore expect the first start of a new container to need access to the Python package index, and we have
  not verified that it works without it.** Plan for network access on the day, and do the first start with internet well
  before the session so you can see `Warmup done`.
- Changing `POLICY_CONFIG` or `CKPT_DIR` makes compose create a new container, which we expect to repeat that sync.

If the GPU machine really has no internet at the session, tell us beforehand; we have a check for this on our list but
have not run it yet.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `error: checkpoint dir not found: ...` / `error: .../params is missing (point at the step dir, e.g. .../<exp>/30000)` (from `serve_munich.sh`) | Wrong `<ckpt_dir>`, or an incomplete download | Point at the step directory that contains `params/` and `assets/`; download again |
| `required variable CKPT_DIR is missing a value` / `... POLICY_CONFIG ...` | Compose was run without the variables | Use `serve_munich.sh`, or set both |
| Errors about missing or mismatching parameters while loading, or a shape error | `<config>` does not match the checkpoint (LoRA vs full fine-tune), or the checkpoint is not a Munich one | Use the config named by the checkpoint's first directory level |
| `FileNotFoundError` for `norm_stats.json` | The `assets/` folder is missing or in the wrong place | Download the whole `<config>/<exp>/<step>/*`; `assets/ebim/task2_munich_real/norm_stats.json` must exist |
| The log hangs or fails while downloading the tokenizer | No internet (`gs://big_vision/paligemma_tokenizer.model`) | Section 6 |
| Out-of-memory (`RESOURCE_EXHAUSTED`) during load or warm-up | Another process holds GPU memory | Stop other GPU jobs (also another server instance); check `nvidia-smi` |
| `NVML: Unknown Error` or the GPU disappears from the container after hours | A system update reloaded the Docker/systemd daemon and the container lost the GPU | Restart the server container. This can happen to a server that has been running for a long time, so restart it shortly before the session |
| `address already in use` | Another process uses the port | Stop it, or pass another `[port]` and tell the client (`POLICY_SERVER_PORT`) |
| Client says `cannot reach policy server at ...: Connection refused` or `timed out` | Server not started, still warming up, wrong host/port, or a firewall | `curl http://<ip>:<port>/healthz` from the laptop; open the TCP port |
| Client says `server is not a munich16 policy` or `server metadata does not match the munich16 layout` | A sim config or a different checkpoint is served | Serve a `*_munich_real` config with its checkpoint |
| Client says `Error in inference server:` followed by a traceback | The server hit an error on a request; the same traceback is in the server log | Read the server log; restart; send us the traceback |
