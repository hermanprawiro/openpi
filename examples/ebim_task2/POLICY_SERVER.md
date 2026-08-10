# Serving a task2 policy with Docker

Run these from the repo root (the mount uses the current directory, so there is no absolute
path to edit). The image `openpi_server` only needs to be built once, so starting the server
later is a single command:

bash (Linux / macOS / WSL):

```bash
docker run -d --name openpi_server --gpus=all -p 8000:8000 \
  -v "$(pwd):/app" \
  -v "$HOME/.cache/openpi:/openpi_assets" \
  -e OPENPI_DATA_HOME=/openpi_assets -e IS_DOCKER=true \
  -e SERVER_ARGS="policy:checkpoint --policy.config=pi05_ebim_task2_fixpos_200_finetune --policy.dir=checkpoints/pi05_ebim_task2_fixpos_200_finetune/pi05_fixpos_200/30000" \
  openpi_server
```

cmd:

```bat
docker run -d --name openpi_server --gpus=all -p 8000:8000 ^
  -v "%CD%:/app" ^
  -v "%USERPROFILE%\.cache\openpi:/openpi_assets" ^
  -e OPENPI_DATA_HOME=/openpi_assets -e IS_DOCKER=true ^
  -e SERVER_ARGS="policy:checkpoint --policy.config=pi05_ebim_task2_fixpos_200_finetune --policy.dir=checkpoints/pi05_ebim_task2_fixpos_200_finetune/pi05_fixpos_200/30000" ^
  openpi_server
```

PowerShell:

```powershell
docker run -d --name openpi_server --gpus=all -p 8000:8000 `
  -v "${PWD}:/app" `
  -v "$env:USERPROFILE\.cache\openpi:/openpi_assets" `
  -e OPENPI_DATA_HOME=/openpi_assets -e IS_DOCKER=true `
  -e SERVER_ARGS="policy:checkpoint --policy.config=pi05_ebim_task2_fixpos_200_finetune --policy.dir=checkpoints/pi05_ebim_task2_fixpos_200_finetune/pi05_fixpos_200/30000" `
  openpi_server
```

Notes:

- To serve a different checkpoint, just change `--policy.dir` (and `--policy.config` if it's a different training config) inside `SERVER_ARGS`.
- Watch startup progress with `docker logs -f openpi_server` — it's ready when you see `server listening on 0.0.0.0:8000`. Startup takes ~1 minute to load weights plus ~40 s to warm up (a throwaway inference that triggers the JIT compile), so the first real request from a client is already fast. To skip the warmup and start serving sooner (at the cost of a slow first request), put `--no-warmup` at the *front* of `SERVER_ARGS`, before `policy:checkpoint`.
- Stop it with `docker rm -f openpi_server`.
- In Git Bash on Windows, prefix the bash command with `MSYS_NO_PATHCONV=1` — MSYS otherwise rewrites the `/app` and `/openpi_assets` container paths into Windows paths.
- If you ever change dependencies (`pyproject.toml`/`uv.lock`) you'll need to rebuild the image first: `docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile`. Code changes in `src/` don't need a rebuild since the repo is volume-mounted into the container.
