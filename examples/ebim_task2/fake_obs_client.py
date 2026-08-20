"""Sim-free smoke test for the eval client's inference path.

Drives PolicyClient (+ InferenceWorker + AsyncChunkBuffer in async mode)
against a live policy server with synthetic observations on a wall-clock
30 Hz grid. No ROS required. Start the server per POLICY_SERVER.md, then:

  python3 examples/ebim_task2/fake_obs_client.py --host localhost --port 8000

Kill/restart the server mid-run to watch the reconnect path: failed
requests surface as error results, the buffer holds the last action, and
the run continues once the server is back (or aborts after the reconnect
budget).
"""

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_client.chunking import CHUNK_MERGE_STRATEGIES
from eval_client.chunking import AsyncChunkBuffer
from eval_client.config import DEFAULT_ENV_PATH
from eval_client.config import load_dotenv
from eval_client.constants import ACTION_DIM
from eval_client.constants import CHUNK_LEN
from eval_client.constants import CONTROL_HZ
from eval_client.constants import DEFAULT_PROMPT
from eval_client.inference import InferenceWorker
from eval_client.inference import PolicyClient
from eval_client.obs import make_observation

load_dotenv(DEFAULT_ENV_PATH)


def synthetic_observation(rng: np.random.Generator) -> dict:
    images = {
        "head": rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
        "wrist_left": rng.integers(0, 256, (480, 848, 3), dtype=np.uint8),
        "wrist_right": rng.integers(0, 256, (480, 848, 3), dtype=np.uint8),
    }
    return make_observation(np.zeros(20, dtype=np.float32), images, DEFAULT_PROMPT)


def run_sync(client: PolicyClient, steps: int, rng) -> None:
    infer_ms = []
    for i in range(steps):
        start = time.monotonic()
        result = client.infer(synthetic_observation(rng))
        actions = np.asarray(result["actions"])
        assert actions.shape == (CHUNK_LEN, ACTION_DIM), f"bad chunk shape {actions.shape}"
        infer_ms.append((time.monotonic() - start) * 1000.0)
        print(f"infer {i + 1}/{steps}: {infer_ms[-1]:.0f} ms")
    print(f"sync: {steps} infers, mean {np.mean(infer_ms):.0f} ms, max {np.max(infer_ms):.0f} ms")


def run_async(client: PolicyClient, steps: int, args, rng) -> None:
    worker = InferenceWorker(client)
    worker.start()
    buffer = AsyncChunkBuffer(args.replan_steps, args.chunk_size_threshold, args.chunk_merge)
    infer_ms = []
    failures = 0
    step_period = 1.0 / CONTROL_HZ

    worker.submit(0, 0, synthetic_observation(rng))
    buffer.on_request_sent()
    print("priming (first call may hit JIT warmup)...")
    while buffer.in_flight:
        if worker.error is not None:
            raise SystemExit(f"worker died during prime: {worker.error}")
        for result in worker.drain(0):
            if result.error is not None:
                raise SystemExit(f"prime failed: {result.error}")
            buffer.on_chunk(result.obs_step, result.actions, next_step=0)
            infer_ms.append(result.infer_ms)
        time.sleep(0.005)
    print(f"primed in {infer_ms[0]:.0f} ms")

    next_tick = time.monotonic()
    for step in range(steps):
        if worker.error is not None:
            print(f"worker died at step {step}: {worker.error}")
            break
        for result in worker.drain(0):
            if result.error is not None:
                failures += 1
                buffer.in_flight = False
                print(f"step {step}: infer failed ({result.error}); reconnected")
            else:
                buffer.on_chunk(result.obs_step, result.actions, next_step=step)
                infer_ms.append(result.infer_ms)
        if buffer.should_request():
            worker.submit(0, step, synthetic_observation(rng))
            buffer.on_request_sent()
        buffer.pop()
        next_tick += step_period
        time.sleep(max(0.0, next_tick - time.monotonic()))

    worker.stop()
    stats = buffer.stats()
    print(f"async: {len(infer_ms)} chunks, infer mean {np.mean(infer_ms):.0f} ms, failures {failures}, stats {stats}")
    if stats["queue_dry_steps"]:
        print(
            "note: dry steps > 0 -- inference is too slow for "
            f"threshold={args.chunk_size_threshold} horizon={args.replan_steps} at 30 Hz"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("POLICY_SERVER_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("POLICY_SERVER_PORT", "8000")))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--async-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replan-steps", type=int, default=25)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.5)
    parser.add_argument("--chunk-merge", choices=sorted(CHUNK_MERGE_STRATEGIES), default="replace")
    parser.add_argument("--max-reconnects", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    print(f"connecting to {args.host}:{args.port} ...")
    client = PolicyClient(args.host, args.port, max_reconnects=args.max_reconnects)
    print(f"server metadata: {client.metadata()}")

    if args.async_inference:
        run_async(client, args.steps, args, rng)
    else:
        run_sync(client, max(1, args.steps // args.replan_steps), rng)


if __name__ == "__main__":
    main()
