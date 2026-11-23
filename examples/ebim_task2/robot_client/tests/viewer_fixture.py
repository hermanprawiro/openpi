"""Run-folder fixture for the viewer tests: drives the real `Runner` (through `test_runner.Rig`) with a real
`EpisodeRecorder`, so `steps.jsonl`, `chunks.npz` and the images are exactly what a robot run writes. The run-level
files the CLI writes around the episode (config.yaml, events.log, result.json) are written here, in the CLI's order.

Manual use:
    cd examples/ebim_task2/robot_client/tests && PYTHONPATH=../.. uv run python -c \
        "from pathlib import Path; from viewer_fixture import make_run; \
         make_run(Path.home()/'.cache/ebim_viewer_runs', 'run_demo', hold_at=(25, 35))"
"""

from pathlib import Path

import numpy as np
from PIL import Image
from robot_client.recorder import RunRecorder
from test_runner import Rig

QUIT_TICK = 100  # end="operator": the operator presses q at this tick
FREEZE_TICK = 60  # end="freeze": the inference worker dies at this tick
INTERRUPT_TICK = 70  # end="interrupted": Ctrl-C at this tick
INTERRUPT_NOTE = "interrupted (Ctrl-C / SIGTERM / SIGHUP)"
CAMERAS = ("head", "wrist_left", "wrist_right")


def make_run(
    out_dir: Path,
    name: str = "run_test",
    *,
    overrides: tuple[str, ...] = (),
    timeout: float = 6.0,
    respond=None,
    ages=None,
    keys=None,
    dry_run: bool = False,
    end: str | None = "operator",
    nan_t_request: bool = False,
    check_obs: bool = False,
    episodes: bool = True,
    hold_at: tuple[int, int] | None = None,
) -> Path:
    """Write one run folder `<out_dir>/<name>` and return it.

    `end`: "operator" (q at tick 100), "timeout" (the episode times out), "freeze" (the worker dies at tick 60),
    "interrupted" (Ctrl-C at tick 70) or None (a live run: nothing is written after the steps).
    """
    if keys is None and end == "operator":
        keys = {QUIT_TICK: "q"}
    rig = Rig(list(overrides), respond=None, keys=keys, timeout=timeout)
    rig.worker.respond = respond or (lambda n: (3, rig.chunk(tag=0.0), None))
    base_ages = ages or (lambda tick: {})
    if hold_at is not None:
        start, stop = hold_at
        rig.ages = lambda tick: {**base_ages(tick), **({"cam/head": 2.0} if start <= tick < stop else {})}
    else:
        rig.ages = base_ages

    def on_tick(tick: int) -> None:
        if end == "freeze" and tick == FREEZE_TICK:
            rig.worker.error = RuntimeError("fixture: worker died")
        if end == "interrupted" and tick == INTERRUPT_TICK:
            raise KeyboardInterrupt

    rig.on_tick = on_tick

    rec = RunRecorder(
        out_dir, name, rig.cfg.redacted_dict(), {"layout": "munich16", "checkpoint": "test/0"}
    )  # config.yaml: JSON round-tripped by the recorder, so state.pin keys are strings as on a real run
    run_dir = rec.dir
    try:
        if end or episodes:  # a check-only run without `end` writes no events at all
            rec.event("run_start", dry_run=dry_run, sides=["left", "right"], active_arms=list(rig.cfg.active_arms))
        if check_obs:
            for camera in CAMERAS:
                Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8)).save(
                    run_dir / f"check_obs_{camera}.jpg", quality=90
                )
        if episodes:
            _episode(rec, rig, end, dry_run=dry_run)
            if nan_t_request:
                _nan_first_t_request(run_dir / "episode_000" / "chunks.npz")
        if end:
            rec.event("run_end")
    finally:
        rec.close()
    return run_dir


def _episode(rec: RunRecorder, rig: Rig, end: str | None, *, dry_run: bool) -> None:
    ep = rec.episode(0)
    rig.runner.recorder = ep
    rec.event("episode_start", episode=0, generation=1, dry_run=dry_run)
    stream_stats = {"node": {}, "streamer": {"cycles": 0, "fault": None}}
    try:
        outcome = rig.run()
    except KeyboardInterrupt:
        ep.finish("freeze", INTERRUPT_NOTE, {"dry_run": dry_run}, stream_stats)
        rec.event("interrupted", note=INTERRUPT_NOTE)
        return
    if end is None:
        ep._steps.close()  # noqa: SLF001 - a live run's files stay unfinished, but do not leak the handle
        return
    stats = {**outcome.stats, "ended_by": outcome.ended_by, "reason": outcome.reason, "dry_run": dry_run}
    rec.event("episode_end", episode=0, ended_by=outcome.ended_by, reason=outcome.reason, steps=outcome.steps)
    if outcome.ended_by == "freeze":
        ep.finish("freeze", "", stats, stream_stats)
    else:
        ep.finish("s", "", stats, stream_stats)
        rec.event("outcome", episode=0, outcome="s")


def _nan_first_t_request(path: Path) -> None:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["t_request"] = arrays["t_request"].copy()
    arrays["t_request"][0] = np.nan
    np.savez(path, **arrays)
