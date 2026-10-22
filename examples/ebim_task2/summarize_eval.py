#!/usr/bin/env python3
"""Summarize one eval run into a self-contained markdown report.

Merges the eval_recorder session (evaluator scores, per-episode manifests)
with the eval_client run (result.jsonl timing/abort metadata):

  --session-dir  <eval_raw_out>/<name>/   (recorder output, ebim repo)
  --run-dir      eval_results/<name>/     (client output, this repo)

Writes <run-dir>/summary.md with three sections (summary stats, episode
tables, per-episode images) and copies each episode's best eval-camera RGB
frame into <run-dir>/images/ so the report survives cleanup of the raw
recorder tree. Episodes whose manifest says discarded (the trailing episode
opened by the client's final reset) are excluded from stats.

Normally invoked by run_eval.sh after a run; standalone re-runs are fine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import re
import shutil
import statistics
from typing import Any

EP_RE = re.compile(r"ep_(\d+)")
CALL_RE = re.compile(r"call_(\d+)")
# The sim labels the four board slots A-D; "B" is where the target board is
# authored, so the other slots measure generalization to unseen positions.
HOME_SLOT = "B"


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Call:
    """One evaluator invocation inside an episode."""

    index: int
    path: Path
    raw: dict[str, Any]

    @property
    def iou_block(self) -> dict[str, Any]:
        block = self.raw.get("iou")
        return block if isinstance(block, dict) else {}

    @property
    def audit(self) -> dict[str, Any]:
        block = self.iou_block.get("physical_audit")
        return block if isinstance(block, dict) else {}

    @property
    def ok(self) -> bool:
        """Evaluator ran to completion and produced an IoU block."""
        return bool(self.raw.get("success")) and bool(self.iou_block)

    @property
    def iou(self) -> float | None:
        return _as_float(self.iou_block.get("iou_thermalpad_vs_target_current"))

    @property
    def orientation_ok(self) -> bool | None:
        val = self.iou_block.get("is_orientation_correct")
        return bool(val) if isinstance(val, bool) else None

    @property
    def orientation_case(self) -> str | None:
        return self.iou_block.get("orientation_case")

    @property
    def orientation_conf(self) -> float | None:
        return _as_float(self.iou_block.get("orientation_confidence"))

    @property
    def coverage(self) -> float | None:
        return _as_float(self.audit.get("coverage"))

    @property
    def spill(self) -> float | None:
        return _as_float(self.audit.get("spill"))

    @property
    def off_board(self) -> bool | None:
        val = self.audit.get("off_board")
        return bool(val) if isinstance(val, bool) else None

    @property
    def trigger(self) -> str:
        return self.raw.get("trigger") or "-"

    @property
    def sim_time(self) -> float | None:
        return _as_float(self.raw.get("sim_time"))

    @property
    def score(self) -> float | None:
        """Valid Placement IoU = orientation success x IoU (rulebook metric)."""
        if self.iou is None:
            return None
        return self.iou if self.orientation_ok else 0.0

    @property
    def placed(self) -> bool:
        """Successful attempt: the pad overlaps the target (IoU > 0) the right way up."""
        return self.ok and self.iou is not None and self.iou > 0 and bool(self.orientation_ok)

    @property
    def failure_reason(self) -> str:
        msg = (self.raw.get("message") or "").strip()
        return msg.splitlines()[0] if msg else "no message"

    def rgb_image(self) -> Path | None:
        art = self.path.parent / "artifacts"
        if not art.is_dir():
            return None
        cands = sorted(p for p in art.glob("eval_camera_rgb_*.jpg") if "bbox" not in p.name)
        return cands[0] if cands else None


@dataclass
class Episode:
    name: str
    number: int
    calls: list[Call] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    client: dict[str, Any] | None = None  # matching result.jsonl record

    @property
    def best(self) -> Call | None:
        """Representative call: highest IoU among successful ones."""
        good = [c for c in self.calls if c.ok and c.iou is not None]
        if good:
            return max(good, key=lambda c: (c.iou, c.index))
        return self.calls[-1] if self.calls else None

    @property
    def status(self) -> str:
        if not self.calls:
            return "no evaluator call"
        best = self.best
        if best is None or not best.ok:
            return "evaluator failed"
        return "ok"

    @property
    def success(self) -> bool:
        best = self.best
        return best is not None and best.placed

    @property
    def discarded(self) -> bool:
        return bool(self.manifest.get("discarded"))

    @property
    def target_slot(self) -> str | None:
        ev = self.manifest.get("start_reset_event")
        if isinstance(ev, dict):
            return ev.get("target_slot")
        return None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float(val: Any) -> float | None:
    try:
        if val is None or isinstance(val, bool):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _num(val: float | None, fmt: str = "{:.4f}") -> str:
    return fmt.format(val) if val is not None else "-"


def _bool(val: bool | None) -> str:
    if val is None:
        return "-"
    return "yes" if val else "no"


def _esc(text: Any) -> str:
    """Escape pipes so free text cannot break a markdown table."""
    return str(text).replace("|", "\\|") if text is not None else "-"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def collect_session(session_dir: Path) -> list[Episode]:
    episodes: list[Episode] = []
    ep_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir() and p.name.startswith("ep_"))
    for ep_dir in ep_dirs:
        m = EP_RE.search(ep_dir.name)
        ep = Episode(
            name=ep_dir.name,
            number=int(m.group(1)) if m else 0,
            manifest=_load_json(ep_dir / "manifest.json"),
        )
        eval_dir = ep_dir / "evaluator"
        if eval_dir.is_dir():
            call_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("call_"))
            for call_dir in call_dirs:
                result = call_dir / "evaluator_result.json"
                if not result.is_file():
                    continue
                raw = _load_json(result)
                cm = CALL_RE.search(call_dir.name)
                idx = _as_float(raw.get("call_index"))
                fallback = int(cm.group(1)) if cm else 0
                ep.calls.append(Call(index=int(idx) if idx is not None else fallback, path=result, raw=raw))
        ep.calls.sort(key=lambda c: c.index)
        episodes.append(ep)
    episodes.sort(key=lambda e: (e.number, e.name))
    return episodes


def load_client_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "result.jsonl"
    if not path.is_file():
        return []
    records = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def merge_client_records(episodes: list[Episode], records: list[dict[str, Any]]) -> None:
    """Recorder episodes are 1-based (ep_001), client records 0-based: every
    client episode triggers exactly one reset request/event pair, so the
    mapping is positional."""
    by_index = {rec.get("episode"): rec for rec in records}
    for ep in episodes:
        ep.client = by_index.get(ep.number - 1)


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def render_summary(
    name: str,
    episodes: list[Episode],
    session_dir: Path,
    run_dir: Path,
    records: list[dict[str, Any]],
    warnings: list[str],
    discarded: list[Episode],
) -> list[str]:
    evaluated = [e for e in episodes if e.status == "ok"]
    ious = [(e, e.best.iou) for e in evaluated if e.best and e.best.iou is not None]
    scores = [(e, e.best.score) for e in evaluated if e.best and e.best.score is not None]
    oriented = [e for e in evaluated if e.best and e.best.orientation_ok]
    coverages = [e.best.coverage for e in evaluated if e.best and e.best.coverage is not None]
    spills = [e.best.spill for e in evaluated if e.best and e.best.spill is not None]

    lines = [f"# {name} - evaluation summary", ""]
    lines.extend(f"> **WARNING:** {warning}" for warning in warnings)
    if warnings:
        lines.append("")
    lines += ["## Summary", ""]

    status = _load_json(session_dir / ".status.json")
    prov = _load_json(session_dir / ".session_provenance.json")
    meta: list[str] = [
        f"- Recorder session: `{session_dir}`",
        f"- Client run dir: `{run_dir}`",
    ]
    if prov.get("wall_time_utc"):
        meta.append(f"- Session start (UTC): `{prov['wall_time_utc']}`")
    if prov.get("bench_head"):
        meta.append(f"- Benchmark HEAD: `{str(prov['bench_head'])[:12]}`")
    if prov.get("submission_head"):
        meta.append(f"- Submission HEAD: `{str(prov['submission_head'])[:12]}`")
    versions = sorted(
        {
            c.iou_block.get("evaluator_version")
            for e in episodes
            for c in e.calls
            if c.iou_block.get("evaluator_version")
        }
    )
    if versions:
        joined = ", ".join(f"`{v}`" for v in versions)
        meta.append(f"- Evaluator version(s): {joined}")
    if status.get("state"):
        meta.append(f"- Last recorder state: `{status['state']}` (resets={status.get('resets', '-')})")
    lines += [*meta, ""]

    if ious:
        best_ep, best_iou = max(ious, key=lambda kv: kv[1])
        vals = [v for _, v in ious]
    else:
        best_ep, best_iou, vals = None, None, []
    if scores:
        best_score_ep, best_score = max(scores, key=lambda kv: kv[1])
    else:
        best_score_ep, best_score = None, None

    n_none = sum(1 for e in episodes if e.status == "no evaluator call")
    n_fail = sum(1 for e in episodes if e.status == "evaluator failed")
    n_off = sum(1 for e in evaluated if e.best and e.best.off_board)
    max_iou_cell = f"| **Max IoU** | **{_num(best_iou)}**"
    max_iou_cell += f" (`{best_ep.name}`) |" if best_ep else " |"
    max_score_cell = f"| Max valid-placement IoU (orientation x IoU) | {_num(best_score)}"
    max_score_cell += f" (`{best_score_ep.name}`) |" if best_score_ep else " |"

    aborted = [r for r in records if r.get("aborted")]
    steps = [_as_float(r.get("steps")) for r in records]
    steps = [v for v in steps if v is not None]
    durations = [_as_float(r.get("duration_s")) for r in records]
    durations = [v for v in durations if v is not None]
    infers = [_as_float(r.get("infer_ms_mean")) for r in records]
    infers = [v for v in infers if v is not None]
    jumps = [_as_float(r.get("switch_jump_max")) for r in records]
    jumps = [v for v in jumps if v is not None]

    lines += [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Episodes recorded | {len(episodes)} |",
        f"| Episodes with a usable evaluator result | {len(evaluated)} |",
        f"| Episodes with no evaluator call | {n_none} |",
        f"| Episodes where the evaluator failed | {n_fail} |",
        max_iou_cell,
        f"| Mean IoU (evaluated episodes) | {_num(statistics.fmean(vals) if vals else None)} |",
        f"| Median IoU (evaluated episodes) | {_num(statistics.median(vals) if vals else None)} |",
        f"| Episodes with IoU > 0 | {sum(1 for v in vals if v > 0)} / {len(vals)} |",
        f"| Orientation correct | {len(oriented)} / {len(evaluated)} |",
        f"| Successful episodes (IoU > 0 and orientation correct) | "
        f"{sum(1 for e in episodes if e.success)} / {len(episodes)} |",
        max_score_cell,
        f"| Mean physical coverage | {_num(statistics.fmean(coverages) if coverages else None)} |",
        f"| Mean physical spill | {_num(statistics.fmean(spills) if spills else None)} |",
        f"| Episodes flagged off-board | {n_off} / {len(evaluated)} |",
        f"| Client episodes (result.jsonl) | {len(records)} |",
        f"| Client aborted episodes | {len(aborted)} |",
        f"| Mean steps per episode | {_num(statistics.fmean(steps) if steps else None, '{:.1f}')} |",
        f"| Mean episode duration (s) | {_num(statistics.fmean(durations) if durations else None, '{:.1f}')} |",
        f"| Mean inference latency (ms) | {_num(statistics.fmean(infers) if infers else None, '{:.1f}')} |",
        f"| Mean hand-off jump, max per episode (rad) | {_num(statistics.fmean(jumps) if jumps else None)} |",
        "",
    ]

    if best_ep is not None and best_ep.best is not None:
        b = best_ep.best
        lines += [
            f"Best episode `{best_ep.name}` (call {b.index}, trigger `{b.trigger}`): "
            f"IoU {_num(b.iou)}, orientation {_bool(b.orientation_ok)} "
            f"({_esc(b.orientation_case)}, conf {_num(b.orientation_conf)}), "
            f"coverage {_num(b.coverage)}, spill {_num(b.spill)}.",
            "",
        ]

    lines += render_slot_table(episodes)

    problems = [e for e in episodes if e.status != "ok"]
    if problems:
        lines += ["Episodes without a usable result:", ""]
        for e in problems:
            reason = e.best.failure_reason if e.best else "evaluator was never called"
            lines.append(f"- `{e.name}` - {e.status}: {_esc(reason[:220])}")
        lines.append("")

    if discarded:
        names = ", ".join(f"`{e.name}`" for e in discarded)
        lines += [
            f"Skipped {len(discarded)} discarded episode(s): {names} "
            "(the trailing episode opened by the client's final reset is discarded on purpose).",
            "",
        ]

    return lines


def render_slot_table(episodes: list[Episode]) -> list[str]:
    """Success (IoU > 0 and orientation correct) per target-board slot. Every
    recorded episode counts as an attempt, so one without a usable evaluator
    result is a failure."""

    def row(label: str, eps: list[Episode]) -> str:
        wins = sum(1 for e in eps if e.success)
        rate = f"{wins / len(eps):.0%}" if eps else "-"
        return f"| {label} | {wins} | {len(eps)} | {rate} |"

    by_slot: dict[str, list[Episode]] = {}
    for ep in episodes:
        by_slot.setdefault(ep.target_slot or "-", []).append(ep)
    lines = [
        "### Success by target slot",
        "",
        "Success = IoU > 0 and orientation correct. `-` = no slot recorded for the episode.",
        "",
        "| Slot | Successes | Episodes | Success rate |",
        "| --- | --- | --- | --- |",
    ]
    for slot in sorted(by_slot, key=lambda s: (s == "-", s)):
        label = f"{_esc(slot)} (home)" if slot == HOME_SLOT else _esc(slot)
        lines.append(row(label, by_slot[slot]))
    if HOME_SLOT in by_slot:
        others = [e for slot, eps in by_slot.items() if slot not in (HOME_SLOT, "-") for e in eps]
        lines.append(row(f"Other than {HOME_SLOT}", others))
    lines += [row("**All**", episodes), ""]
    return lines


def _client_cells(ep: Episode) -> str:
    rec = ep.client or {}
    if rec.get("aborted"):
        aborted = f"yes: {_esc(rec.get('abort_reason', '?'))}"
    elif ep.client is None:
        aborted = "-"
    else:
        aborted = "no"
    return (
        f"{_num(_as_float(rec.get('steps')), '{:.0f}')} | "
        f"{_num(_as_float(rec.get('duration_s')), '{:.1f}')} | "
        f"{_num(_as_float(rec.get('infer_ms_mean')), '{:.1f}')} | "
        f"{_num(_as_float(rec.get('switch_jump_max')))} | "
        f"{aborted}"
    )


def render_tables(episodes: list[Episode]) -> list[str]:
    lines = [
        "## Episode tables",
        "",
        "One row per episode; evaluator columns come from the recorder, the "
        "client columns (steps / duration / inference / jump / aborted) from "
        "result.jsonl. Slot is the target board's slot for the episode; Success = "
        "IoU > 0 and orientation correct. When an episode was evaluated more than once, the row "
        "shows the call with the highest IoU; every call is listed in the "
        "per-call table below.",
        "",
        "| Episode | Slot | Calls | Call used | Trigger | Success | IoU | Orient. correct | "
        "Orient. case | Orient. conf | Coverage | Spill | Off-board | Status | Steps | "
        "Duration (s) | Infer (ms) | Jump (rad) | Aborted |",
        "| " + " | ".join(["---"] * 19) + " |",
    ]
    for ep in episodes:
        c = ep.best
        slot = _esc(ep.target_slot)
        if c is None:
            lines.append(
                f"| `{ep.name}` | {slot} | 0 | - | - | no | - | - | - | - | - | - | - | "
                f"{ep.status} | {_client_cells(ep)} |"
            )
            continue
        lines.append(
            f"| `{ep.name}` | {slot} | {len(ep.calls)} | {c.index} | {_esc(c.trigger)} | "
            f"{_bool(ep.success)} | {_num(c.iou)} | {_bool(c.orientation_ok)} | {_esc(c.orientation_case)} | "
            f"{_num(c.orientation_conf)} | {_num(c.coverage)} | {_num(c.spill)} | "
            f"{_bool(c.off_board)} | {ep.status} | {_client_cells(ep)} |"
        )
    lines.append("")

    multi = [ep for ep in episodes if len(ep.calls) > 1]
    if multi:
        lines += [
            "### Per-call detail (episodes with more than one evaluator call)",
            "",
            "| Episode | Call | Trigger | Sim time | IoU | Orient. correct | Orient. case | "
            "Orient. conf | Coverage | Spill | Off-board | Evaluator ok |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for ep in multi:
            for c in ep.calls:
                lines.append(
                    f"| `{ep.name}` | {c.index} | {_esc(c.trigger)} | "
                    f"{_num(c.sim_time, '{:.2f}')} | {_num(c.iou)} | "
                    f"{_bool(c.orientation_ok)} | {_esc(c.orientation_case)} | "
                    f"{_num(c.orientation_conf)} | {_num(c.coverage)} | {_num(c.spill)} | "
                    f"{_bool(c.off_board)} | {_bool(bool(c.raw.get('success')))} |"
                )
        lines.append("")
    return lines


def render_images(episodes: list[Episode], images_dir: Path) -> list[str]:
    """Copy each episode's best eval-camera frame into images_dir (next to the
    summary) so the report stays valid if the raw recorder tree is removed."""
    lines = [
        "## Episode images",
        "",
        "Each entry repeats the episode's table row and shows the `eval_camera` RGB "
        "frame captured by that evaluator call.",
        "",
    ]
    for ep in episodes:
        lines += [f"### `{ep.name}`", ""]
        c = ep.best
        if c is None:
            lines += [f"- Status: {ep.status}", ""]
            continue
        lines += [
            f"- Status: {ep.status} - call {c.index} of {len(ep.calls)}, trigger `{c.trigger}`",
            f"- IoU: {_num(c.iou)}",
            f"- Orientation: correct={_bool(c.orientation_ok)}, "
            f"case={_esc(c.orientation_case)}, confidence={_num(c.orientation_conf)}",
            f"- Physical audit: coverage={_num(c.coverage)}, spill={_num(c.spill)}, off_board={_bool(c.off_board)}",
        ]
        if ep.target_slot:
            lines.append(f"- Target slot: `{ep.target_slot}`")
        if ep.client is not None:
            rec = ep.client
            client_line = (
                f"- Client: steps={_num(_as_float(rec.get('steps')), '{:.0f}')}, "
                f"duration={_num(_as_float(rec.get('duration_s')), '{:.1f}')}s, "
                f"infer={_num(_as_float(rec.get('infer_ms_mean')), '{:.1f}')}ms"
            )
            if rec.get("aborted"):
                client_line += f", aborted: {_esc(rec.get('abort_reason', '?'))}"
            lines.append(client_line)
        if not c.ok:
            lines.append(f"- Evaluator message: {_esc(c.failure_reason[:400])}")
        lines.append("")

        img = c.rgb_image()
        if img is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            copied = images_dir / f"{ep.name}{img.suffix}"
            shutil.copyfile(img, copied)
            lines += [f"![{ep.name} eval_camera RGB]({images_dir.name}/{copied.name})", ""]
        else:
            lines += ["_No eval_camera RGB artifact for this episode._", ""]
    return lines


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def write_summary(session_dir: Path, run_dir: Path, out: Path | None = None) -> Path:
    session_dir = Path(session_dir)
    run_dir = Path(run_dir)
    out = Path(out) if out is not None else run_dir / "summary.md"

    all_episodes = collect_session(session_dir)
    discarded = [e for e in all_episodes if e.discarded]
    episodes = [e for e in all_episodes if not e.discarded]
    records = load_client_records(run_dir)

    warnings: list[str] = []
    if not (run_dir / "result.jsonl").is_file():
        warnings.append(f"no result.jsonl in `{run_dir}` - client columns are empty.")
    elif len(records) != len(episodes):
        warnings.append(
            f"episode count mismatch: recorder has {len(episodes)} episode(s), "
            f"client result.jsonl has {len(records)} record(s) - client columns may be misaligned."
        )
    merge_client_records(episodes, records)

    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += render_summary(session_dir.name, episodes, session_dir, run_dir, records, warnings, discarded)
    lines += render_tables(episodes)
    lines += render_images(episodes, out.parent / "images")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge recorder + client eval artifacts into one markdown summary.")
    ap.add_argument("--session-dir", type=Path, required=True, help="recorder output: <eval_raw_out>/<name>/")
    ap.add_argument("--run-dir", type=Path, required=True, help="client output: eval_results/<name>/")
    ap.add_argument("--out", type=Path, default=None, help="output file (default: <run-dir>/summary.md)")
    args = ap.parse_args()

    if not args.session_dir.is_dir():
        raise SystemExit(f"session dir not found: {args.session_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    out = write_summary(args.session_dir, args.run_dir, args.out)
    episodes = [e for e in collect_session(args.session_dir) if not e.discarded]
    n_ok = sum(1 for e in episodes if e.status == "ok")
    print(f"{args.session_dir.name}: {len(episodes)} episodes ({n_ok} evaluated) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
