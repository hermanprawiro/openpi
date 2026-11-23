"""Human-readable labels for the munich16 state/action dims, derived from `layout`."""

from dataclasses import asdict, dataclass

from robot_client import layout

JOINT_NAMES = ("base yaw", "shoulder", "upper-arm roll", "elbow", "forearm roll", "wrist pitch", "flange roll")


@dataclass(frozen=True)
class Label:
    index: int
    short: str
    long: str
    side: str  # "left" | "right"
    kind: str  # "joint" | "gripper"
    state_unit: str
    action_unit: str


def _build() -> tuple[Label, ...]:
    labels = []
    for side, sl in layout.ARM_SLICES.items():
        prefix = side[0].upper()
        for n, index in enumerate(range(sl.start, sl.stop), start=1):
            labels.append(
                Label(
                    index=index,
                    short=f"{prefix} J{n} {JOINT_NAMES[n - 1]}",
                    long=f"{side} FR3 joint {n} ({side}_fr3v2_joint{n}), target/measured angle",
                    side=side,
                    kind="joint",
                    state_unit="rad",
                    action_unit="rad",
                )
            )
    for side, index in layout.GRIPPER_INDEX.items():
        labels.append(
            Label(
                index=index,
                short=f"{side[0].upper()} gripper",
                long=f"{side} gripper: state = knuckle angle (rad), action = width fraction 0-1",
                side=side,
                kind="gripper",
                state_unit="rad",
                action_unit="0-1",
            )
        )
    return tuple(sorted(labels, key=lambda label: label.index))


LABELS: tuple[Label, ...] = _build()


def display(i: int) -> str:
    return f"[{i}] {LABELS[i].short}"


def as_json() -> list[dict]:
    return [{**asdict(label), "display": display(label.index)} for label in LABELS]
