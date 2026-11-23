from robot_client import labels, layout


def test_table_matches_layout():
    assert len(labels.LABELS) == layout.ACTION_DIM == 16
    assert [l.index for l in labels.LABELS] == list(range(16))
    for side, sl in layout.ARM_SLICES.items():
        assert all(labels.LABELS[i].side == side and labels.LABELS[i].kind == "joint" for i in range(sl.start, sl.stop))
    for side, i in layout.GRIPPER_INDEX.items():
        g = labels.LABELS[i]
        assert (g.side, g.kind, g.state_unit, g.action_unit) == (side, "gripper", "rad", "0-1")


def test_display_names():
    assert labels.display(11) == "[11] R J4 elbow"
    assert labels.display(0) == "[0] L J1 base yaw"
    assert labels.display(7) == "[7] L gripper"
    assert labels.display(15) == "[15] R gripper"
    assert "right_fr3v2_joint4" in labels.LABELS[11].long


def test_as_json_round_trip():
    rows = labels.as_json()
    assert rows[8] == {"index": 8, "short": "R J1 base yaw", "long": labels.LABELS[8].long, "side": "right",
                       "kind": "joint", "state_unit": "rad", "action_unit": "rad", "display": "[8] R J1 base yaw"}
