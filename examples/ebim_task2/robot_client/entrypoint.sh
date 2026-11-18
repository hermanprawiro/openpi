#!/bin/bash
# Entrypoint of the robot client image (examples/ebim_task2/Dockerfile.robot_client).
#   <subcommand> [args]          -> python3 -m robot_client <subcommand> [args]
#   bash|sh|pytest|python3|ros2  -> run as given (ROS and the franka messages sourced)

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# shellcheck disable=SC1091
source /opt/franka_ws/install/setup.bash

# compose passes FASTRTPS_DEFAULT_PROFILES_FILE through as an empty string when the operator did not set
# it; Fast DDS would try to load "" as a profile file, so an empty value means "unset".
if [ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
    unset FASTRTPS_DEFAULT_PROFILES_FILE
elif [ ! -r "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    echo "WARNING: FASTRTPS_DEFAULT_PROFILES_FILE=$FASTRTPS_DEFAULT_PROFILES_FILE is not a readable file in the container" >&2
fi

# RMW / domain vs. the site YAML (--site / --set from the arguments, else $ROBOT_SITE). The CLI repeats this
# check per command and refuses motion on a mismatch.
python3 - "$@" <<'PY' || true
import os
import sys

from robot_client.config import check_ros_env, load_site_config

site, sets, args = os.environ.get("ROBOT_SITE"), [], sys.argv[1:]
for i, arg in enumerate(args):
    value = args[i + 1] if i + 1 < len(args) else None
    if arg == "--site" and value is not None:
        site = value
    elif arg.startswith("--site="):
        site = arg.partition("=")[2]
    elif arg == "--set" and value is not None:
        sets.append(value)
    elif arg.startswith("--set="):
        sets.append(arg.partition("=")[2])
if not site or not os.path.isfile(site):
    sys.exit(0)
try:
    cfg = load_site_config(site, sets, os.environ)
except Exception:  # noqa: BLE001 - the CLI reports configuration errors itself
    sys.exit(0)
for problem in check_ros_env(cfg, os.environ):
    print(f"WARNING: {problem}; set the environment variable or the ros section of the site YAML", file=sys.stderr)
PY

case "${1:-}" in
    bash | sh | pytest | python | python3 | ros2)
        exec "$@"
        ;;
esac
exec python3 -m robot_client "$@"
