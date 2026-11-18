from pathlib import Path
import sys

import pytest

# Make `import robot_client` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def pytest_configure(config):
    config.addinivalue_line("markers", "ros: needs rclpy (runs inside the client container)")


def pytest_collection_modifyitems(config, items):
    try:
        import rclpy  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="rclpy not importable")
        for item in items:
            if "ros" in item.keywords:
                item.add_marker(skip)
