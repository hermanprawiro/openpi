from pathlib import Path
import sys

# The converter is run as a script from examples/ebim_task2/, so that directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
