from pathlib import Path
import sys

# Make `import eval_client` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
