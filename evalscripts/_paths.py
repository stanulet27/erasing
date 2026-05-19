import sys
from pathlib import Path

ERASING_ROOT = Path(__file__).resolve().parents[1]

if str(ERASING_ROOT) not in sys.path:
    sys.path.insert(0, str(ERASING_ROOT))
