import sys
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data"


print(ROOT_DIR, DATA_PATH)