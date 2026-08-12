from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cp4282_gs import DATA_ROOT


required = [
    DATA_ROOT / "init.ply",
    DATA_ROOT / "transforms_train.json",
    DATA_ROOT / "transforms_test.json",
]
missing = [path for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing setup files:\n" + "\n".join(str(path) for path in missing))

print(f"Data directory: {DATA_ROOT}")
print("Setup files are present.")
