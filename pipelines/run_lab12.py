import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab12_drift_detection import run_drift_detection

if __name__ == "__main__":
    run_drift_detection()
