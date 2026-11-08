"""Train the COVID-19 classifier or export its probabilities."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.dataset_cli import run_classifier


if __name__ == "__main__":
    run_classifier("covid19")
