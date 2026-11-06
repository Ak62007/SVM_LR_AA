"""Experiment 01: correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import cli

if __name__ == '__main__':
    cli(1)
