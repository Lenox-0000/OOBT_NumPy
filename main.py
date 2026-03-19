import subprocess
from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
SCRIPTS = ["build_numpy.py"]


def run_benchmark() -> None:
    for script in SCRIPTS:
        try:
            subprocess.run([sys.executable, SCRIPTS_DIR / script], check=True)
        except subprocess.CalledProcessError as e:
            print(e)
            sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
