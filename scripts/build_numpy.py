"""
build_numpy.py

Fetches the latest stable release tag from the official NumPy GitHub repository
and builds it from source in the local runner environment.

Usage:
    python scripts/build_numpy.py
"""

import subprocess
import sys
import os
import urllib.request
import json


NUMPY_REPO = "https://api.github.com/repos/numpy/numpy/releases/latest"
CLONE_DIR = "numpy_source"


def get_latest_stable_tag() -> str:
    """Fetch the latest stable release tag name from GitHub API."""
    print("[INFO] Fetching latest stable NumPy release tag...")
    req = urllib.request.Request(
        NUMPY_REPO,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "OOBT-NumPy"}
    )
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
    tag = data["tag_name"]
    print(f"[INFO] Latest stable tag: {tag}")
    return tag


def run(cmd: list[str], cwd: str = None) -> None:
    """Run a shell command, raising an error on failure."""
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)


def clone_repo(tag: str) -> None:
    """Clone the NumPy repository at the given tag."""
    if os.path.exists(CLONE_DIR):
        print(f"[INFO] Directory '{CLONE_DIR}' already exists, skipping clone.")
        return
    print(f"[INFO] Cloning NumPy at tag {tag}...")
    run([
        "git", "clone",
        "--depth", "1",
        "--branch", tag,
        "--recurse-submodules",
        "https://github.com/numpy/numpy.git",
        CLONE_DIR
    ])


def build_numpy() -> None:
    """Build and install NumPy from source."""
    print("[INFO] Installing build dependencies...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "meson-python", "meson", "ninja", "Cython"])

    print("[INFO] Building and installing NumPy from source...")
    run([
        sys.executable, "-m", "pip", "install",
        "--no-build-isolation",
        "-e", "."
    ], cwd=CLONE_DIR)


def verify_install() -> None:
    """Verify that NumPy was installed correctly by importing it."""
    print("[INFO] Verifying NumPy installation...")
    result = subprocess.run(
        [sys.executable, "-c", "import numpy; print(f'[OK] NumPy {numpy.__version__} installed successfully.')"],
        capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] NumPy import verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    tag = get_latest_stable_tag()
    clone_repo(tag)
    build_numpy()
    verify_install()
