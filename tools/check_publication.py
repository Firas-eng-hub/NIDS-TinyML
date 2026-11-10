"""Reject local-only files in the Git index, publishable files, and reachable history."""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
LOCAL_DIRS = {
    "data",
    "artifacts",
    "notebooks",
    "local_experiments",
    "venv",
    ".venv",
    ".claude",
}
LOCAL_SUFFIXES = {
    ".csv",
    ".parquet",
    ".feather",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".ipynb",
    ".pcap",
    ".pcapng",
    ".joblib",
    ".onnx",
    ".pkl",
    ".pickle",
}


def local_only(path):
    p = Path(path)
    return (
        bool(p.parts and p.parts[0] in LOCAL_DIRS)
        or p.suffix.lower() in LOCAL_SUFFIXES
        or "wireshark" in path.lower()
        or p.name == "splits.json"
    )


def git(*args):
    return subprocess.check_output([*GIT, *args], cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    tracked = git("ls-files", "-z").split("\0")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    problems = []
    for name in sorted(set(tracked + untracked) - {""}):
        if local_only(name):
            problems.append(f"Local-only file is publishable: {name}")
        path = ROOT / name
        if path.is_file() and path.stat().st_size < 5_000_000:
            # Report locations only; never echo credential values.
            if re.search(rb"KGAT_[A-Za-z0-9]{16,}", path.read_bytes()):
                problems.append(f"Hardcoded Kaggle credential in {name}")
    if args.history:
        objects = git("rev-list", "--objects", "--all").splitlines()
        for entry in objects:
            _, _, name = entry.partition(" ")
            if name and local_only(name):
                problems.append(f"Local-only path in reachable history: {name}")
    if problems:
        raise SystemExit("\n".join(problems))
    print(
        "Publication checks passed: no local-only files or Kaggle credentials in publishable files."
    )
    if args.history:
        print(
            "Reachable Git history contains no excluded data, models, notebooks or captures."
        )


if __name__ == "__main__":
    main()
