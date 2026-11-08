#!/usr/bin/env python3
"""Write the Unity camera-replay scene for one NeRF-Synthetic scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msbench.unity_replay import write_synthetic_replay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="NeRF-Synthetic scene root")
    parser.add_argument("--output", type=Path, required=True, help="Replay scene directory")
    parser.add_argument("--background", choices=("black", "white"), default="white")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(write_synthetic_replay(args.dataset, args.output, background=args.background, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
