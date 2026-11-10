"""Lightweight CLI for isolated benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msbench.runner import STAGES, plan_run, run_plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a method in its own Python environment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtimes", type=Path, required=True)
    parser.add_argument("--stage", action="append", choices=STAGES, required=True,
                        help="Repeat to run multiple stages in the given order.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="Working directory for existing dataset/output paths.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = plan_run(args.config, args.runtimes, args.stage, args.workspace)
        if args.dry_run:
            # Runtime env values may contain credentials; do not print them.
            preview = {key: value for key, value in plan.items() if key not in {"runtime", "config"}}
            preview["python"] = plan["runtime"]["python"]
            print(json.dumps(preview, indent=2))
        else:
            print(f"Run manifest: {run_plan(plan)}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
