"""CLI bootstrap; isolated runs do not import PyTorch or renderer modules."""

from __future__ import annotations

import sys


def __getattr__(name: str):
    if name == "app":
        from msbench.cli.app import app
        return app
    raise AttributeError(name)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from msbench.cli.run import main as run_main
        run_main(sys.argv[2:])
        return
    try:
        from msbench.cli.app import app
    except ModuleNotFoundError as exc:
        message = (
            f"Native commands need the runtime dependencies (missing {exc.name}). "
            "Install msbench[runtime] in a method environment.\n"
            "The control environment supports: msbench run --help\n"
        )
        if len(sys.argv) == 1 or sys.argv[1] in {"--help", "-h"}:
            print(message)
            return
        raise SystemExit(message) from exc
    app()


if __name__ == "__main__":
    main()
