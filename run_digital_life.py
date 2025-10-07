"""Convenient entry point for launching a Digital Life instance."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

# Ensure the src/ package is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from src import TrueDigitalLife  # noqa: E402  (import after sys.path tweak)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a JSON object")
    return data


def _setup_signal_handlers(stop_event: threading.Event, life: TrueDigitalLife) -> None:
    def _handler(_signum, _frame):
        stop_event.set()
        life.shutdown(wait=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Digital Life instance")
    parser.add_argument("--config", type=Path, help="Path to a JSON config file", default=None)
    parser.add_argument("--host", type=str, help="Override host binding", default=None)
    parser.add_argument("--port", type=int, help="Override port binding", default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0 = until interrupted)")
    parser.add_argument("--genesis", action="store_true", help="Force creation of a new genesis block")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    merged_config: Dict[str, Any] = {}
    if args.config:
        merged_config.update(_load_config(args.config))
    if args.host:
        merged_config["host"] = args.host
    if args.port:
        merged_config["port"] = args.port

    life = TrueDigitalLife(genesis=args.genesis, config=merged_config if merged_config else None)

    stop_event = threading.Event()
    _setup_signal_handlers(stop_event, life)

    try:
        if args.duration and args.duration > 0:
            stop_event.wait(args.duration)
            life.shutdown(wait=True)
        else:
            print("Digital Life running. Press Ctrl+C to stop.")
            stop_event.wait()
    finally:
        life.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
