#!/usr/bin/env python3
"""
Election Room Monitor
─────────────────────
Uses ffmpeg to split a video stream into 10-second chunks, then sends
each chunk's key frames / audio to a local LM Studio model to detect
suspicious activities (writing on ballots, altering protocols, etc.).

Usage examples:
  # Named session (recommended)
  ./run.sh -t "секция_12_Sofia" recording.mp4

  # Auto-named session (timestamp)
  ./run.sh recording.mp4

  # Live webcam
  ./run.sh -t "секция_5" webcam

  # IP camera via RTSP
  ./run.sh -t "камера_А" rtsp://192.168.1.100/stream

Each session writes to its own directory:
  sessions/<title>/
    monitor.log    ← full log for this session
    alerts.jsonl   ← suspicious events, one JSON per line
    chunks/        ← temporary audio/video segments (auto-cleaned)
"""

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from monitor import ElectionMonitor


def _slugify(text: str) -> str:
    """Make a string safe for use as a directory/filename."""
    text = text.strip()
    text = re.sub(r"[^\w\-.]", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "session"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Election room monitor — detects suspicious activities via LM Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "input",
        help=(
            "Video source: file path, RTSP/HTTP URL, "
            "or 'webcam' for the default camera (index 0)."
        ),
    )
    p.add_argument(
        "--title", "-t",
        default=None,
        metavar="NAME",
        help=(
            "Session name used for log and alert filenames. "
            "Defaults to a timestamp (e.g. session_20260417_184200)."
        ),
    )
    p.add_argument(
        "--workers", "-w",
        type=int, default=4, metavar="N",
        help="Max concurrent analysis threads (default: 4).",
    )
    p.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


def main():
    args = build_parser().parse_args()

    # ── Session directory ──────────────────────────────────────────────
    raw_title = args.title or datetime.now().strftime("session_%Y%m%d_%H%M%S")
    title     = _slugify(raw_title)
    session_dir = Path("sessions") / title
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────
    log_path = session_dir / "monitor.log"
    level    = logging.DEBUG if args.debug else logging.INFO
    fmt      = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(level=level, handlers=[file_handler, stream_handler])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    log = logging.getLogger(__name__)
    log.info("Session '%s' → %s", title, session_dir)

    # ── Run ────────────────────────────────────────────────────────────
    input_source = args.input
    if input_source.lower() == "webcam":
        input_source = "0"

    monitor = ElectionMonitor(
        input_source = input_source,
        session_dir  = session_dir,
        title        = title,
        max_workers  = args.workers,
    )
    monitor.run()


if __name__ == "__main__":
    main()
