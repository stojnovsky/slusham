#!/usr/bin/env python3
"""
Election Room Monitor
─────────────────────
Uses ffmpeg to split a video stream into 10-second chunks, then sends
each chunk's key frames to a local LM Studio vision model to detect
suspicious activities (writing on ballots, altering protocols, etc.).

Usage examples:
  # Watch from a file (for testing)
  python main.py recording.mp4

  # Watch a webcam (device index 0)
  python main.py webcam

  # Watch an IP camera via RTSP
  python main.py rtsp://192.168.1.100/stream

  # Override model or endpoint via environment
  LM_STUDIO_MODEL=my-model python main.py rtsp://...
"""

import argparse
import logging
import sys

from monitor import ElectionMonitor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Election room video monitor — detects suspicious activities via LM Studio.",
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

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("monitor.log"),
        ],
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    input_source = args.input
    if input_source.lower() == "webcam":
        input_source = "0"   # ffmpeg device index

    monitor = ElectionMonitor(input_source, max_workers=args.workers)
    monitor.run()


if __name__ == "__main__":
    main()
