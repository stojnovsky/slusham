"""
Stream segmenter — wraps ffmpeg to cut a stream into fixed-length chunks.

Two modes depending on config.VIDEO_ENABLED:

  VIDEO_ENABLED=True  → full MP4 chunks (video + audio)
                         chunks/chunk_0000.mp4, chunk_0001.mp4 …

  VIDEO_ENABLED=False → audio-only WAV chunks (no video written at all)
                         chunks/chunk_0000.wav, chunk_0001.wav …
                         16 kHz mono PCM — ready for Whisper directly.

Detection strategy
──────────────────
ffmpeg writes chunk N, then starts chunk N+1.
When chunk N+1 first appears, chunk N is sealed and fired to the callback.
When ffmpeg exits, the last chunk is also fired.
"""

import subprocess
import threading
import time
import logging
from pathlib import Path
from typing import Callable, Optional

import config

log = logging.getLogger(__name__)


class VideoSegmenter:
    def __init__(
        self,
        input_source: str,
        output_dir: str,
        chunk_duration: int = 10,
        on_chunk_ready: Optional[Callable[[Path], None]] = None,
    ):
        self.input_source   = input_source
        self.output_dir     = Path(output_dir)
        self.chunk_duration = chunk_duration
        self.on_chunk_ready = on_chunk_ready

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._process: Optional[subprocess.Popen] = None
        self._watcher: Optional[threading.Thread] = None
        self._running = False

        # Set by start() — either "mp4" or "wav"
        self._ext = "wav" if not config.VIDEO_ENABLED else "mp4"

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> subprocess.Popen:
        self._running = True

        if config.VIDEO_ENABLED:
            cmd = self._cmd_video()
        else:
            cmd = self._cmd_audio_only()

        log.info("ffmpeg command: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        self._watcher = threading.Thread(
            target=self._watch_loop, daemon=True, name="chunk-watcher"
        )
        self._watcher.start()
        return self._process

    def stop(self):
        self._running = False
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=5)

    def wait(self):
        if self._process:
            self._process.wait()
            self._running = False

    # ------------------------------------------------------------------ #
    #  ffmpeg command builders                                             #
    # ------------------------------------------------------------------ #

    def _cmd_video(self) -> list[str]:
        """Full video+audio MP4 segments."""
        pattern = str(self.output_dir / "chunk_%04d.mp4")
        return [
            "ffmpeg",
            "-re",
            "-i", self.input_source,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(self.chunk_duration),
            "-reset_timestamps", "1",
            "-segment_format", "mp4",
            pattern,
            "-y",
        ]

    def _cmd_audio_only(self) -> list[str]:
        """Audio-only WAV segments — no video track written."""
        pattern = str(self.output_dir / "chunk_%04d.wav")
        return [
            "ffmpeg",
            "-re",
            "-i", self.input_source,
            "-vn",                      # drop video stream entirely
            "-acodec", "pcm_s16le",     # 16-bit PCM — Whisper's native format
            "-ar", "16000",             # 16 kHz
            "-ac", "1",                 # mono
            "-f", "segment",
            "-segment_time", str(self.chunk_duration),
            pattern,
            "-y",
        ]

    # ------------------------------------------------------------------ #
    #  Watcher loop                                                        #
    # ------------------------------------------------------------------ #

    def _watch_loop(self):
        seen: list[Path] = []
        notified: set[Path] = set()
        glob = f"chunk_*.{self._ext}"

        while self._running or (self._process and self._process.poll() is None):
            current = sorted(self.output_dir.glob(glob))

            new_files = [f for f in current if f not in seen]
            if new_files:
                for f in new_files:
                    seen.append(f)
                for ready in seen[:-1]:
                    if ready not in notified:
                        notified.add(ready)
                        self._emit(ready)

            time.sleep(1)

        # ffmpeg exited — last chunk is complete
        if seen:
            last = seen[-1]
            if last not in notified and last.exists():
                self._emit(last)

    def _emit(self, chunk: Path):
        size = chunk.stat().st_size if chunk.exists() else 0
        log.info("Chunk ready: %s  (%.1f KB)", chunk.name, size / 1024)
        if self.on_chunk_ready:
            try:
                self.on_chunk_ready(chunk)
            except Exception:
                log.exception("on_chunk_ready callback raised")
