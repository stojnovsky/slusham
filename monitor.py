"""
ElectionMonitor — ties segmenter, video analyzer, and audio analyzer.

For each 10-second chunk, video and audio analysis run concurrently.
Results are merged: if either channel flags suspicious activity the
combined alert is printed and logged.
"""

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from segmenter       import VideoSegmenter
from analyzer        import ChunkAnalyzer
from audio_analyzer  import AudioAnalyzer
import config

log = logging.getLogger(__name__)

# ANSI colours (disabled automatically on non-tty)
_tty = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty else text

RED    = lambda t: _c("91;1", t)
YELLOW = lambda t: _c("93;1", t)
CYAN   = lambda t: _c("96",   t)
GREEN  = lambda t: _c("92",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)

_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class ElectionMonitor:
    def __init__(self, input_source: str, max_workers: int = 4):
        self.input_source  = input_source
        self._video_az     = ChunkAnalyzer() if config.VIDEO_ENABLED else None
        self._audio_az     = AudioAnalyzer() if config.AUDIO_ENABLED else None
        self._executor     = ThreadPoolExecutor(max_workers=max_workers,
                                                thread_name_prefix="analyzer")
        self._alerts: list[dict] = []
        self._lock = threading.Lock()

        self._segmenter = VideoSegmenter(
            input_source   = input_source,
            output_dir     = config.CHUNKS_DIR,
            chunk_duration = config.CHUNK_DURATION,
            on_chunk_ready = self._on_chunk_ready,
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        self._print_header()
        self._segmenter.start()
        try:
            self._segmenter.wait()
        except KeyboardInterrupt:
            print()
            log.info("Keyboard interrupt — stopping.")
            self._segmenter.stop()
        finally:
            self._executor.shutdown(wait=True)
            self._print_summary()

    # ------------------------------------------------------------------ #
    #  Per-chunk pipeline                                                  #
    # ------------------------------------------------------------------ #

    def _on_chunk_ready(self, chunk: Path):
        self._executor.submit(self._process_chunk, chunk)

    def _process_chunk(self, chunk: Path):
        """Run video + audio analysis concurrently, then report merged result."""
        inner = ThreadPoolExecutor(max_workers=2, thread_name_prefix="av")
        futures = {}
        if self._video_az:
            futures["video"] = inner.submit(self._video_az.analyze, chunk)
        if self._audio_az:
            futures["audio"] = inner.submit(self._audio_az.analyze, chunk)

        results = {}
        for key, fut in futures.items():
            try:
                results[key] = fut.result()
            except Exception as exc:
                log.error("%s analysis raised: %s", key, exc)
                results[key] = None
        inner.shutdown(wait=False)

        self._report_merged(chunk, results.get("video"), results.get("audio"))

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    def _report_merged(self, chunk: Path, video: dict | None, audio: dict | None):
        ts   = datetime.now().strftime("%H:%M:%S")
        name = chunk.name

        v_sus  = video and video.get("suspicious", False) and \
                 video.get("confidence", 0) >= config.SUSPICIOUS_CONFIDENCE_THRESHOLD
        a_sus  = audio and audio.get("suspicious", False) and \
                 audio.get("confidence", 0) >= config.SUSPICIOUS_CONFIDENCE_THRESHOLD

        if v_sus or a_sus:
            # Pick the highest severity across both channels
            v_sev = video.get("severity", "none") if video else "none"
            a_sev = audio.get("severity", "none") if audio else "none"
            severity = v_sev if _SEV_RANK.get(v_sev, 0) >= _SEV_RANK.get(a_sev, 0) else a_sev
            color    = RED if severity in ("high", "critical") else YELLOW

            print(color(f"\n!!! ALERT [{severity.upper()}]  {ts}"))
            print(f"    Chunk : {name}")

            if v_sus and video:
                print(f"    VIDEO : [{video.get('confidence', 0):.0%}] "
                      f"{', '.join(video.get('activities', [])) or video.get('description', '')}")

            if a_sus and audio:
                excerpt = audio.get("excerpt", "")
                acts    = ", ".join(audio.get("activities", []))
                print(f"    AUDIO : [{audio.get('confidence', 0):.0%}] {acts}")
                if excerpt:
                    print(f"    Quote : \"{excerpt}\"")

            print()

            entry = {
                "timestamp": datetime.now().isoformat(),
                "chunk": str(chunk),
                "video": video,
                "audio": audio,
            }
            with self._lock:
                self._alerts.append(entry)
            with open("alerts.jsonl", "a") as fh:
                fh.write(json.dumps(entry) + "\n")

        else:
            # Quiet OK line — show both channels concisely
            parts = []
            if video:
                vd = video.get("description", "")[:55]
                parts.append(f"video: {vd}")
            if audio:
                ad = audio.get("description", "")
                if "skipped" not in ad and "error" not in ad:
                    parts.append(f"audio: {ad[:55]}")
            line = "  |  ".join(parts) if parts else "—"
            print(f"  {DIM(ts)}  {GREEN('OK')}  {DIM(name)}  {line}")

    # ------------------------------------------------------------------ #
    #  Header / summary                                                    #
    # ------------------------------------------------------------------ #

    def _print_header(self):
        video_status = "enabled" if config.VIDEO_ENABLED else "disabled"
        audio_status = "enabled" if config.AUDIO_ENABLED else "disabled"
        channels = " + ".join(filter(None, [
            "video" if config.VIDEO_ENABLED else None,
            "audio" if config.AUDIO_ENABLED else None,
        ])) or "none"
        print(BOLD("=" * 64))
        print(BOLD(f"  Election Room Monitor  ({channels})"))
        print(BOLD("=" * 64))
        print(f"  Input    : {self.input_source}")
        print(f"  Model    : {config.LM_STUDIO_MODEL}")
        print(f"  Endpoint : {config.LM_STUDIO_BASE_URL}")
        print(f"  Chunk    : {config.CHUNK_DURATION}s  |  "
              f"Video: {video_status}  |  "
              f"Audio: {audio_status}")
        if config.AUDIO_ENABLED:
            print(f"  Whisper  : {config.WHISPER_MODEL} on {config.WHISPER_DEVICE}")
        print(BOLD("-" * 64))
        print(DIM("  Press Ctrl+C to stop.\n"))

    def _print_summary(self):
        n = len(self._alerts)
        print()
        print(BOLD("=" * 64))
        if n == 0:
            print(GREEN("  No suspicious activity detected."))
        else:
            print(RED(f"  {n} alert(s) recorded — see alerts.jsonl"))
            for a in self._alerts:
                ts    = a.get("timestamp", "")[:19]
                vsev  = (a.get("video") or {}).get("severity", "none")
                asev  = (a.get("audio") or {}).get("severity", "none")
                sev   = vsev if _SEV_RANK.get(vsev, 0) >= _SEV_RANK.get(asev, 0) else asev
                vact  = ", ".join((a.get("video") or {}).get("activities", []))
                aact  = ", ".join((a.get("audio") or {}).get("activities", []))
                label = " | ".join(filter(None, [vact, aact])) or "—"
                print(f"    {ts}  [{sev.upper()}]  {label}")
        print(BOLD("=" * 64))
