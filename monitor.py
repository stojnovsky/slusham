"""
ElectionMonitor — ties segmenter, video analyzer, and audio analyzer.

Each session has its own directory (sessions/<title>/) containing:
  alerts.jsonl   — one JSON line per alert, written immediately on detection
  monitor.log    — full application log (configured in main.py)
  chunks/        — temporary audio/video segments

Alerts are written and flushed to disk the moment they are detected,
not at process exit.
"""

import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
GREEN  = lambda t: _c("92",   t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)

_SEV_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class ElectionMonitor:
    def __init__(
        self,
        input_source: str,
        session_dir: Path,
        title: str,
        max_workers: int = 4,
    ):
        self.input_source = input_source
        self.session_dir  = Path(session_dir)
        self.title        = title

        self._alerts_path = self.session_dir / "alerts.jsonl"
        self._chunks_dir  = self.session_dir / "chunks"
        self._chunks_dir.mkdir(parents=True, exist_ok=True)

        self._video_az = ChunkAnalyzer() if config.VIDEO_ENABLED else None
        self._audio_az = AudioAnalyzer() if config.AUDIO_ENABLED else None
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analyzer"
        )
        self._alerts: list[dict] = []
        self._lock    = threading.Lock()
        # Keep the alerts file open for the whole session — writes are
        # flushed immediately so the file is always up-to-date on disk.
        self._alerts_fh = open(self._alerts_path, "a", encoding="utf-8", buffering=1)

        self._segmenter = VideoSegmenter(
            input_source   = input_source,
            output_dir     = str(self._chunks_dir),
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
            self._alerts_fh.close()
            self._print_summary()

    # ------------------------------------------------------------------ #
    #  Per-chunk pipeline                                                  #
    # ------------------------------------------------------------------ #

    def _on_chunk_ready(self, chunk: Path):
        self._executor.submit(self._process_chunk, chunk)

    def _process_chunk(self, chunk: Path):
        inner   = ThreadPoolExecutor(max_workers=2, thread_name_prefix="av")
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

        v_sus = (video and video.get("suspicious", False) and
                 video.get("confidence", 0) >= config.SUSPICIOUS_CONFIDENCE_THRESHOLD)
        a_sus = (audio and audio.get("suspicious", False) and
                 audio.get("confidence", 0) >= config.SUSPICIOUS_CONFIDENCE_THRESHOLD)

        if v_sus or a_sus:
            v_sev    = video.get("severity", "none") if video else "none"
            a_sev    = audio.get("severity", "none") if audio else "none"
            severity = v_sev if _SEV_RANK.get(v_sev, 0) >= _SEV_RANK.get(a_sev, 0) else a_sev
            color    = RED if severity in ("high", "critical") else YELLOW

            print(color(f"\n!!! ALERT [{severity.upper()}]  {ts}"), flush=True)
            print(f"    Session : {self.title}", flush=True)
            print(f"    Chunk   : {name}", flush=True)

            if v_sus and video:
                print(
                    f"    VIDEO   : [{video.get('confidence', 0):.0%}] "
                    f"{', '.join(video.get('activities', [])) or video.get('description', '')}",
                    flush=True,
                )
            if a_sus and audio:
                acts    = ", ".join(audio.get("activities", []))
                excerpt = audio.get("excerpt", "")
                print(f"    AUDIO   : [{audio.get('confidence', 0):.0%}] {acts}", flush=True)
                if excerpt:
                    print(f"    Quote   : \"{excerpt}\"", flush=True)
            print(flush=True)

            entry = {
                "session":   self.title,
                "timestamp": datetime.now().isoformat(),
                "chunk":     str(chunk),
                "video":     video,
                "audio":     audio,
            }
            with self._lock:
                self._alerts.append(entry)
                # buffering=1 (line-buffered) means each write is flushed immediately
                self._alerts_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        else:
            parts = []
            if video:
                parts.append(f"video: {video.get('description', '')[:55]}")
            if audio:
                ad = audio.get("description", "")
                if "skipped" not in ad and "error" not in ad:
                    parts.append(f"audio: {ad[:55]}")
            line = "  |  ".join(parts) if parts else "—"
            print(f"  {DIM(ts)}  {GREEN('OK')}  {DIM(name)}  {line}", flush=True)

    # ------------------------------------------------------------------ #
    #  Header / summary                                                    #
    # ------------------------------------------------------------------ #

    def _print_header(self):
        video_status = "enabled"  if config.VIDEO_ENABLED else "disabled"
        audio_status = "enabled"  if config.AUDIO_ENABLED else "disabled"
        channels = " + ".join(filter(None, [
            "video" if config.VIDEO_ENABLED else None,
            "audio" if config.AUDIO_ENABLED else None,
        ])) or "none"
        print(BOLD("=" * 64))
        print(BOLD(f"  Election Room Monitor  ({channels})"))
        print(BOLD("=" * 64))
        print(f"  Session  : {self.title}")
        print(f"  Alerts   : {self._alerts_path}")
        print(f"  Log      : {self.session_dir / 'monitor.log'}")
        print(f"  Input    : {self.input_source}")
        print(f"  Model    : {config.LM_STUDIO_MODEL}")
        print(f"  Endpoint : {config.LM_STUDIO_BASE_URL}")
        print(f"  Chunk    : {config.CHUNK_DURATION}s  |  "
              f"Video: {video_status}  |  Audio: {audio_status}")
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
            print(RED(f"  {n} alert(s) recorded → {self._alerts_path}"))
            for a in self._alerts:
                ts   = a.get("timestamp", "")[:19]
                vsev = (a.get("video") or {}).get("severity", "none")
                asev = (a.get("audio") or {}).get("severity", "none")
                sev  = vsev if _SEV_RANK.get(vsev, 0) >= _SEV_RANK.get(asev, 0) else asev
                vact = ", ".join((a.get("video") or {}).get("activities", []))
                aact = ", ".join((a.get("audio") or {}).get("activities", []))
                label = " | ".join(filter(None, [vact, aact])) or "—"
                print(f"    {ts}  [{sev.upper()}]  {label}")
        print(BOLD("=" * 64))
