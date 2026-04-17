"""
Frame extractor + LM Studio analyzer.

Two modes are chosen automatically on first use:
  • vision  — model accepts image_url content (e.g. LLaVA, moondream)
  • text     — model is text-only; we describe each chunk via ffmpeg
                motion/scene-change stats and ask the model to judge

Frame extraction uses explicit seek (-ss before -i) which is far more
reliable than the fps filter on copy-segmented files.
"""

import base64
import json
import logging
import subprocess
import shutil
import threading
from pathlib import Path
from typing import Optional

from openai import OpenAI

import config

log = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────── #

_SYSTEM = """You are a security observer monitoring an election counting room.
Your only job is to detect potential fraud or irregularities.

Watch for:
1. Someone writing on a ballot or bulletin with a pen, pencil, or marker
2. Someone adding or altering entries in official election protocols
3. Someone removing, hiding, folding, or concealing ballots
4. People blocking the camera view of documents
5. Unauthorized handling or disposal of election materials
6. Someone deliberately defacing or marking valid ballots to make them invalid
   (tearing, crossing out, adding stray marks so the ballot cannot be counted)
7. Someone incorrectly sorting valid ballots into the invalid/spoiled pile
8. Suspicious separation or stacking of ballots that may indicate invalid ballots
   being mixed with valid ones or vice versa
9. Someone pointing at, showing, or guiding another person's hand toward a specific
   position on a ballot — indicating vote direction or coercion
10. Someone photographing or filming a marked ballot (proof-of-vote for vote buying)

Reply with ONLY a valid JSON object — no markdown, no extra text:
{
  "suspicious": <true|false>,
  "confidence": <0.0-1.0>,
  "severity": <"none"|"low"|"medium"|"high"|"critical">,
  "activities": [<short descriptions, empty list if none>],
  "description": "<one sentence summary of what is visible>"
}"""

_USER_VISION = (
    "These {n} frames are from a 10-second election room video segment. "
    "Analyze them for suspicious activity."
)

_USER_TEXT = """These motion statistics are from a 10-second election room video segment.

Chunk: {chunk}
Scene changes detected: {scenes}
Max inter-frame motion score: {motion:.3f}
Activity regions (pixels with movement): {regions}

Given these signals, assess whether suspicious activity is likely (writing on ballots,
altering protocols, removing ballots). Respond with the JSON format described."""

# ─────────────────────────────────────────────────────────────────────── #

# Global semaphore — keeps concurrent model calls to 1 so the model
# doesn't crash when many chunks arrive at once (e.g. from a file).
_model_sem = threading.Semaphore(1)


class _JinjaTemplateError(Exception):
    """Raised when LM Studio reports a jinja2 template rendering error."""


class ChunkAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
        )
        self.frames_root = Path(config.FRAMES_DIR)
        self.frames_root.mkdir(parents=True, exist_ok=True)

        # None = not yet determined, True/False once probed
        self._vision_supported: Optional[bool] = None
        self._vision_probe_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def analyze(self, chunk: Path) -> dict:
        log.info("Analyzing %s", chunk.name)

        frames = self._extract_frames(chunk)
        if not frames:
            log.warning("No frames extracted from %s — trying text mode", chunk.name)

        with _model_sem:
            if frames and self._use_vision():
                result = self._query_vision(frames, chunk)
            else:
                result = self._query_text(chunk)

        result["chunk"]      = str(chunk)
        result["chunk_name"] = chunk.name

        self._cleanup_frames(chunk)
        return result

    # ------------------------------------------------------------------ #
    #  Vision-support probe                                               #
    # ------------------------------------------------------------------ #

    def _use_vision(self) -> bool:
        """Return True if vision mode is confirmed or not yet tested."""
        with self._vision_probe_lock:
            if self._vision_supported is None:
                return True          # optimistic — first real call will tell us
            return self._vision_supported

    def _mark_vision(self, supported: bool):
        with self._vision_probe_lock:
            if self._vision_supported != supported:
                self._vision_supported = supported
                if not supported:
                    log.warning(
                        "Vision not supported by '%s'. "
                        "Switching to text/motion mode.\n"
                        "  To enable vision, load a vision model in LM Studio\n"
                        "  (e.g. llava-v1.6-mistral-7b, moondream2, bakllava).",
                        config.LM_STUDIO_MODEL,
                    )
                else:
                    log.info("Vision mode confirmed for '%s'.", config.LM_STUDIO_MODEL)

    # ------------------------------------------------------------------ #
    #  Frame extraction — seek-based (more reliable than fps filter)      #
    # ------------------------------------------------------------------ #

    def _extract_frames(self, chunk: Path) -> list[Path]:
        frame_dir = self.frames_root / chunk.stem
        frame_dir.mkdir(parents=True, exist_ok=True)

        n   = config.FRAMES_PER_CHUNK
        dur = config.CHUNK_DURATION
        # Spread timestamps evenly, e.g. [2, 5, 8] for 10 s / 3 frames
        step = dur / (n + 1)
        timestamps = [step * (i + 1) for i in range(n)]

        frames: list[Path] = []
        for i, t in enumerate(timestamps):
            out = frame_dir / f"frame_{i:02d}.jpg"
            # Accurate seek: -ss AFTER -i reads every frame until target time.
            # Fast seek (ss before -i) often misses in copy-segmented files.
            cmd = [
                "ffmpeg",
                "-i", str(chunk),
                "-ss", f"{t:.2f}",
                "-frames:v", "1",
                "-q:v", "3",
                str(out),
                "-y",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                frames.append(out)
            else:
                log.info("Frame seek t=%.1fs failed for %s: %s",
                         t, chunk.name, r.stderr.splitlines()[-1] if r.stderr else "no output")

        # Last-resort fallback: grab the very first decoded frame
        if not frames:
            out = frame_dir / "frame_00.jpg"
            cmd = ["ffmpeg", "-i", str(chunk), "-frames:v", "1", "-q:v", "3", str(out), "-y"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                frames.append(out)
                log.info("Fallback first-frame extraction succeeded for %s", chunk.name)

        log.info("Extracted %d/%d frames from %s", len(frames), n, chunk.name)
        return frames

    def _cleanup_frames(self, chunk: Path):
        d = self.frames_root / chunk.stem
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------ #
    #  Vision query                                                        #
    # ------------------------------------------------------------------ #

    def _query_vision(self, frames: list[Path], chunk: Path) -> dict:
        content: list[dict] = [
            {"type": "text", "text": _USER_VISION.format(n=len(frames))}
        ]
        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._b64(f)}"},
            })

        try:
            resp = self._call_model(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": content},
                ],
            )
            self._mark_vision(True)
            return self._parse_json(resp.choices[0].message.content or "")

        except _JinjaTemplateError:
            # Jinja bug in model template — not a vision limitation.
            # Retry with merged system+user message to bypass template rendering.
            return self._query_vision_flat(frames, chunk)

        except Exception as exc:
            err = str(exc)
            # Only mark vision unsupported for actual image-rejection errors
            if any(k in err.lower() for k in ("vision", "image", "multimodal", "no image")):
                self._mark_vision(False)
                return self._query_text(chunk)
            log.error("Vision query failed: %s", exc)
            return self._error_result(chunk, err)

    def _query_vision_flat(self, frames: list[Path], chunk: Path) -> dict:
        """Retry vision with system prompt merged into user message (bypasses jinja)."""
        content: list[dict] = [
            {"type": "text", "text": _SYSTEM + "\n\n" + _USER_VISION.format(n=len(frames))}
        ]
        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self._b64(f)}"},
            })
        try:
            resp = self._call_model(
                messages=[{"role": "user", "content": content}],
            )
            self._mark_vision(True)
            return self._parse_json(resp.choices[0].message.content or "")
        except Exception as exc:
            log.error("Vision flat retry failed: %s", exc)
            return self._error_result(chunk, str(exc))

    # ------------------------------------------------------------------ #
    #  Text fallback — use ffmpeg motion stats as input                   #
    # ------------------------------------------------------------------ #

    def _query_text(self, chunk: Path) -> dict:
        stats = self._motion_stats(chunk)
        user_msg = _USER_TEXT.format(
            chunk   = chunk.name,
            scenes  = stats["scene_changes"],
            motion  = stats["max_motion"],
            regions = stats["active_pixels"],
        )
        try:
            resp = self._call_model(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
            )
            return self._parse_json(resp.choices[0].message.content or "")

        except _JinjaTemplateError:
            # Retry with merged message to bypass template bug
            try:
                resp = self._call_model(
                    messages=[{"role": "user",
                                "content": _SYSTEM + "\n\n" + user_msg}],
                )
                return self._parse_json(resp.choices[0].message.content or "")
            except Exception as exc2:
                log.error("Text flat retry failed: %s", exc2)
                return self._error_result(chunk, str(exc2))

        except Exception as exc:
            log.error("Text query failed: %s", exc)
            return self._error_result(chunk, str(exc))

    # ------------------------------------------------------------------ #
    #  Motion stats via ffmpeg                                             #
    # ------------------------------------------------------------------ #

    def _motion_stats(self, chunk: Path) -> dict:
        """
        Run ffmpeg with scene-change detection and return a small stats dict.
        Uses: select filter with scene threshold + showinfo to count changes.
        """
        cmd = [
            "ffmpeg",
            "-i", str(chunk),
            "-vf", "select='gt(scene,0.15)',metadata=print:file=-",
            "-an", "-f", "null", "-",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        output = r.stderr + r.stdout

        # Count lines that mention "scene_score"
        scene_changes = output.count("scene_score")

        # Use mpdecimate to estimate motion (counts non-dropped frames = motion frames)
        cmd2 = [
            "ffmpeg",
            "-i", str(chunk),
            "-vf", "mpdecimate=hi=64:lo=32:frac=0.33",
            "-an", "-f", "null", "-",
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        kept = r2.stderr.count("keep")
        drop = r2.stderr.count("drop")
        total = kept + drop
        max_motion = kept / total if total > 0 else 0.0

        return {
            "scene_changes": scene_changes,
            "max_motion":    max_motion,
            "active_pixels": f"{max_motion * 100:.1f}% of frames show movement",
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _call_model(self, messages: list[dict], max_tokens: int = 400):
        """
        Call the model, raising _JinjaTemplateError on known template bugs
        so callers can retry with a flattened message format.
        """
        try:
            return self.client.chat.completions.create(
                model=config.LM_STUDIO_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if "jinja" in str(exc).lower() or "rendering prompt" in str(exc).lower():
                if not getattr(self, "_jinja_warned", False):
                    self._jinja_warned = True
                    log.warning(
                        "LM Studio jinja template bug in '%s'.\n"
                        "  Retrying with flat message format (workaround).\n"
                        "  Permanent fix: in LM Studio open My Models → '%s' →\n"
                        "  Prompt Template and switch to the lmstudio-community version.",
                        config.LM_STUDIO_MODEL, config.LM_STUDIO_MODEL,
                    )
                raise _JinjaTemplateError(str(exc)) from exc
            raise

    @staticmethod
    def _b64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode()

    @staticmethod
    def _parse_json(text: str) -> dict:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "suspicious":  False,
            "confidence":  0.0,
            "severity":    "none",
            "activities":  [],
            "description": text[:200] if text else "no response",
        }

    @staticmethod
    def _error_result(chunk: Optional[Path], reason: str) -> dict:
        return {
            "suspicious":  False,
            "confidence":  0.0,
            "severity":    "none",
            "activities":  [],
            "description": f"[error: {reason}]",
            "chunk":       str(chunk),
        }
