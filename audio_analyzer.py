"""
Audio analyzer — transcribes each 10-second chunk and asks the LM
whether the speech contains suspicious election-fraud activity.

Transcription strategy (tried in order):
  1. LM Studio /v1/audio/transcriptions  (if a Whisper model is loaded)
  2. faster-whisper                       (pip install faster-whisper)
  3. openai-whisper                       (pip install openai-whisper)
  4. Skip with a logged warning

After transcription the plain-text transcript is sent to the same LM
used for video analysis, with an election-specific prompt.
"""

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

from openai import OpenAI

import config

log = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────── #

# Primes faster-whisper with Bulgarian election vocabulary.
# This is the single most effective way to prevent Russian mis-detection —
# Whisper sees real Bulgarian words first and anchors to that language.
_WHISPER_PROMPT = (
    "Преброяване на бюлетини. Секционна избирателна комисия. "
    "Протокол. Урна. Действителни и недействителни бюлетини. "
    "Преференции. Партии и коалиции. Изборен ден."
)

_SYSTEM = """You are a security monitor for a Bulgarian election ballot-counting room.
The transcript is in BULGARIAN. Do NOT interpret it as Russian or any other language.
Evaluate it directly in Bulgarian — do not translate.
Flag any speech that violates official counting rules or indicates fraud.

WRITING & DOCUMENT TAMPERING
Instructions to write on, alter, or forge ballots or protocols:
"напиши това", "смени числото", "добави гласове", "оправи го", "зачеркни", "задраскай"

REMOVING OR CONCEALING BALLOTS
Instructions to remove, hide, or destroy election materials:
"вземи ги", "скрий го", "изхвърли", "сложи в чантата", "махни ги", "прибери бюлетините"

FRAUD COORDINATION
Secretive coordination language suggesting planned fraud:
"преди да видят", "не им казвай", "бързо", "никой не гледа", "докато не са тук", "тихо"

INVALID BALLOT MANIPULATION
Making valid ballots invalid, or fraudulently classifying them:
"направи я невалидна", "развали я", "сложи знак", "бракувай я", "недействителна",
"тури я при невалидните", "тя не се брои", "не важи", "хвърли при развалените"
NOTE: machine-vote ballots (от машинно гласуване) can NEVER be invalid — flag immediately if discussed.

VOTE DIRECTION & VOTE BUYING
Instructing how to vote, or buying votes:
"за кого ще пишеш", "пиши за ...", "гласувай за ...", "кого трябва да пишем",
"платиха ми да пиша за", "дадоха ми пари", "договорихме се за", "всички пишем за",
"покажи ми как да отбележа", "евро", "пари", "да помогнем", "помощ", "попълни"

MIXING BALLOTS FROM DIFFERENT URNS
Ballots from different urns must be counted separately:
"смесете ги", "сложи от другата урна", "смесете двете купчини"

COUNTING IRREGULARITIES
Announced verbal count not matching what is written, or skipping the initial total count:
Numbers called out that seem inconsistent, or "не броим в началото", "пропуснете общия брой"

UNAUTHORISED PRESENCE
Letting in people not allowed during counting:
"влез", "остани вътре", "не ги пускай навън", discussion of outsiders being present

EARLY STREAM TERMINATION
Stopping the broadcast before the protocol is signed and ballot bags sealed:
"спри камерата", "изключи видеото", "спри излъчването", "изключете устройството"
NOTE: the stream must stay on until: protocol signed → results announced → bags sealed.

Reply with ONLY a valid JSON object — no markdown, no extra text:
{
  "suspicious": <true|false>,
  "confidence": <0.0-1.0>,
  "severity": <"none"|"low"|"medium"|"high"|"critical">,
  "activities": [<short descriptions, empty list if none>],
  "description": "<one sentence summary of what was said>",
  "excerpt": "<the most suspicious quote verbatim, or empty string>"
}"""

_USER = """Transcript of a 10-second election room audio segment:

\"\"\"{transcript}\"\"\"

Is there any suspicious activity in this speech?"""

_EMPTY_TRANSCRIPT = "[no speech detected]"

# ─────────────────────────────────────────────────────────────────────── #

# Shared semaphore with video analyzer — only one model call at a time
from analyzer import _model_sem   # noqa: E402  (circular-safe: no class init)


class AudioAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LM_STUDIO_BASE_URL,
            api_key=config.LM_STUDIO_API_KEY,
            timeout=config.LM_STUDIO_TIMEOUT,
        )
        self.audio_dir = Path(config.AUDIO_DIR)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._transcriber      = None   # lazy-loaded
        self._transcriber_name = None
        self._transcribe_lock  = threading.Lock()
        self._lmstudio_audio   = None   # True/False once probed

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def analyze(self, chunk: Path) -> dict:
        log.info("Audio analysis of %s", chunk.name)

        # Audio-only mode: chunk IS already a WAV — no extraction needed
        if chunk.suffix.lower() == ".wav":
            wav = chunk
            owns_wav = False
        else:
            wav = self._extract_audio(chunk)
            owns_wav = True

        if wav is None:
            return self._skip_result(chunk, "audio extraction failed")

        transcript = self._transcribe(wav)
        if owns_wav:
            wav.unlink(missing_ok=True)

        if not transcript or transcript == _EMPTY_TRANSCRIPT:
            log.debug("No speech in %s", chunk.name)
            return self._skip_result(chunk, "no speech detected")

        log.debug("Transcript [%s]: %s", chunk.name, transcript[:120])
        result = self._query_model(transcript)
        result["chunk"]      = str(chunk)
        result["chunk_name"] = chunk.name
        result["transcript"] = transcript
        result["source"]     = "audio"
        return result

    # ------------------------------------------------------------------ #
    #  Audio extraction                                                    #
    # ------------------------------------------------------------------ #

    def _extract_audio(self, chunk: Path) -> Optional[Path]:
        out = self.audio_dir / f"{chunk.stem}.wav"
        cmd = [
            "ffmpeg",
            "-i", str(chunk),
            "-vn",                          # no video
            "-acodec", "pcm_s16le",         # 16-bit PCM — what Whisper expects
            "-ar", "16000",                 # 16 kHz
            "-ac", "1",                     # mono
            str(out),
            "-y",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
            log.debug("Audio extraction failed or silent: %s", r.stderr[-200:])
            return None
        return out

    # ------------------------------------------------------------------ #
    #  Transcription                                                       #
    # ------------------------------------------------------------------ #

    def _transcribe(self, wav: Path) -> str:
        # 1. Try LM Studio's audio endpoint
        if self._lmstudio_audio is not False:
            result = self._transcribe_lmstudio(wav)
            if result is not None:
                return result

        # 2. Try faster-whisper / openai-whisper
        transcriber = self._get_local_transcriber()
        if transcriber:
            return transcriber(wav)

        log.warning(
            "No transcription engine available. "
            "Install faster-whisper:  pip install faster-whisper"
        )
        return _EMPTY_TRANSCRIPT

    def _transcribe_lmstudio(self, wav: Path) -> Optional[str]:
        try:
            with open(wav, "rb") as fh:
                resp = self.client.audio.transcriptions.create(
                    model="whisper-1",          # LM Studio maps this to the loaded model
                    file=fh,
                    language=config.AUDIO_LANGUAGE,
                )
            self._lmstudio_audio = True
            return resp.text.strip() or _EMPTY_TRANSCRIPT
        except Exception as exc:
            err = str(exc).lower()
            if any(k in err for k in ("not found", "404", "no model", "unsupported")):
                if self._lmstudio_audio is None:
                    log.info("LM Studio audio endpoint not available — using local Whisper.")
                self._lmstudio_audio = False
            else:
                log.debug("LM Studio audio transcription error: %s", exc)
                self._lmstudio_audio = False
            return None

    def _get_local_transcriber(self):
        with self._transcribe_lock:
            if self._transcriber is not None:
                return self._transcriber

            # faster-whisper
            try:
                from faster_whisper import WhisperModel   # type: ignore
                log.info("Loading faster-whisper '%s' on %s …",
                         config.WHISPER_MODEL, config.WHISPER_DEVICE)
                _model = WhisperModel(
                    config.WHISPER_MODEL,
                    device=config.WHISPER_DEVICE,
                    compute_type="int8",
                )

                def _fw(wav: Path) -> str:
                    segs, _ = _model.transcribe(
                        str(wav),
                        beam_size=10,
                        language=config.AUDIO_LANGUAGE,
                        # Bulgarian election vocabulary primes the model and
                        # prevents it from drifting to Russian (same Cyrillic script).
                        initial_prompt=_WHISPER_PROMPT,
                        vad_filter=True,
                        vad_parameters={"min_silence_duration_ms": 300},
                        temperature=0.0,          # greedy — stable for short clips
                        condition_on_previous_text=False,
                    )
                    return " ".join(s.text for s in segs).strip() or _EMPTY_TRANSCRIPT

                self._transcriber      = _fw
                self._transcriber_name = "faster-whisper"
                log.info("faster-whisper ready.")
                return self._transcriber
            except ImportError:
                pass

            # openai-whisper
            try:
                import whisper as _whisper   # type: ignore
                log.info("Loading openai-whisper '%s' …", config.WHISPER_MODEL)
                _model = _whisper.load_model(config.WHISPER_MODEL)

                def _ow(wav: Path) -> str:
                    result = _model.transcribe(
                        str(wav),
                        language=config.AUDIO_LANGUAGE,
                        fp16=False,
                    )
                    return result["text"].strip() or _EMPTY_TRANSCRIPT

                self._transcriber      = _ow
                self._transcriber_name = "openai-whisper"
                log.info("openai-whisper ready.")
                return self._transcriber
            except ImportError:
                pass

            # Nothing available
            self._transcriber = False
            return None

    # ------------------------------------------------------------------ #
    #  LM analysis of transcript                                          #
    # ------------------------------------------------------------------ #

    def _query_model(self, transcript: str) -> dict:
        user_content = _USER.format(transcript=transcript)
        with _model_sem:
            # Try with system message first; fall back to flat if jinja bug
            for msgs in [
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user",   "content": user_content}],
                [{"role": "user",   "content": _SYSTEM + "\n\n" + user_content}],
            ]:
                try:
                    resp = self.client.chat.completions.create(
                        model=config.LM_STUDIO_MODEL,
                        messages=msgs,
                        temperature=0.1,
                        max_tokens=350,
                    )
                    return self._parse_json(resp.choices[0].message.content or "")
                except Exception as exc:
                    err = str(exc)
                    if "jinja" in err.lower() or "rendering prompt" in err.lower():
                        if len(msgs) > 1 or msgs[0]["role"] == "system":
                            log.info("Jinja template error — retrying with flat message.")
                            continue   # try flat format
                    log.error("Audio model query failed: %s", exc)
                    return self._error_result(None, err)
        return self._error_result(None, "all retry attempts failed")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(text: str) -> dict:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return __import__("json").loads(text[start:end])
            except __import__("json").JSONDecodeError:
                pass
        return {
            "suspicious":  False,
            "confidence":  0.0,
            "severity":    "none",
            "activities":  [],
            "description": text[:200] if text else "no response",
            "excerpt":     "",
        }

    @staticmethod
    def _error_result(chunk, reason: str) -> dict:
        return {
            "suspicious":  False,
            "confidence":  0.0,
            "severity":    "none",
            "activities":  [],
            "description": f"[audio error: {reason}]",
            "excerpt":     "",
        }

    @staticmethod
    def _skip_result(chunk: Path, reason: str) -> dict:
        return {
            "suspicious":  False,
            "confidence":  0.0,
            "severity":    "none",
            "activities":  [],
            "description": f"[audio skipped: {reason}]",
            "excerpt":     "",
            "chunk":       str(chunk),
            "chunk_name":  chunk.name,
            "source":      "audio",
        }
