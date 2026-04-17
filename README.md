# Election Room Video Monitor

Watches a live camera or recorded video from an election counting room and flags suspicious activity in real time using a local AI model via **LM Studio**.

Each 10-second chunk of video is analysed on two independent channels:

| Channel | How it works |
|---------|-------------|
| **Video** | ffmpeg extracts key frames → sent to a vision model (LLaVA, BakLLaVA, moondream …) |
| **Audio** | faster-whisper transcribes Bulgarian speech → transcript sent to a text model |

Alerts are printed to the terminal and appended to `alerts.jsonl`.

---

## Requirements

| Dependency | Install |
|------------|---------|
| Python 3.11+ | [python.org](https://python.org) |
| ffmpeg | `brew install ffmpeg` (Mac) / `apt install ffmpeg` (Linux) |
| LM Studio | [lmstudio.ai](https://lmstudio.ai) — run the local server on port 1234 |

---

## Setup

```bash
# 1. Clone / enter the project
cd video-elections

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# 3. Make the run script executable (one time)
chmod +x run.sh
```

---

## LM Studio setup

1. Open **LM Studio** and go to the **Local Server** tab (`<->` icon).
2. Load a model:
   - **Audio-only mode (default):** any text model works — e.g. `openai/gpt-oss-20b`
   - **Video mode:** load a vision model — e.g. `llava-v1.6-mistral-7b`, `moondream2`, `bakllava-1`
3. Click **Start Server** (default port: `1234`).
4. Verify the server is running:
   ```bash
   curl http://localhost:1234/v1/models
   ```

> **LLaVA jinja template bug:** if you see `"Unexpected escaped character: s"` errors, go to  
> My Models → your model → Prompt Template and switch to the `lmstudio-community` version of the model.  
> The monitor retries automatically as a workaround, but the lmstudio-community version is more stable.

---

## Running

```bash
# Local video file
./run.sh test/recording.mp4

# Webcam (device index 0)
./run.sh webcam

# IP camera via RTSP
./run.sh rtsp://192.168.1.100/stream

# Debug mode — shows ffmpeg output and API details
./run.sh test/recording.mp4 --debug

# Limit concurrent analysis threads (default: 4)
./run.sh rtsp://... --workers 2
```

Stop at any time with **Ctrl+C**. A summary of all alerts is printed on exit.

---

## Configuration

All settings are in [`config.py`](config.py). Most can also be overridden with environment variables.

### Toggle video / audio analysis

```python
# config.py
VIDEO_ENABLED = False   # True  → analyse frames with vision model
AUDIO_ENABLED = True    # False → skip audio transcription
```

Or per-run:
```bash
VIDEO_ENABLED=true  ./run.sh recording.mp4
AUDIO_ENABLED=false ./run.sh recording.mp4
```

### Whisper model (audio transcription)

```python
WHISPER_MODEL  = "medium"   # tiny | base | small | medium | large-v3
WHISPER_DEVICE = "cpu"      # cpu | cuda
AUDIO_LANGUAGE = "bg"       # ISO 639-1 code — "bg" = Bulgarian
```

Larger models are more accurate but slower on CPU:

| Model | Size | Bulgarian quality |
|-------|------|-------------------|
| `tiny` | 75 MB | poor |
| `base` | 145 MB | poor |
| `small` | 466 MB | acceptable |
| `medium` | 1.5 GB | **good (recommended)** |
| `large-v3` | 3 GB | best |

The model is downloaded automatically from Hugging Face on first use and then cached locally.

### Change the LM Studio model or endpoint

```bash
LM_STUDIO_MODEL=llava-v1.6-mistral-7b ./run.sh recording.mp4
LM_STUDIO_BASE_URL=http://192.168.1.50:1234/v1 ./run.sh recording.mp4
```

### Chunk duration and frames

```python
CHUNK_DURATION   = 10   # seconds per segment
FRAMES_PER_CHUNK = 3    # frames extracted from each segment for vision
```

### Alert confidence threshold

```python
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.5   # 0.0–1.0
```

Lower = more sensitive (more false positives). Higher = stricter.

---

## What the AI looks for

### Video (vision model)
1. Someone writing on a ballot with a pen, pencil, or marker
2. Someone altering entries in official election protocols
3. Someone removing, hiding, or concealing ballots
4. People blocking the camera view of documents
5. Unauthorized handling or disposal of election materials
6. Deliberately defacing valid ballots to make them invalid (tearing, crossing out)
7. Incorrectly sorting valid ballots into the invalid/spoiled pile
8. Someone pointing at or guiding another person's hand to direct a vote
9. Someone photographing a marked ballot (proof-of-vote for payment)

### Audio (Whisper + text model — Bulgarian)
1. Instructions to write on, alter, or forge ballots/protocols  
   _"напиши това", "смени числото", "добави гласове"_
2. Instructions to remove or conceal election documents  
   _"вземи ги", "скрий го", "изхвърли", "сложи в чантата"_
3. Coordination language implying fraud  
   _"преди да видят", "не им казвай", "бързо", "никой не гледа"_
4. Coercion or pressure on election workers
5. Discussion of falsifying counts or swapping ballots
6. Making valid ballots invalid  
   _"направи я невалидна", "развали я", "зачеркни", "бракувай я"_
7. Fraudulent classification of ballots  
   _"тури я при невалидните", "тя не се брои", "хвърли при развалените"_
8. Vote direction or vote buying  
   _"за кого ще пишеш", "пиши за …", "платиха ми да пиша за", "дадоха ми пари"_

---

## Output

### Terminal

```
  18:42:11  OK  chunk_0003.mp4  audio: Обсъждат процедурата по броене.

!!! ALERT [HIGH]  18:42:34
    Chunk  : chunk_0007.mp4
    AUDIO  : [87%] vote direction, vote buying
    Quote  : "пиши за тях, така се договорихме"
```

### alerts.jsonl

Every alert is appended as a JSON line:

```json
{
  "timestamp": "2026-04-17T18:42:34.123456",
  "chunk": "chunks/chunk_0007.mp4",
  "video": null,
  "audio": {
    "suspicious": true,
    "confidence": 0.87,
    "severity": "high",
    "activities": ["vote direction", "vote buying"],
    "description": "Speaker instructs someone to vote for a specific party.",
    "excerpt": "пиши за тях, така се договорихме",
    "source": "audio"
  }
}
```

---

## Project structure

```
video-elections/
├── main.py            # CLI entry point
├── monitor.py         # Orchestration — runs video + audio per chunk
├── segmenter.py       # ffmpeg wrapper — splits stream into 10s chunks
├── analyzer.py        # Frame extraction + vision model query
├── audio_analyzer.py  # Audio extraction + Whisper + text model query
├── config.py          # All settings (override with env vars)
├── run.sh             # Convenience wrapper — uses venv/bin/python
├── requirements.txt
├── chunks/            # Temporary video segments (auto-created)
├── frames/            # Temporary frame JPEGs (auto-created, deleted after use)
├── audio/             # Temporary WAV files (auto-created, deleted after use)
├── alerts.jsonl       # Persistent alert log
└── monitor.log        # Full application log
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'openai'`**  
Use `./run.sh` instead of `python main.py`, or activate the venv first:  
```bash
source venv/bin/activate
python main.py ...
```

**`Connection error` to LM Studio**  
LM Studio server is not running. Open LM Studio → Local Server tab → Start Server.

**Audio transcription is garbled**  
Switch to a larger Whisper model:  
```bash
WHISPER_MODEL=large-v3 ./run.sh recording.mp4
```

**`Unexpected escaped character: s` from LM Studio**  
Jinja template bug in the loaded model. The monitor retries automatically.  
Permanent fix: load the `lmstudio-community` version of the model in LM Studio.

**No frames extracted from chunks**  
Usually caused by copy-segmented files with missing keyframes. The monitor falls back to the first available frame automatically. If it persists, re-encode the source:  
```bash
ffmpeg -i original.mkv -c:v libx264 -c:a aac reencoded.mp4
./run.sh reencoded.mp4
```
