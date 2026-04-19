import os

# ── AI backend — any OpenAI-compatible API ────────────────────────────
#
# Provider examples:
#
#   LM Studio (local)
#     AI_BASE_URL = "http://localhost:1234/v1"
#     AI_MODEL    = "openai/gpt-oss-20b"          # text
#     AI_MODEL    = "llava-v1.6-mistral-7b"        # vision
#
#   LM Studio (remote machine)
#     AI_BASE_URL = "http://192.168.0.117:1234/v1"
#
#   Ollama (local)
#     AI_BASE_URL = "http://localhost:11434/v1"
#     AI_MODEL    = "llama3.2"                     # text
#     AI_MODEL    = "llava"                        # vision
#     AI_MODEL    = "moondream"                    # vision (lightweight)
#     AI_API_KEY  = "ollama"                       # Ollama ignores the key
#
#   OpenAI
#     AI_BASE_URL = "https://api.openai.com/v1"
#     AI_MODEL    = "gpt-4o"
#     AI_API_KEY  = "sk-..."                       # real key required
#
# Whisper audio transcription is attempted via /v1/audio/transcriptions.
# LM Studio and OpenAI support it; Ollama does NOT (falls back to local Whisper).
# ─────────────────────────────────────────────────────────────────────

# AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:1234/v1")
AI_BASE_URL = os.getenv("AI_BASE_URL", "http://192.168.0.117:1234/v1")
AI_MODEL    = os.getenv("AI_MODEL",    "openai/gpt-oss-20b")
AI_API_KEY  = os.getenv("AI_API_KEY",  "lm-studio")  # provider ignores it but SDK requires one
AI_TIMEOUT  = int(os.getenv("AI_TIMEOUT", "120"))     # seconds — increase for slow remote machines

# Video segmentation
CHUNK_DURATION   = 10       # seconds per chunk
CHUNKS_DIR       = "chunks" # where ffmpeg writes segments
FRAMES_DIR       = "frames" # where extracted frames land
FRAMES_PER_CHUNK = 3        # frames to sample from each 10-second chunk

# Video / Audio toggles
VIDEO_ENABLED = os.getenv("VIDEO_ENABLED", "false").lower() in ("1", "true", "yes")
AUDIO_ENABLED = os.getenv("AUDIO_ENABLED", "true").lower()  in ("1", "true", "yes")
AUDIO_DIR     = "audio"     # temp WAV files (deleted after analysis)

# Whisper transcription — two paths (tried in order):
#   1. AI backend audio endpoint (/v1/audio/transcriptions)
#      → works with LM Studio (load a Whisper model) and OpenAI
#      → Ollama does NOT support this endpoint — skips to path 2
#   2. faster-whisper (local CPU fallback)
#      → faster-whisper/CTranslate2 supports: "cpu" | "cuda" (NVIDIA only)
#      → AMD/Vulkan is NOT supported — use "cpu" on AMD/Mac machines
# Options: "tiny" | "base" | "small" | "medium" | "large-v3"
WHISPER_MODEL   = os.getenv("WHISPER_MODEL",   "medium")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE",  "cpu")   # AMD/Vulkan not supported
AUDIO_LANGUAGE  = os.getenv("AUDIO_LANGUAGE",  "bg")    # ISO 639-1 — "bg" = Bulgarian

# Analysis
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.5  # flag alert if confidence >= this
