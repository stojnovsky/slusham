import os

# LM Studio — OpenAI-compatible endpoint
# LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://192.168.0.117:1234/v1")
LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "openai/gpt-oss-20b")
# LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "openai/gpt-oss-120b")
# LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "llava-v1.6-mistral-7b")
LM_STUDIO_API_KEY  = "lm-studio"   # LM Studio ignores the key but the SDK requires one

# Video segmentation
CHUNK_DURATION  = 10          # seconds per chunk
CHUNKS_DIR      = "chunks"    # where ffmpeg writes segments
FRAMES_DIR      = "frames"    # where extracted frames land
FRAMES_PER_CHUNK = 3          # frames to sample from each 10-second chunk
                               #   → at t=2s, t=5s, t=8s

# Video / Audio toggles
VIDEO_ENABLED   = False   # set True to re-enable frame analysis
AUDIO_ENABLED   = True
AUDIO_DIR       = "audio"       # temp WAV files (deleted after analysis)
# Whisper transcription — two paths (tried in order):
#   1. LM Studio audio endpoint  → uses your RX 9070 via Vulkan automatically
#      Load any Whisper model in LM Studio (e.g. whisper-large-v3) and it
#      will be used via /v1/audio/transcriptions — no config change needed.
#   2. faster-whisper (local fallback) → CPU only on this Mac
#      faster-whisper/CTranslate2 does NOT support AMD or Vulkan.
#      WHISPER_DEVICE valid values: "cpu" | "cuda" (NVIDIA only) | "auto"
#
# Options: "tiny", "base", "small", "medium", "large-v3"
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")   # AMD/Vulkan not supported — use cpu
AUDIO_LANGUAGE  = os.getenv("AUDIO_LANGUAGE", "bg")    # bg = Bulgarian

# Network
# Increase if the remote LM Studio machine is slow to respond.
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "120"))  # seconds

# Analysis
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.5   # flag if confidence >= this
