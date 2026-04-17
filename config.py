import os

# LM Studio — OpenAI-compatible endpoint
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "openai/gpt-oss-20b")
# LM_STUDIO_MODEL    = os.getenv("LM_STUDIO_MODEL",    "mys/ggml_bakllava-1")
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
# Whisper model used by faster-whisper when LM Studio has no audio endpoint.
# Options: "tiny", "base", "small", "medium", "large-v3"
WHISPER_MODEL   = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE  = os.getenv("WHISPER_DEVICE", "cpu")   # "cpu" or "cuda"
AUDIO_LANGUAGE  = os.getenv("AUDIO_LANGUAGE", "bg")    # bg = Bulgarian

# Analysis
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.5   # flag if confidence >= this
