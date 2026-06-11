"""
Global configuration for the PSM project.
Override via environment variables or edit directly.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# LLM
MODEL_OLLAMA_NAME: str = os.getenv("PSM_OLLAMA_MODEL", "llama3.2:1b")
MODEL_N_CTX: int = int(os.getenv("PSM_N_CTX", "1024"))
MODEL_TEMPERATURE: float = float(os.getenv("PSM_TEMPERATURE", "0.7"))
MODEL_MAX_TOKENS: int = int(os.getenv("PSM_MAX_TOKENS", "128"))

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH: str = os.getenv("PSM_DB_PATH", str(DATA_DIR / "psm.db"))

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

# ── Big Five estimator ────────────────────────────────────────────────────────
# HuggingFace model for Big Five personality prediction from text
BIG5_MODEL: str = os.getenv(
    "PSM_BIG5_MODEL",
    "Minej/bert-base-personality",   # swap for any compatible HF model
)

# ── Personality state update ──────────────────────────────────────────────────
ALPHA: float = float(os.getenv("PSM_ALPHA", "0.99"))  # EMA smoothing factor

# ── Memory trigger thresholds ─────────────────────────────────────────────────
TRIGGER_TOKEN_LIMIT: int = int(os.getenv("PSM_TRIGGER_TOKENS", "700"))
TRIGGER_CONTEXT_RATIO: float = float(os.getenv("PSM_TRIGGER_CTX", "0.65"))
TRIGGER_TURN_LIMIT: int = int(os.getenv("PSM_TRIGGER_TURNS", "20"))

# ── Default personality (neutral 0.5 for all traits) ─────────────────────────
DEFAULT_PERSONALITY: dict[str, float] = {
    "openness": 0.5,
    "conscientiousness": 0.5,
    "extraversion": 0.5,
    "agreeableness": 0.5,
    "neuroticism": 0.5,
}
