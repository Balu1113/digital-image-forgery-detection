"""Central configuration for the image forgery detection application.

All values can be overridden through environment variables so the same code
runs unchanged in development, CI and production.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Runtime configuration."""

    SECRET_KEY = os.environ.get("SESSION_SECRET", "dev-secret-key")

    MODEL_PATH = os.environ.get(
        "MODEL_PATH", str(BASE_DIR / "image_real_or_fake.keras")
    )

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))

    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

    ELA_QUALITY = int(os.environ.get("ELA_QUALITY", 90))

    # Threshold on P(real) at or above which an image is reported as authentic.
    # Lower it to flag more images as doctored; raise it to be more lenient.
    CLASSIFICATION_THRESHOLD = float(os.environ.get("CLASSIFICATION_THRESHOLD", 0.45))

    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = _as_bool(os.environ.get("FLASK_DEBUG"), default=False)

    # --- reinforcement learning (contextual bandit decision policy) ----------
    # When enabled, each prediction picks a decision threshold (an "action")
    # via UCB1 for the model's confidence bucket ("state") and user feedback
    # on the verdict supplies the reward (+1 correct / -1 incorrect).
    RL_ENABLED = _as_bool(os.environ.get("RL_ENABLED"), default=True)
    RL_ACTIONS = tuple(
        float(x) for x in os.environ.get("RL_ACTIONS", "0.35,0.45,0.55,0.65").split(",") if x.strip()
    )
    RL_STATES = int(os.environ.get("RL_STATES", 10))
    RL_EXPLORATION = float(os.environ.get("RL_EXPLORATION", 2.0))
    POLICY_PATH = os.environ.get("POLICY_PATH", str(BASE_DIR / "bandit_policy.json"))


def get_config():
    return Config
