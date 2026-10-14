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


def get_config():
    return Config
