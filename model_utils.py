"""ELA generation and forgery inference utilities.

The model used by this project is a fusion of InceptionV3 and VGG16 that was
trained on *Error Level Analysis* (ELA) images.  This module takes care of:

* turning any RGB image into an ELA image,
* resizing/normalising it for the network,
* lazily loading the Keras model (thread-safe), and
* running a prediction and turning the raw sigmoid output into a label.
"""

import io
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# TensorFlow is imported lazily to keep module import light where possible.
_MODEL = None
_MODEL_PATH = None
_MODEL_LOCK = threading.Lock()

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "image_real_or_fake.keras")
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_ELA_QUALITY = 90

# Threshold on P(real): at or above this the image is reported as authentic.
# Calibrated on authentic photos vs. manipulated images; lower it to flag more
# images as doctored, raise it to be more lenient toward authentic photos.
DEFAULT_THRESHOLD = 0.45


def load_model(model_path=None, force_reload=False):
    """Load and cache the Keras model.

    Loading is guarded by a lock so concurrent requests in the first burst do
    not each build their own copy of a 180 MB network.
    """
    global _MODEL, _MODEL_PATH

    model_path = str(model_path or DEFAULT_MODEL_PATH)

    if _MODEL is not None and not force_reload and _MODEL_PATH == model_path:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is None or force_reload or _MODEL_PATH != model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found at '{model_path}'. "
                    "Set the MODEL_PATH environment variable to point to your .keras model."
                )
            from tensorflow.keras import models

            _MODEL = models.load_model(model_path)
            _MODEL_PATH = model_path
        return _MODEL


def is_model_available(model_path=None):
    return os.path.exists(str(model_path or DEFAULT_MODEL_PATH))


def is_model_loaded(model_path=None):
    if _MODEL is None:
        return False
    if model_path is None:
        return True
    return os.path.abspath(str(model_path)) == os.path.abspath(str(_MODEL_PATH))


def open_image(image):
    """Coerce ``image`` (path, file-like or PIL image) into an RGB PIL image."""
    if isinstance(image, Image.Image):
        pil = image
    else:
        if hasattr(image, "seek"):
            try:
                image.seek(0)
            except Exception:
                pass
        pil = Image.open(image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil


def image_to_ela(image, quality=DEFAULT_ELA_QUALITY):
    """Return a new PIL image containing the ELA of ``image``.

    ``image`` may be a path, a file-like object or an opened PIL image.
    The original pixels are never mutated.
    """
    image = open_image(image)

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(quality))
    buffer.seek(0)
    resaved = Image.open(buffer)

    ela_image = ImageChops.difference(image, resaved)

    extrema = ela_image.getextrema()
    max_value = max(band[1] for band in extrema)
    if max_value == 0:
        max_value = 1

    scale = 255.0 / max_value
    return ImageEnhance.Brightness(ela_image).enhance(scale)


def get_model_input_size(model=None, default=DEFAULT_IMAGE_SIZE):
    """Infer the (width, height) the model expects from its input shape."""
    model = model or load_model()
    shape = model.input_shape
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if not shape or len(shape) < 3:
        return default
    height, width = shape[1], shape[2]
    if height is None or width is None:
        return default
    return (int(width), int(height))


def preprocess_image(
    image,
    target_size=DEFAULT_IMAGE_SIZE,
    quality=DEFAULT_ELA_QUALITY,
    normalize=False,
):
    """Convert ``image`` to an ELA array of shape (1, H, W, 3).

    ``normalize`` must match how the model was trained.  The shipped model was
    trained with an unscaled ``ImageDataGenerator`` (0-255 pixel range), so the
    default is *no* normalisation.  Set it to True only for models trained on
    0-1 inputs.
    """
    ela_image = image_to_ela(image, quality=quality)
    ela_image = ela_image.resize(target_size, Image.LANCZOS)
    array = np.asarray(ela_image, dtype="float32")
    if normalize:
        array = array / 255.0
    return np.expand_dims(array, axis=0), ela_image


def predict_forgery(
    image,
    model_path=None,
    threshold=DEFAULT_THRESHOLD,
    quality=DEFAULT_ELA_QUALITY,
):
    """Run forgery detection on ``image`` (path, file-like or PIL image).

    The network was trained with ``flow_from_directory``, which labels the
    alphabetically-last class as 1.  For the ``Real``/``Doctored`` folders that
    means the sigmoid output is **P(real)**, so a high value means authentic.

    Returns a dictionary::

        {
            "label": "Real" | "Doctored",
            "confidence": float,            # 0-100, confidence in ``label``
            "forgery_probability": float,   # 1 - P(real)
            "real_probability": float,      # raw sigmoid output
            "ela_image": PIL.Image,         # the ELA visualisation
        }
    """
    model = load_model(model_path)
    target_size = get_model_input_size(model)

    pil_image = open_image(image)
    array, ela_image = preprocess_image(pil_image, target_size=target_size, quality=quality)

    real_probability = float(np.ravel(model.predict(array, verbose=0))[0])
    forgery_probability = 1.0 - real_probability

    if real_probability >= threshold:
        label = "Real"
        confidence = real_probability * 100.0
    else:
        label = "Doctored"
        confidence = forgery_probability * 100.0

    return {
        "label": label,
        "confidence": confidence,
        "forgery_probability": forgery_probability,
        "real_probability": real_probability,
        "ela_image": ela_image,
    }


def pil_to_base64(image, fmt="JPEG", quality=90):
    """Encode a PIL image as a base64 string (no data URI prefix)."""
    import base64

    buffer = io.BytesIO()
    if fmt.upper() == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, fmt.upper(), quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
