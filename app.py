"""Flask inference microservice for digital image forgery detection.

This service owns the TensorFlow model and exposes a small JSON API around
:mod:`model_utils`.  It is intended to be called by the Django API gateway
(see ``backend/``), not directly by browsers, although the endpoints are public.

Endpoints
---------
``GET  /``               Service metadata
``GET  /health``         Liveness + model status
``POST /predict``        Analyse an image (multipart field ``file``)
``POST /api/predict``    Alias of ``/predict``
``GET  /policy``         Current bandit (RL decision policy) statistics
``POST /feedback``       Reward signal for a past decision (state, action, reward)
"""

import logging
import os

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import model_utils
from bandit import ContextualBandit
from config import get_config

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("forgery-detect")


def _allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def create_app(config_object=None):
    app = Flask(__name__)
    app.config.from_object(config_object or get_config())

    model_path = app.config["MODEL_PATH"]
    threshold = app.config["CLASSIFICATION_THRESHOLD"]
    quality = app.config["ELA_QUALITY"]
    allowed = app.config["ALLOWED_EXTENSIONS"]

    rl_enabled = app.config["RL_ENABLED"]
    policy = ContextualBandit(
        num_states=app.config["RL_STATES"],
        actions=app.config["RL_ACTIONS"],
        exploration=app.config["RL_EXPLORATION"],
        path=app.config["POLICY_PATH"] if rl_enabled else None,
    )
    app.config["BANDIT"] = policy

    @app.route("/")
    def index():
        return jsonify(
            {
                "service": "forgery-detection-inference",
                "status": "ok",
                "endpoints": {
                    "health": "/health",
                    "predict": "POST /predict",
                    "policy": "GET /policy",
                    "feedback": "POST /feedback",
                },
                "model_available": model_utils.is_model_available(model_path),
            }
        )

    @app.route("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_path": model_path,
                "model_available": model_utils.is_model_available(model_path),
                "model_loaded": model_utils.is_model_loaded(model_path),
                "threshold": threshold,
            }
        )

    @app.route("/predict", methods=["POST"])
    @app.route("/api/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Send it as form field 'file'."}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not _allowed_file(file.filename, allowed):
            pretty = ", ".join(sorted(allowed))
            return jsonify({"error": f"Invalid file type. Supported: {pretty}."}), 400

        try:
            file.stream.seek(0)
            with Image.open(file.stream) as probe:
                probe.verify()
            file.stream.seek(0)

            result = model_utils.predict_forgery(
                file.stream,
                model_path=model_path,
                threshold=threshold,
                quality=quality,
            )

            # RL: the bandit picks the decision threshold ("action") for this
            # prediction's confidence bucket ("state"); the user's verdict
            # feedback later supplies the reward.
            p_real = result["real_probability"]
            label = result["label"]
            confidence = result["confidence"]
            decision = None
            if rl_enabled:
                state = policy.state_for(p_real)
                action = policy.select(state)
                used_threshold = policy.actions[action]
                if p_real >= used_threshold:
                    label, confidence = "Real", p_real * 100.0
                else:
                    label, confidence = "Doctored", (1.0 - p_real) * 100.0
                decision = {"state": state, "action": action, "threshold": used_threshold}
        except UnidentifiedImageError:
            return jsonify({"error": "Uploaded file is not a valid image."}), 400
        except (ValueError, OSError) as exc:
            logger.warning("Bad image input: %s", exc)
            return jsonify({"error": f"Could not process image: {exc}"}), 400
        except FileNotFoundError as exc:
            logger.error("Model unavailable: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except Exception:  # pragma: no cover - defensive
            logger.exception("Unexpected error during prediction")
            return jsonify({"error": "Internal error while processing the image."}), 500

        file.stream.seek(0)
        original_b64 = model_utils.pil_to_base64(Image.open(file.stream).convert("RGB"))
        ela_b64 = model_utils.pil_to_base64(result["ela_image"])

        return jsonify(
            {
                "prediction": label,
                "confidence": round(confidence, 2),
                "forgery_probability": round(result["forgery_probability"], 4),
                "real_probability": round(result["real_probability"], 4),
                "filename": secure_filename(file.filename),
                "original_image": original_b64,
                "ela_image": ela_b64,
                "policy": decision,
            }
        )

    @app.route("/policy")
    def policy_stats():
        return jsonify({"enabled": rl_enabled, **policy.summary()})

    @app.route("/feedback", methods=["POST"])
    def feedback():
        """Learn from a user's verdict on a past prediction."""
        if not rl_enabled:
            return jsonify({"error": "Reinforcement learning is disabled (RL_ENABLED=false)."}), 400

        data = request.get_json(silent=True) or {}
        try:
            state = int(data["state"])
            action = int(data["action"])
            reward = float(data["reward"])
        except (KeyError, TypeError, ValueError):
            return (
                jsonify({"error": "Body must include numeric 'state', 'action' and 'reward'."}),
                400,
            )

        count = data.get("count", True)
        if isinstance(count, str):
            count = count.strip().lower() in {"1", "true", "yes", "on"}

        try:
            policy.observe(state, action, reward, count=count)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"status": "ok", "policy": policy.summary()})

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_error):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        return jsonify({"error": f"File too large. Maximum size is {limit_mb}MB."}), 413

    @app.errorhandler(404)
    def handle_not_found(_error):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(500)
    def handle_server_error(_error):  # pragma: no cover - defensive
        return jsonify({"error": "Internal server error"}), 500

    logger.info("Model configured at %s (available=%s)", model_path, model_utils.is_model_available(model_path))
    return app


app = create_app()


if __name__ == "__main__":
    cfg = get_config()
    app.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
