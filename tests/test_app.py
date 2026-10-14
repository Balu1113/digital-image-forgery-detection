import io

import pytest


def test_index_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["service"] == "forgery-detection-inference"
    assert "predict" in payload["endpoints"]


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "model_available" in payload


def test_predict_requires_file(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_rejects_bad_extension(client, sample_image):
    response = client.post(
        "/predict",
        data={"file": (sample_image, "notes.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.get_json()["error"]


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"not an image"), "fake.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_end_to_end(client, sample_image, model_available):
    if not model_available:
        pytest.skip("Model file not present; skipping inference test.")

    sample_image.seek(0)
    response = client.post(
        "/predict",
        data={"file": (sample_image, "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.data
    payload = response.get_json()
    assert payload["prediction"] in {"Real", "Doctored"}
    assert 0.0 <= payload["confidence"] <= 100.0
    assert 0.0 <= payload["forgery_probability"] <= 1.0
    assert payload["original_image"]
    assert payload["ela_image"]


# ---------------------------------------------------------------- RL / bandit


def test_index_lists_rl_endpoints(client):
    payload = client.get("/").get_json()
    assert "policy" in payload["endpoints"]
    assert "feedback" in payload["endpoints"]


def test_policy_endpoint(client):
    response = client.get("/policy")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["enabled"] is True
    assert payload["actions"] == [0.35, 0.45, 0.55, 0.65]
    assert len(payload["states"]) == 10
    assert payload["pulls"] >= 0


def test_predict_uses_bandit_decision(client, sample_image, monkeypatch):
    """The bandit's chosen threshold decides the label, and is reported."""
    from PIL import Image

    import model_utils

    def fake_predict(image, model_path=None, threshold=0.45, quality=90):
        return {
            "label": "Doctored",  # what the static 0.45 threshold would say
            "confidence": 60.0,
            "forgery_probability": 0.6,
            "real_probability": 0.4,
            "ela_image": Image.new("RGB", (32, 32), (10, 20, 30)),
        }

    monkeypatch.setattr(model_utils, "predict_forgery", fake_predict)
    sample_image.seek(0)
    response = client.post(
        "/predict",
        data={"file": (sample_image, "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()

    policy = payload["policy"]
    assert policy["state"] == 4  # P(real)=0.4 -> bucket 4 of 10
    assert policy["action"] == 0  # untried action selected first (UCB1)
    assert policy["threshold"] == 0.35
    # 0.4 >= 0.35 -> the bandit overrides the static 0.45 verdict.
    assert payload["prediction"] == "Real"


def test_feedback_requires_valid_body(client):
    assert client.post("/feedback", json={}).status_code == 400
    assert (
        client.post("/feedback", json={"state": "x", "action": 0, "reward": 1}).status_code
        == 400
    )
    assert (
        client.post("/feedback", json={"state": 0, "action": 0, "reward": 9}).status_code
        == 400
    )
    assert (
        client.post("/feedback", json={"state": 99, "action": 0, "reward": 1}).status_code
        == 400
    )


def test_feedback_learns_and_supports_corrections(client):
    # First reward: counts as a pull for (state 9, action 3).
    response = client.post("/feedback", json={"state": 9, "action": 3, "reward": 1})
    assert response.status_code == 200
    states = response.get_json()["policy"]["states"]
    assert states[9]["counts"][3] == 1
    assert states[9]["means"][3] == 1.0

    # A changed verdict corrects the mean without a second pull.
    response = client.post(
        "/feedback", json={"state": 9, "action": 3, "reward": -2, "count": False}
    )
    assert response.status_code == 200
    states = response.get_json()["policy"]["states"]
    assert states[9]["counts"][3] == 1
    assert states[9]["means"][3] == -1.0

    # The policy file was written so learning survives restarts.
    policy_path = app_policy_path(client)
    assert policy_path.exists()


def app_policy_path(client):
    from pathlib import Path

    return Path(client.application.config["POLICY_PATH"])
