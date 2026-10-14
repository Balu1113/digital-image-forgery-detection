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
