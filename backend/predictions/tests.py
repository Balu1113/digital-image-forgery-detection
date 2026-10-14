import base64
import io
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from rest_framework.test import APITestCase


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 110, 130)).save(buffer, "PNG")
    return buffer.getvalue()


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (180, 60, 60)).save(buffer, "JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def _upload():
    return SimpleUploadedFile("sample.png", _png_bytes(), content_type="image/png")


FAKE_RESULT = {
    "prediction": "Real",
    "confidence": 88.0,
    "forgery_probability": 0.12,
    "real_probability": 0.88,
    "original_image": _jpeg_b64(),
    "ela_image": _jpeg_b64(),
}


class AuthTests(APITestCase):
    def test_register_returns_token(self):
        response = self.client.post(
            "/api/auth/register/",
            {"username": "alice", "password": "supersecret123", "email": "a@example.com"},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("token", response.json())
        self.assertTrue(User.objects.filter(username="alice").exists())

    def test_login_and_me(self):
        User.objects.create_user(username="bob", password="supersecret123")
        login = self.client.post(
            "/api/auth/login/", {"username": "bob", "password": "supersecret123"}, format="json"
        )
        self.assertEqual(login.status_code, 200)
        token = login.json()["token"]

        me = self.client.get("/api/auth/me/", HTTP_AUTHORIZATION=f"Token {token}")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["username"], "bob")


class PredictionTests(APITestCase):
    def test_predict_requires_auth(self):
        response = self.client.post("/api/predictions/", {"image": _upload()}, format="multipart")
        self.assertEqual(response.status_code, 401)

    def test_history_requires_auth(self):
        response = self.client.get("/api/predictions/")
        self.assertEqual(response.status_code, 401)

    @patch("predictions.views.analyze_image", return_value=FAKE_RESULT)
    def test_authenticated_history_and_delete(self, mocked):
        user = User.objects.create_user(username="carol", password="supersecret123")
        self.client.force_authenticate(user=user)

        created = self.client.post("/api/predictions/", {"image": _upload()}, format="multipart")
        self.assertEqual(created.status_code, 201)
        pred_id = created.json()["id"]

        history = self.client.get("/api/predictions/")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 1)

        deleted = self.client.delete(f"/api/predictions/{pred_id}/")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(len(self.client.get("/api/predictions/").json()), 0)

    def test_rejects_invalid_extension(self):
        user = User.objects.create_user(username="dave", password="supersecret123")
        self.client.force_authenticate(user=user)
        bad = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
        response = self.client.post("/api/predictions/", {"image": bad}, format="multipart")
        self.assertEqual(response.status_code, 400)


class HealthTests(APITestCase):
    @patch("predictions.views.check_health", return_value={"reachable": True})
    def test_health(self, mocked):
        response = self.client.get("/api/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["django"], "ok")
        self.assertTrue(response.json()["inference"]["reachable"])
