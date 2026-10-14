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


def _jpeg_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 10, 10)).save(buffer, "JPEG")
    return buffer.getvalue()


def _stored_prediction(user, **overrides):
    from .models import Prediction

    defaults = dict(
        user=user,
        filename="stored.jpg",
        label="Real",
        confidence=80.0,
        forgery_probability=0.2,
        real_probability=0.8,
        original_image=SimpleUploadedFile("o.jpg", _jpeg_bytes(), "image/jpeg"),
        ela_image=SimpleUploadedFile("e.jpg", _jpeg_bytes(), "image/jpeg"),
        policy_state=4,
        policy_action=1,
        policy_threshold=0.45,
    )
    defaults.update(overrides)
    return Prediction.objects.create(**defaults)


class FeedbackTests(APITestCase):
    def setUp(self):
        self.owner = User.objects.create_user(username="fb_user", password="supersecret123")
        self.prediction = _stored_prediction(self.owner)
        self.url = f"/api/predictions/{self.prediction.pk}/feedback/"

    def test_feedback_requires_auth(self):
        response = self.client.post(self.url, {"verdict": "correct"})
        self.assertEqual(response.status_code, 401)

    def test_feedback_on_someone_elses_prediction_is_404(self):
        other = User.objects.create_user(username="intruder", password="supersecret123")
        self.client.force_authenticate(user=other)
        response = self.client.post(self.url, {"verdict": "correct"})
        self.assertEqual(response.status_code, 404)

    def test_invalid_verdict_rejected(self):
        self.client.force_authenticate(user=self.owner)
        response = self.client.post(self.url, {"verdict": "maybe"})
        self.assertEqual(response.status_code, 400)
        self.prediction.refresh_from_db()
        self.assertEqual(self.prediction.feedback, "pending")

    @patch("predictions.views.record_feedback", return_value=True)
    def test_correct_feedback_relays_reward(self, relay):
        self.client.force_authenticate(user=self.owner)
        response = self.client.post(self.url, {"verdict": "correct"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["feedback"], "correct")
        self.assertTrue(payload["synced"])

        # First feedback = one fresh pull: (state, action, reward, count).
        relay.assert_called_once_with(4, 1, 1.0, count=True)

        self.prediction.refresh_from_db()
        self.assertEqual(self.prediction.feedback, "correct")
        self.assertIsNotNone(self.prediction.feedback_at)

    @patch("predictions.views.record_feedback", return_value=True)
    def test_changing_feedback_sends_delta_without_new_pull(self, relay):
        self.client.force_authenticate(user=self.owner)
        self.client.post(self.url, {"verdict": "correct"})
        self.client.post(self.url, {"verdict": "incorrect"})

        self.assertEqual(relay.call_count, 2)
        args, kwargs = relay.call_args
        self.assertEqual(args, (4, 1, -2.0))
        self.assertFalse(kwargs["count"])

        self.prediction.refresh_from_db()
        self.assertEqual(self.prediction.feedback, "incorrect")

    @patch("predictions.views.record_feedback", return_value=False)
    def test_feedback_stored_when_service_unreachable(self, relay):
        self.client.force_authenticate(user=self.owner)
        response = self.client.post(self.url, {"verdict": "incorrect"})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["synced"])
        self.prediction.refresh_from_db()
        self.assertEqual(self.prediction.feedback, "incorrect")

    @patch("predictions.views.record_feedback")
    def test_prediction_without_policy_skips_relay(self, relay):
        legacy = _stored_prediction(
            self.owner, policy_state=None, policy_action=None, policy_threshold=None
        )
        self.client.force_authenticate(user=self.owner)
        response = self.client.post(
            f"/api/predictions/{legacy.pk}/feedback/", {"verdict": "correct"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["synced"])
        relay.assert_not_called()
        legacy.refresh_from_db()
        self.assertEqual(legacy.feedback, "correct")


class PolicyStorageTests(APITestCase):
    @patch(
        "predictions.views.analyze_image",
        return_value={
            **FAKE_RESULT,
            "policy": {"state": 7, "action": 2, "threshold": 0.55},
        },
    )
    def test_create_stores_bandit_decision(self, mocked):
        user = User.objects.create_user(username="policy_user", password="supersecret123")
        self.client.force_authenticate(user=user)

        created = self.client.post("/api/predictions/", {"image": _upload()}, format="multipart")
        self.assertEqual(created.status_code, 201)

        payload = created.json()
        self.assertEqual(payload["policy_state"], 7)
        self.assertEqual(payload["policy_action"], 2)
        self.assertEqual(payload["policy_threshold"], 0.55)
        self.assertEqual(payload["feedback"], "pending")

        listed = self.client.get("/api/predictions/").json()[0]
        self.assertEqual(listed["policy_state"], 7)
        self.assertEqual(listed["feedback"], "pending")
