"""REST API views: auth, inference gateway and prediction history."""

import base64
import io
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import get_object_or_404
from django.utils import timezone
from PIL import Image, UnidentifiedImageError
from rest_framework import generics, permissions, status
from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Prediction
from .serializers import PredictionSerializer, UserPublicSerializer, UserSerializer
from .services import InferenceServiceError, analyze_image, check_health, record_feedback

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _to_jpeg_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=92)
    return buffer.getvalue()


class RegisterView(generics.CreateAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return Response(
            {"token": token.key, "user": UserPublicSerializer(user).data},
            status=status.HTTP_201_CREATED,
        )


class LoginView(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "user": UserPublicSerializer(user).data})


class LogoutView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        Token.objects.filter(user=request.user).delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(UserPublicSerializer(request.user).data)


class PredictionListCreateView(generics.ListCreateAPIView):
    serializer_class = PredictionSerializer
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)

    def create(self, request, *args, **kwargs):
        upload = request.FILES.get("image") or request.FILES.get("file")
        if upload is None:
            return Response({"error": "No image uploaded. Send it as form field 'image'."}, status=400)

        extension = Path(upload.name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            return Response({"error": f"Invalid file type. Supported: {allowed}."}, status=400)

        raw = upload.read()
        if not raw:
            return Response({"error": "Uploaded file is empty."}, status=400)
        if len(raw) > settings.MAX_UPLOAD_SIZE:
            limit_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
            return Response({"error": f"File too large. Maximum size is {limit_mb}MB."}, status=413)

        try:
            Image.open(io.BytesIO(raw)).verify()
        except (UnidentifiedImageError, OSError):
            return Response({"error": "Uploaded file is not a valid image."}, status=400)

        try:
            result = analyze_image(raw, upload.name)
        except InferenceServiceError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        try:
            ela_bytes = base64.b64decode(result["ela_image"])
        except (KeyError, ValueError):
            return Response({"error": "Inference service returned malformed data."}, status=502)

        # The bandit's decision (state/action/threshold) travels with the
        # result so the reward can be attributed when feedback arrives.
        policy = result.get("policy") or {}
        prediction = Prediction(
            user=request.user,
            filename=os.path.basename(upload.name)[:255],
            label=result["prediction"],
            confidence=result["confidence"],
            forgery_probability=result["forgery_probability"],
            real_probability=result.get("real_probability"),
            policy_state=policy.get("state"),
            policy_action=policy.get("action"),
            policy_threshold=policy.get("threshold"),
        )
        stem = uuid.uuid4().hex
        prediction.original_image.save(f"{stem}.jpg", ContentFile(_to_jpeg_bytes(raw)), save=False)
        prediction.ela_image.save(f"{stem}_ela.jpg", ContentFile(ela_bytes), save=False)
        prediction.save()

        serializer = self.get_serializer(prediction)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class PredictionDetailView(generics.RetrieveDestroyAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)


class FeedbackView(APIView):
    """Record the user's verdict and relay the reward to the bandit."""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, pk):
        prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
        verdict = str(request.data.get("verdict", "")).strip().lower()
        if verdict not in {Prediction.Feedback.CORRECT, Prediction.Feedback.INCORRECT}:
            return Response(
                {"error": "verdict must be 'correct' or 'incorrect'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        reward = 1.0 if verdict == Prediction.Feedback.CORRECT else -1.0
        previous = prediction.feedback
        previous_reward = {
            Prediction.Feedback.PENDING: 0.0,
            Prediction.Feedback.CORRECT: 1.0,
            Prediction.Feedback.INCORRECT: -1.0,
        }[previous]

        prediction.feedback = verdict
        prediction.feedback_at = timezone.now()
        prediction.save(update_fields=["feedback", "feedback_at"])

        synced = False
        delta = reward - previous_reward
        if (
            delta != 0
            and prediction.policy_state is not None
            and prediction.policy_action is not None
        ):
            # First feedback counts as a fresh pull; a changed verdict only
            # corrects the accumulated reward (count=False).
            synced = record_feedback(
                prediction.policy_state,
                prediction.policy_action,
                delta,
                count=previous == Prediction.Feedback.PENDING,
            )

        return Response({"feedback": prediction.feedback, "synced": synced})


class HealthView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response({"django": "ok", "inference": check_health()})
