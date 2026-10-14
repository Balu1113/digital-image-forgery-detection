from django.contrib.auth.models import User
from rest_framework import serializers

from .models import Prediction


class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ("id", "username", "email", "password")

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email", ""),
            password=validated_data["password"],
        )


class UserPublicSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email")


class PredictionSerializer(serializers.ModelSerializer):
    original_image_url = serializers.SerializerMethodField()
    ela_image_url = serializers.SerializerMethodField()

    class Meta:
        model = Prediction
        fields = (
            "id",
            "filename",
            "label",
            "confidence",
            "forgery_probability",
            "real_probability",
            "created_at",
            "original_image_url",
            "ela_image_url",
        )

    def _absolute(self, field):
        if not field:
            return None
        request = self.context.get("request")
        url = field.url
        return request.build_absolute_uri(url) if request else url

    def get_original_image_url(self, obj):
        return self._absolute(obj.original_image)

    def get_ela_image_url(self, obj):
        return self._absolute(obj.ela_image)
