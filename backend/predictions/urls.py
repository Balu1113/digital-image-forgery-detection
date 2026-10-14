from django.urls import path

from .views import (
    FeedbackView,
    HealthView,
    LoginView,
    LogoutView,
    MeView,
    PredictionDetailView,
    PredictionListCreateView,
    RegisterView,
)

urlpatterns = [
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", LoginView.as_view(), name="login"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("auth/me/", MeView.as_view(), name="me"),
    path("health/", HealthView.as_view(), name="health"),
    path("predictions/", PredictionListCreateView.as_view(), name="prediction-list"),
    path("predictions/<int:pk>/", PredictionDetailView.as_view(), name="prediction-detail"),
    path(
        "predictions/<int:pk>/feedback/",
        FeedbackView.as_view(),
        name="prediction-feedback",
    ),
]
