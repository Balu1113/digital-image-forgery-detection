from django.conf import settings
from django.db import models


class Prediction(models.Model):
    """A single forgery-analysis result produced by the inference service."""

    class Label(models.TextChoices):
        REAL = "Real", "Real"
        DOCTORED = "Doctored", "Doctored"

    class Feedback(models.TextChoices):
        PENDING = "pending", "Pending"
        CORRECT = "correct", "Correct"
        INCORRECT = "incorrect", "Incorrect"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name="predictions",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    filename = models.CharField(max_length=255, blank=True)
    original_image = models.ImageField(upload_to="originals/%Y/%m/%d/")
    ela_image = models.ImageField(upload_to="ela/%Y/%m/%d/")
    label = models.CharField(max_length=16, choices=Label.choices)
    confidence = models.FloatField(help_text="0-100 confidence in `label`.")
    forgery_probability = models.FloatField(help_text="1 - P(real), 0-1.")
    real_probability = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # RL decision the inference bandit made for this prediction (used to
    # attribute the reward when the user gives feedback).
    policy_state = models.PositiveSmallIntegerField(null=True, blank=True)
    policy_action = models.PositiveSmallIntegerField(null=True, blank=True)
    policy_threshold = models.FloatField(null=True, blank=True)

    # User verdict on the analysis: the reward signal for the bandit.
    feedback = models.CharField(
        max_length=10, choices=Feedback.choices, default=Feedback.PENDING, blank=True
    )
    feedback_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"])]

    def __str__(self):
        who = self.user.username if self.user else "anonymous"
        return f"#{self.pk} {who} - {self.label} ({self.confidence:.1f}%)"
