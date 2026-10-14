from django.contrib import admin

from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "filename", "user", "label", "confidence", "created_at")
    list_filter = ("label", "created_at")
    search_fields = ("filename", "user__username")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at",)
    list_select_related = ("user",)
