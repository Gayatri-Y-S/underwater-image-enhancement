from django.conf import settings
from django.db import models


class EnhancementJob(models.Model):
    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="enhancement_jobs")
    original_name = models.CharField(max_length=255)
    input_image = models.ImageField(upload_to="inputs/")
    enhanced_image = models.ImageField(upload_to="outputs/", blank=True, null=True)
    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default="queued")
    psnr = models.FloatField(blank=True, null=True)
    ssim = models.FloatField(blank=True, null=True)
    uiqm = models.FloatField(blank=True, null=True)
    mu = models.FloatField(blank=True, null=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.original_name} ({self.status})"
