from django.contrib import admin

from .models import EnhancementJob


@admin.register(EnhancementJob)
class EnhancementJobAdmin(admin.ModelAdmin):
    list_display = ("id", "original_name", "status", "psnr", "ssim", "uiqm", "created_at")
    search_fields = ("original_name", "status")
