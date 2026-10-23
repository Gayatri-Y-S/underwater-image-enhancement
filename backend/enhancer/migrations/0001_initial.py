from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="EnhancementJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("original_name", models.CharField(max_length=255)),
                ("input_image", models.ImageField(upload_to="inputs/")),
                ("enhanced_image", models.ImageField(blank=True, null=True, upload_to="outputs/")),
                ("status", models.CharField(choices=[("queued", "Queued"), ("completed", "Completed"), ("failed", "Failed")], default="queued", max_length=32)),
                ("psnr", models.FloatField(blank=True, null=True)),
                ("ssim", models.FloatField(blank=True, null=True)),
                ("uiqm", models.FloatField(blank=True, null=True)),
                ("mu", models.FloatField(blank=True, null=True)),
                ("notes", models.TextField(blank=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
