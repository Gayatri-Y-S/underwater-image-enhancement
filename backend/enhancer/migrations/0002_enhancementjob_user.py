from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


def assign_existing_jobs(apps, schema_editor):
    EnhancementJob = apps.get_model("enhancer", "EnhancementJob")
    User = apps.get_model("auth", "User")
    fallback_user, _ = User.objects.get_or_create(
        username="legacy_user",
        defaults={"email": "legacy@example.com"},
    )
    EnhancementJob.objects.filter(user__isnull=True).update(user=fallback_user)


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("enhancer", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="enhancementjob",
            name="user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="enhancement_jobs",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.RunPython(assign_existing_jobs, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="enhancementjob",
            name="user",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="enhancement_jobs",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
