from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.http import Http404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import EnhancementJob
from .pipeline import BalancedAdaptionCompensationPipeline
from .serializers import EnhancementJobSerializer, EnhancementRequestSerializer


PIPELINE = BalancedAdaptionCompensationPipeline(weights_dir=settings.MODEL_WEIGHTS_DIR)


class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


class ProjectSummaryView(APIView):
    def get(self, request):
        return Response(
            {
                "title": "Underwater Image Enhancement with Balanced Adaption Compensation",
                "training_plan": {
                    "train_dataset": "EUVP",
                    "train_split": "80% train / 20% validation",
                    "test_dataset": "UIEB",
                    "fusion_formula": "output = μ·FST + (1−μ)·FSR",
                    "metrics": ["PSNR", "SSIM", "UIQM"],
                },
                "modules": [
                    "ResUNet + CBAM for semantic feature transfer (FST)",
                    "Physical model for scene-relevant reconstruction (FSR)",
                    "MuNet for adaptive balance factor μ",
                ],
            }
        )


class EnhanceImageView(APIView):
    def post(self, request):
        serializer = EnhancementRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_image = serializer.validated_data["image"]
        job = EnhancementJob.objects.create(
            original_name=uploaded_image.name,
            input_image=uploaded_image,
            status="queued",
        )

        try:
            input_array = self._read_image(job.input_image.path)
            result = PIPELINE.enhance(input_array)
            output_name = f"{Path(uploaded_image.name).stem}_{uuid4().hex[:8]}.png"
            success, buffer = cv2.imencode(".png", cv2.cvtColor(result.fused_image, cv2.COLOR_RGB2BGR))
            if not success:
                raise ValueError("Failed to encode enhanced image")

            job.enhanced_image.save(output_name, ContentFile(buffer.tobytes()), save=False)
            job.psnr = result.psnr
            job.ssim = result.ssim
            job.uiqm = result.uiqm
            job.mu = result.mu
            job.notes = result.notes
            job.status = "completed"
            job.save()
        except Exception as exc:
            job.status = "failed"
            job.notes = str(exc)
            job.save(update_fields=["status", "notes", "updated_at"])
            return Response(
                {
                    "message": "Enhancement failed.",
                    "error": str(exc),
                    "job": EnhancementJobSerializer(job, context={"request": request}).data,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Enhancement completed successfully.",
                "job": EnhancementJobSerializer(job, context={"request": request}).data,
            },
            status=status.HTTP_201_CREATED,
        )

    def _read_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path)
        if image is None:
            raise ValueError("Uploaded image could not be decoded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


class EnhancementJobDetailView(APIView):
    def get(self, request, pk: int):
        try:
            job = EnhancementJob.objects.get(pk=pk)
        except EnhancementJob.DoesNotExist as exc:
            raise Http404 from exc
        return Response(EnhancementJobSerializer(job, context={"request": request}).data)
