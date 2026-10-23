from django.contrib.auth.models import User
from rest_framework import serializers

from .models import EnhancementJob


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=6)
    confirm_password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ["username", "email", "password", "confirm_password"]

    def validate(self, attrs):
        if attrs["password"] != attrs["confirm_password"]:
            raise serializers.ValidationError({"confirm_password": "Passwords do not match."})
        if User.objects.filter(username=attrs["username"]).exists():
            raise serializers.ValidationError({"username": "Username is already taken."})
        return attrs

    def create(self, validated_data):
        validated_data.pop("confirm_password")
        return User.objects.create_user(**validated_data)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email"]


class EnhancementRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()


class EnhancementJobSerializer(serializers.ModelSerializer):
    input_image_url = serializers.SerializerMethodField()
    enhanced_image_url = serializers.SerializerMethodField()
    user = UserSerializer(read_only=True)

    class Meta:
        model = EnhancementJob
        fields = [
            "id",
            "user",
            "original_name",
            "status",
            "psnr",
            "ssim",
            "uiqm",
            "mu",
            "notes",
            "created_at",
            "input_image_url",
            "enhanced_image_url",
        ]

    def get_input_image_url(self, obj: EnhancementJob) -> str | None:
        request = self.context.get("request")
        if not obj.input_image:
            return None
        url = obj.input_image.url
        return request.build_absolute_uri(url) if request else url

    def get_enhanced_image_url(self, obj: EnhancementJob) -> str | None:
        request = self.context.get("request")
        if not obj.enhanced_image:
            return None
        url = obj.enhanced_image.url
        return request.build_absolute_uri(url) if request else url
