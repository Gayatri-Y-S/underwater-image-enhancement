from django.urls import path

from .api_views import (
    CsrfCookieView,
    EnhancementJobDetailView,
    EnhancementJobListView,
    EnhanceImageView,
    HealthView,
    LoginView,
    LogoutView,
    ProfileView,
    ProjectSummaryView,
    RegisterView,
)


urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("auth/csrf/", CsrfCookieView.as_view(), name="csrf-cookie"),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", LoginView.as_view(), name="login"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("auth/profile/", ProfileView.as_view(), name="profile"),
    path("project-summary/", ProjectSummaryView.as_view(), name="project-summary"),
    path("enhance/", EnhanceImageView.as_view(), name="enhance"),
    path("jobs/", EnhancementJobListView.as_view(), name="job-list"),
    path("jobs/<int:pk>/", EnhancementJobDetailView.as_view(), name="job-detail"),
]
