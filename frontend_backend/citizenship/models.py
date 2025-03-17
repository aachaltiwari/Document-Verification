from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
from django.contrib.auth.models import User


class ValidationResult(models.Model):
    """Model to store validation results."""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='validation_results',
    )
    blur_score = models.FloatField()
    blur_status = models.CharField(max_length=10, default="Unknown")
    brightness_score = models.FloatField()
    brightness_status = models.CharField(max_length=10, default="Unknown")
    distance_score = models.FloatField()
    distance_status = models.CharField(max_length=10, default="Unknown")
    liveness_status = models.CharField(max_length=10, default="Unknown")
    trust_score = models.FloatField()
    overall_status = models.CharField(max_length=10, default="Unknown")
    created_at = models.DateTimeField(auto_now_add=True)
    face_recognition_passed = models.BooleanField(default=False)
    similarity_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Validation Result {self.id} for {self.user.username}"

def upload_to(instance, filename):
  return 'citizenship_documents/{}/{}.jpg'.format(instance.user.id, filename)


class CitizenshipDetail(models.Model):  # RESTORE THE MODEL
    """Model to store citizenship details."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='citizenship_details')
    # Add other fields here
    full_name = models.CharField(max_length=255, default="Unknown")
    place_of_birth = models.CharField(max_length=255, default="Unknown Place")
    district_of_birth = models.CharField(max_length=255, blank=True)  # Required
    citizenship_number = models.CharField(max_length=255, blank=True)  # Required

    # Address Information
    permanent_address_district = models.CharField(max_length=255, default="Unknown District")
    municipality = models.CharField(max_length=255, default="Unknown Municipality")
    ward_number = models.CharField(max_length=10, default="Unknown Ward")
    date_of_birth = models.DateField(blank=True, null=True)

    # Parent Information
    father_name = models.CharField(max_length=255, blank=True)
    father_address = models.CharField(max_length=255, blank=True)
    mother_name = models.CharField(max_length=255, blank=True)
    mother_address = models.CharField(max_length=255, blank=True)

    # Spouse Information (Optional)
    spouse_name = models.CharField(max_length=255, null=True, blank=True)  # Spouse's Full Name
    spouse_address = models.CharField(max_length=255, null=True, blank=True)  # Spouse's Address

    # Photos for Verification
    front_image = models.ImageField(upload_to=upload_to, blank=True, null=True)
    back_image = models.ImageField(upload_to=upload_to, blank=True, null=True)

    def __str__(self):
        return f"Citizenship details for {self.user.username}"


class VerificationScore(models.Model):
    """Model to store the verification score for users."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='verification_score')
    total_score = models.IntegerField(default=0)

    def __str__(self):
        return f"Verification score for {self.user.username}: {self.total_score}"