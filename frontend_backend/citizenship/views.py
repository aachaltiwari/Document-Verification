from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def verify(request):
    return render(request, 'verify.html')

import os
import time
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .forms import CitizenshipForm, LoginForm, UserRegistrationForm, CitizenshipDetailForm
from .models import User, CitizenshipDetail, VerificationScore


# Function to handle file upload
def handle_uploaded_file(file, user, filename_prefix):
   """
   Handles the file upload and saving, including validation, error handling, and file naming.
   """
   # Build the file path
   timestamp = int(time.time())
   filename = f"{filename_prefix}_{user.id}_{timestamp}.jpg"

   # Build the file path
   filepath = os.path.join('citizenship_documents', str(user.id), filename)

   # Get the FileSystemStorage
   fs = FileSystemStorage(location=settings.MEDIA_ROOT)

   # Save the file
   filename = fs.save(filepath, file)
   return filename


# Define the validate_image function
def validate_image(image):
    if not image.name.endswith(('.png', '.jpg', '.jpeg')):
        raise ValidationError('Invalid file type. Only .png, .jpg, and .jpeg are allowed.')
    if image.size > 5 * 1024 * 1024:
        raise ValidationError('File size exceeds the limit of 5MB.')

# Citizenship Details View
@login_required
def citizenship_details(request):
    """Handle submission of citizenship details, including images."""
    citizenship_detail, _ = CitizenshipDetail.objects.get_or_create(user=request.user)
    form = CitizenshipDetailForm(instance=citizenship_detail)  # Initialize form here
    if request.method == 'POST':
        form = CitizenshipDetailForm(request.POST, request.FILES, instance=citizenship_detail)
        if form.is_valid():
            try:
                # Save Files
                front_photo = request.FILES.get('front')
                back_photo = request.FILES.get('back')

                front_filename = None  # Initialize front_filename
                back_filename = None  # Initialize back_filename

                if front_photo:
                    validate_image(front_photo)  # Validate front image
                    front_filename = handle_uploaded_file(front_photo, request.user, "front")

                if back_photo:
                    validate_image(back_photo)  # Validate back image
                    back_filename = handle_uploaded_file(back_photo, request.user, "back")

                # Update database even if some files are missing
                if front_filename:
                    citizenship_detail.front_image = front_filename
                if back_filename:
                    citizenship_detail.back_image = back_filename

                citizenship_detail.save() #saves filename in model
                messages.success(request, 'Citizenship details submitted successfully')
                return redirect('users:dashboard')

            except ValidationError as e:
                messages.error(request, f'Validation failed: {str(e)}')
                return render(request, 'users/citizenship_details.html', {'form': form})
        else:
            messages.error(request, 'Please correct the errors below.')
            return render(request, 'users/citizenship_details.html', {'form': form})
    return render(request, 'users/citizenship_details.html', {'form': form})