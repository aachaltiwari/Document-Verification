import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dlib
import logging
import os
import json
from django.conf import settings
from django.utils import timezone
from scipy.interpolate import interp1d

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Dlib for face and landmark detection
detector = dlib.get_frontal_face_detector()
predictor_path = "/Users/arpanneupane75/Downloads/test - Copy/test/static/models/shape_predictor_68_face_landmarks.dat"

# Load the shape predictor within a try block
try:
    predictor = dlib.shape_predictor(predictor_path)
    print("Dlib shape predictor loaded successfully.")
except Exception as e:
    print(f"Error loading predictor: {e}")
    predictor = None  # Ensure predictor is None if loading fails


def decode_image(image_file):
    """Decode image from file-like object."""
    try:
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image.")
        return image
    except Exception as e:
        logging.error(f"Image decoding failed: {str(e)}")
        raise ValueError(f"Image decoding failed: {str(e)}")


def check_blur(image):
    """Detect if an image is blurry using Laplacian variance."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 10  # Use the specified threshold
        return {
            "status": not is_blurry,
            "message": "Image is clear." if not is_blurry else "Image is too blurry.",
            "laplacian_variance": laplacian_var
        }
    except Exception as e:
        logging.error(f"Blur check failed: {e}")
        return {"status": False, "message": f"Blur check failed: {e}"}


def check_brightness(image):
    """Check if the image has adequate brightness."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        is_bright = 40 <= brightness <= 230  # Use the specified range
        return {
            "status": is_bright,
            "message": "Brightness is adequate." if is_bright else "Image is too dark or too bright.",
            "brightness": brightness
        }
    except Exception as e:
        logging.error(f"Brightness check failed: {e}")
        return {"status": False, "message": f"Brightness check failed: {e}"}


def estimate_distance(image):
    """Estimate the distance of the face from the camera using face width measurement."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if not faces:
            return {"status": False, "message": "No face detected."}

        if len(faces) > 1:
            return {"status": False, "message": "Multiple faces detected."}

        face = faces[0]
        face_width = face.right() - face.left()

        # Pre-defined calibration data
        calibration_data = {120: 25, 90: 35, 60: 50, 30: 80}

        # Interpolate estimated distance
        face_widths = np.array(list(calibration_data.keys()))
        distances = np.array(list(calibration_data.values()))

        # Interpolation function
        f = interp1d(face_widths, distances, fill_value="extrapolate")
        estimated_distance = float(f(face_width))

        # Ensure the distance is within a reasonable range
        if 15 <= estimated_distance <= 130:  # Use the specified range
            return {"status": True, "message": f"Face is at {estimated_distance:.2f} cm.", "distance": estimated_distance}
        else:
            return {"status": False, "message": "Face is too far or too close.", "distance": estimated_distance}
    except Exception as e:
        logging.error(f"Distance estimation failed: {e}")
        return {"status": False, "message": f"Distance estimation failed: {e}"}


def calculate_ear(eye):
    """Calculate the Eye Aspect Ratio (EAR)."""
    try:
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logging.error(f"EAR calculation failed: {e}")
        return 0.0  # Return a default EAR value in case of an error

def detect_blink(landmarks_list):
    """Detect blinking by counting EAR drops below a threshold."""
    try:
        blink_count = 0
        frame_counter = 0
        EAR_THRESHOLD = 0.20  # Optimized threshold (commonly 0.2 to 0.25)
        CONSECUTIVE_FRAMES = 1  # Faster detection for natural blinking

        if not landmarks_list or len(landmarks_list) == 0:
            return {"status": False, "message": "No face landmarks detected for blink detection."}

        for landmarks in landmarks_list:
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                frame_counter = 0

        return {
            "status": blink_count >= 1,
            "message": f"{blink_count} blinks detected." if blink_count >= 1 else "Insufficient blinks detected.",
            "blink_count": blink_count
        }
    except Exception as e:
        logging.error(f"Blink detection failed: {e}")
        return {"status": False, "message": f"Blink detection failed: {e}"}


def extract_landmarks(frame):
    """Extract facial landmarks from a single frame."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            return None
        face = faces[0]
        landmarks = predictor(gray, face)
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    except Exception as e:
        logging.error(f"Landmark extraction failed: {e}")
        return None


@csrf_exempt
def validate_user(request):
    """Handle photo capture, real-time checks, and liveness validation in one step."""
    if request.method != "POST":
        return JsonResponse({
            "status": "error",
            "message": "Invalid request method. Please use POST."
        }, status=405)  # Method Not Allowed

    try:
        # Decode video frames
        video_frames_data = request.FILES.getlist("video_frames[]")
        if not video_frames_data:
            return JsonResponse({
                "status": "error",
                "message": "No video frames provided."
            }, status=400)  # Bad Request

        # Decode frames from binary data
        video_frames = []
        for frame_file in video_frames_data:
            image = decode_image(frame_file) # Use decode_image to read each frame
            if image is None:
                return JsonResponse({
                    "status": "error",
                    "message": "Failed to decode one or more frames."
                }, status=400)  # Bad Request
            video_frames.append(image)


        # Perform real-time checks on the first frame
        first_frame = video_frames[0]
        blur_check = check_blur(first_frame)
        brightness_check = check_brightness(first_frame)
        distance_check = estimate_distance(first_frame)

        # Perform liveness check based on video frames
        landmarks_list = []
        for frame in video_frames:
            landmarks = extract_landmarks(frame)
            if landmarks:
                landmarks_list.append(landmarks)

        blink_detected = detect_blink(landmarks_list)  # No default message

        # Combine all checks results
        validation_results = {
            "blur_check": blur_check,
            "brightness_check": brightness_check,
            "distance_check": distance_check,
            "blink_detected": blink_detected,
        }

        # Determine overall status - MUST PASS ALL CHECKS
        overall_status = all(result['status'] for result in validation_results.values())

        # Construct response message. Provide SPECIFIC failure reasons.
        failure_reasons = []
        if not blur_check['status']:
            failure_reasons.append(blur_check['message'])
        if not brightness_check['status']:
            failure_reasons.append(brightness_check['message'])
        if not distance_check['status']:
            failure_reasons.append(distance_check['message'])
        if not blink_detected['status']:
            failure_reasons.append(blink_detected['message'])

        if overall_status:
            overall_message = "User validation passed."
        else:
            overall_message = "User validation failed: " + ", ".join(failure_reasons)  # Specific reasons

        # Create response data
        response_data = {
            "status": "success" if overall_status else "failure",
            "message": overall_message,
            "validation_results": validation_results,
        }

        # Convert numpy.bool_ to bool for JSON serialization
        for key, value in response_data["validation_results"].items():
            if isinstance(value, dict) and "status" in value:
                response_data["validation_results"][key]["status"] = bool(value["status"])

        # SAVE VALIDATION DATA AND QUALITY IMAGE IF VALIDATION PASSES
        if overall_status:
            # Select best frame

            best_frame = None
            best_score = -1  # Initialize with a low score

            for frame in video_frames:
                # Calculate sharpness (Laplacian variance)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

                # Calculate brightness
                brightness = np.mean(gray)

                # Calculate distance
                distance_result = estimate_distance(frame)
                distance = distance_result.get('distance', None)  # Get distance or None

                # Define weights for each factor (adjust as needed)
                sharpness_weight = 0.6
                brightness_weight = 0.2
                distance_weight = 0.2

                # Score the frame
                score = (
                    sharpness_weight * laplacian_var +
                    brightness_weight * (100 - abs(brightness - 128)) +  # Prefer brightness near 128
                    distance_weight * (100 - abs(distance - 50)) if distance is not None else 0 # If distance is valid
                )

                if score > best_score:
                    best_score = score
                    best_frame = frame

            if best_frame is None:
                quality_image = first_frame  # Default to first frame if no better frame is found
            else:
                quality_image = best_frame

            save_validation_data(quality_image, response_data)

        # Return JSON response
        return JsonResponse(response_data)

    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        return JsonResponse({
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }, status=500)


def save_validation_data(image, validation_data):
    """Saves the validation data and image to a structured directory."""
    try:
        # Create a directory to save the images and data (structured by timestamp)
        timestamp = timezone.now().strftime("%Y%m%d_%H%M%S")
        save_directory = os.path.join(settings.MEDIA_ROOT, 'images', timestamp)
        os.makedirs(save_directory, exist_ok=True)  # Ensure directory exists

        # Save the validation data to a JSON file
        data_file_path = os.path.join(save_directory, 'validation_data.json')
        with open(data_file_path, 'w') as f:
            json.dump(validation_data, f, indent=4)  # Save with indentation for readability

        # Save the image
        filename = 'validation_image.jpg'
        file_path = os.path.join(save_directory, filename)
        cv2.imwrite(file_path, image)  # Use cv2.imwrite to save the image
        logging.info(f"Saved image to {file_path}")

        logging.info(f"Validation data and image saved to {save_directory}")

    except Exception as e:
        logging.error(f"Error saving validation data: {str(e)}")