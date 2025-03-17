import cv2
import numpy as np
import random
from PIL import ImageColor

def enhance_image_stepwise(image_array, contrast, kernel_size, brightness=1):
    # Step 1: Denoise the image (works on color images)
    denoised_image = cv2.bilateralFilter(image_array, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: Increase contrast using scaling (works on color images)
    contrast_enhanced = cv2.convertScaleAbs(denoised_image, alpha=contrast, beta=brightness)
    
    # Step 3: Strong sharpening with a custom kernel (works on color images)
    strong_sharpen_kernel = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, strong_sharpen_kernel)
    
    # Step 4: Soften the image using Gaussian Blur (works on color images)
    softened_image_gaussian = cv2.GaussianBlur(sharpened, (kernel_size, kernel_size), 0)
    
    return softened_image_gaussian


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated


def add_random_noise(image, noise_intensity=0.1):
    noise = np.random.randint(-255 * noise_intensity, 255 * noise_intensity, image.shape, dtype=np.int16)
    noisy_image = cv2.add(image.astype(np.int16), noise)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def add_random_background(image, background_colors):
    # Convert the image to grayscale for mask creation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a mask where the background is white
    mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
    # Randomly select a background color
    background_color = random.choice(background_colors)
    # Convert the background color to RGB using PIL.ImageColor
    background_color_rgb = ImageColor.getcolor(background_color, "RGB")
    # Create a new background
    background = np.full_like(image, background_color_rgb)
    # Combine the image and background using the mask
    result = cv2.bitwise_and(image, image, mask=mask) + cv2.bitwise_and(background, background, mask=~mask)
    return result




# Usage
# input_image_path = "dataset/generated_images/2.png"
# output_image_path = f"dataset/preprocessed_images/2.png"
# contrast = 1.5
# kernel_size = 1
# angle = -4
# background_colors = [
#     'white', '#F0F8FF', '#FAEBD7', '#F5F5DC', '#FFF8DC', '#FFE4C4',
#     '#FFEBCD', '#F5DEB3', '#D2B48C', '#EEE8AA', '#98FB98', '#AFEEEE',
#     '#DDA0DD', '#FFB6C1', '#E0E0E0', '#D3D3D3', '#C0C0C0', '#BEBEBE',
#     '#A9A9A9', '#DCDCDC', '#D8D8D8', '#D6D6D6', '#CCCCCC', '#C7C7C7'
# ]

# image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
# enhanced_image = enhance_image_stepwise(image, contrast, kernel_size)
# rotated_image = rotate_image(enhanced_image, angle)
# noisy_image = add_random_noise(rotated_image, noise_intensity=0.05)
# final_image = add_random_background(noisy_image, background_colors)
# cv2.imwrite(output_image_path, final_image)