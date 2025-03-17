import os
import cv2
import numpy as np
from transformers import AutoProcessor, VisionEncoderDecoderModel, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

from text_detection import text_detection

# Load your fine-tuned Nepali OCR model and processor
processor = AutoProcessor.from_pretrained("models/trocr/part_3/nepali_ocr_processor", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("models/trocr/part_3/nepali_ocr_model")


def ocr_detect(path):
    # Open and preprocess the image
    image = Image.open(path).convert("RGB")


    image_cv = np.array(image)
    
    # Step 1: Denoise the full image (this version will serve as our 'clean' background)
    denoised_image = cv2.bilateralFilter(image_cv, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Step 2: Create a mask for the text
    # Convert original image to grayscale for mask creation
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    # Use Otsu's thresholding to detect dark text on a light background.
    # THRESH_BINARY_INV makes the text white (foreground) and background black.
    _, text_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Step 3: Create a background mask by inverting the text mask
    background_mask = cv2.bitwise_not(text_mask)
    # Convert the mask to 3 channels so it can be used with a color image
    background_mask_colored = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    
    # Step 4: Composite the final image
    # Where the background mask is white, take pixels from the denoised image;
    # otherwise, use the original image (preserving the text).
    image = np.where(background_mask_colored == 255, denoised_image, image_cv)
    
    
    # # Display the image
    # plt.imshow(image)
    # plt.axis('off')  # Hide axis labels
    # plt.show()
    
    # Preprocess image and generate text
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text


def replace_chars(extracted_text, char_map):
    return ''.join(char_map.get(char, char) for char in extracted_text)


def ocr_document(image_path):
    text_detection(image_path)
    image_dir = "results/cropped_images"

    nepali_mapping = {'ः': '¶','ऊ': '§','ऋ': '†','ऌ': '‡','ऐ': '‰','ङ': '¤','ञ': '¢','ॐ': '£','ॠ': '¿',
        '०': '∞','१': '∑','२': '∂','३': '∇','४': '∆','५': '⌘','६': '≈','७': '≠','८': '¢','९': '¥'}
    nepali_reverse_mapping = {v: k for k, v in nepali_mapping.items()}

    ocr_results = None
    results = []

    for image_name in sorted(os.listdir(image_dir)):
        if image_name.endswith('.png'):
            image_path = os.path.join(image_dir, image_name)
            # Detect text using the OCR model
            extracted_text = ocr_detect(image_path)
            extracted_text = replace_chars(extracted_text, nepali_reverse_mapping)
            
            results.append(extracted_text)

            ocr_results = " ".join(results)

    with open("results/ocr_nep_results.txt", "w", encoding="utf-8") as f:
        for text in results:
            f.write(text + "\n")

    return ocr_results


#usage
# ocr_results = ocr_document('input/images/citizenship.jpg')
# print(ocr_results)