from image_generate import create_image_with_text
from edit_image import enhance_image_stepwise, rotate_image, add_random_background, add_random_noise

import random
import cv2
import csv
import os

# Background colors (light colors)
background_colors = [
    'white', '#F0F8FF', '#FAEBD7', '#F5F5DC', '#FFF8DC', '#FFE4C4',
    '#FFEBCD', '#F5DEB3', '#D2B48C', '#EEE8AA', '#98FB98', '#AFEEEE',
    '#DDA0DD', '#FFB6C1', '#E0E0E0', '#D3D3D3', '#C0C0C0', '#BEBEBE',
    '#A9A9A9', '#DCDCDC', '#D8D8D8', '#D6D6D6', '#CCCCCC', '#C7C7C7'
]


def get_image_path(index, text, fonts):
    nepali_word = text
    output_path = f"dataset_test/generated_images/{index}.png"
    create_image_with_text(nepali_word, fonts, output_path)

    input_image_path = output_path
    output_image_path = f"dataset_test/preprocessed_images/{index}.png"
    contrast = random.choice([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3])
    kernel_size = random.choice([1, 1, 1, 3, 3, 5])
    angle = random.choice([-4, -3, -2, -1, -1, 0,
                        0, 1, 1, 2, 3, 4])
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    enhanced_image = enhance_image_stepwise(image, contrast, kernel_size)
    rotated_image = rotate_image(enhanced_image, angle)
    noisy_image = add_random_noise(rotated_image, noise_intensity=0.05)
    final_image = add_random_background(noisy_image, background_colors)
    cv2.imwrite(output_image_path, final_image)

    return output_image_path


# File paths
input_txt_file = "dataset_test/data.txt"
output_csv_file = "dataset_test/data.csv"

fonts_folder = "fonts"
fonts = [os.path.join(fonts_folder, font) for font in os.listdir(fonts_folder) if font.endswith('.ttf')]

if not os.path.exists("dataset_test/generated_images"):
        os.makedirs("dataset_test/generated_images")
if not os.path.exists("dataset_test/preprocessed_images"):
        os.makedirs("dataset_test/preprocessed_images")

with open(input_txt_file, "r", encoding="utf-8") as txt_file, open(output_csv_file, "w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_file", "text"])
    count = 0
    for line in txt_file:
        
        text = line.strip()
        if len(text) > 31:
            # print(text, len(text))
            continue    
        if text:
            image_path = get_image_path(count, text, fonts)
            csv_writer.writerow([image_path, text])
            count += 1

        if count % 1000 == 0:
            print(f"Reached line {count}...")
            
        if count == 20000:
            break

print(f"Data has been successfully saved to {output_csv_file}")



