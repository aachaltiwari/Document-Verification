from PIL import Image, ImageDraw, ImageFont
import random

def create_image_with_text(nepali_word, fonts, output_path, compression_quality=100, size_ratio=0.5):
    background_colors = [
        'white', 'white', 'white', 'white', 'white',
        '#F4F4F4', '#F7F7F7', '#F9F9F9', '#F2F2F2', '#EFEFEF',
        '#E0E0E0', '#D3D3D3', '#C0C0C0', '#BEBEBE', '#A9A9A9',
        '#DCDCDC', '#D8D8D8', '#D6D6D6', '#CCCCCC', '#C7C7C7',
        '#F5DEB3', '#D2B48C', '#BC8F8F', '#DEB887', '#CD853F',
        '#FFFACD', '#FAFAD2', '#FFFFE0', '#FFF8DC', '#EEE8AA',
        '#FFE4B5', '#FFDAB9', '#FFDEAD', '#FFE4C4', '#FFEFD5',
        '#F0F8FF', '#ADD8E6', '#B0E0E6', '#87CEEB', '#AFEEEE',
        '#98FB98', '#90EE90', '#ADFF2F', '#7FFFAA', '#BDFCC9',
        '#FFB6C1', '#FFC0CB', '#DDA0DD', '#E6E6FA', '#F5DEB3',
        '#FADADD', '#F5E1FD', '#E1F5FE', '#D4F1F9', '#F0EAD6',
        '#F5F5DC', '#FAEBD7', '#FFEBCD', '#FFF8DC', '#FDF5E6',
        '#EED9C4', '#EECBAD', '#CDAA7D', '#E3C9A8', '#D2B48C',
        '#E5E5E5', '#EAEAEA', '#EDEDED', '#F1F1F1', '#F6F6F6',
        '#FFFFF0', '#FFFFE0', '#FFFDD0', '#FDFD96', '#FAF884',
        '#FFE5B4', '#FFDAB9', '#FFCC99', '#FFC87C', '#FFB347'
    ]
    
    # Randomly select a font and background color
    font_path = random.choice(fonts)
    background_color = random.choice(background_colors)
    
    # Initialize font and calculate text dimensions
    font_size = 50
    font = ImageFont.truetype(font_path, font_size)
    
    # Create a temporary image to calculate text size
    temp_img = Image.new('RGB', (1, 1), 'white')
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), nepali_word, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Add padding and create the final image
    padding = 25
    img_width = text_width + 2 * padding
    img_height = text_height + 2 * padding
    img = Image.new('RGB', (img_width, img_height), background_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate text position (centered vertically)
    text_x = padding
    text_y = (img_height - text_height) // 3.5
    draw.text((text_x, text_y), nepali_word, fill="black", font=font)
    
    # Resize the image while maintaining aspect ratio
    new_width = int(img_width * size_ratio)
    new_height = int(img_height * size_ratio)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save the image with compression
    img.save(output_path, 'PNG', quality=compression_quality, optimize=True)



# # Usage
# fonts = ['fonts/PragatiNarrow-Regular.ttf']
# nepali_word = 'ॐ र्हि ऋषि 123 १२३९'
# output_path = 'dataset/test/output.png'
# create_image_with_text(nepali_word, fonts, output_path)