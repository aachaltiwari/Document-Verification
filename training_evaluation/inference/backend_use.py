import os
from transformers.utils.logging import set_verbosity_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_verbosity_error()

from trocr_english import ocr_document as ocr_document_en
from trocr_nepali import ocr_document as ocr_document_np
from document_classification import document_predict
from face_recog import face_rec
from fuzzy_logic import find_similarity_location, load_json_file, extract_result


print()
#################################################################################################################
# Usage of document classification

# inputs
image_path = 'input/images/id6.jpg'

document_type = document_predict(image_path)
for key, value in document_type.items():
    print(f"{key}: {value}")

print()
#################################################################################################################
# Usage of face recognition

# inputs
citizenship_face = "input/images/id6.jpg"
pp_face = "input/images/aachal_pp.png"

results = face_rec(citizenship_face, pp_face)
for key, value in results.items():
    print(f"{key}: {value}")

print()
#################################################################################################################
# Usage of OCR detection

# inputs
image_path = 'input/images/id6.jpg'
data_json = load_json_file('input/forms/citizenship.json')
language = "en"

text_output = None
if language == "en":
    text_output = ocr_document_en(image_path)
elif language == "np":
    text_output = ocr_document_np(image_path)

results = extract_result(data_json, text_output)
for key, value in results.items():
    print(f"{key}: {value}")

