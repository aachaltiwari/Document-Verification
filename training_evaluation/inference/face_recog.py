import cv2
import face_recognition
import numpy as np


def linear_to_quadratic(x):
    return ((-(x/100-1)**2) + 1)*100


def face_rec(img_path1, img_path2):
    
    def get_face_encoding_and_location(img_path):
        try:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            if len(face_locations) == 0:
                raise ValueError("No faces detected in the image.")
            elif len(face_locations) > 1:
                return face_encodings[0], face_locations[0]
            else:
                return face_encodings[0], face_locations[0]
            
        except Exception as e:
            raise RuntimeError(f"Error processing image '{img_path}': {str(e)}")
    
    img_encoding1, face_locations1 = get_face_encoding_and_location(img_path1)
    img_encoding2, face_locations2 = get_face_encoding_and_location(img_path2)
    
    # Calculate the face distance and convert to percentage similarity
    face_distances = face_recognition.face_distance([img_encoding1], img_encoding2)
    similarity_score = 1 - face_distances[0]  # Similarity is 1 minus the distance
    similarity_percentage = similarity_score * 100  # Convert to percentage
    
    # Compare faces using face_recognition
    match_results = face_recognition.compare_faces([img_encoding1], img_encoding2)
    
    results = {
        'similarity_percentage': linear_to_quadratic(similarity_percentage),
        'face_distance': face_distances[0],
        'face_locations': [face_locations1, face_locations2],
        'match_results': match_results,
        'all_face_locations_img1': face_recognition.face_locations(cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)),
        'all_face_locations_img2': face_recognition.face_locations(cv2.cvtColor(cv2.imread(img_path2), cv2.COLOR_BGR2RGB))
    }
    
    return results


# usage
# citizenship_face = "input/images/aachal.jpg"
# pp_face = "input/images/aachal_pp.png"
# result = face_rec(citizenship_face, pp_face)
# for key, value in result.items():
#     print(f"{key}: {value}")
