import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Load model
model = tf.keras.models.load_model("models/document_classifier/best_model_4class.keras")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


img_height, img_width = 500, 500
class_labels = ["citizenship_back", "citizenship_front", "id_card", "random"]

# Load and predict
def document_predict(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.image.grayscale_to_rgb(img_array)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_probabilities = prediction[0]
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    
    dict_probabilities = dict(zip(class_labels, class_probabilities))

    # plt.imshow(img)
    # plt.title(f"Predicted: {predicted_label}")
    # plt.axis('off')
    # plt.show()

    return dict_probabilities


# Example usage:
# output = document_predict("input/images/aachal.jpg")
# for key, value in output.items():
#         print(f"{key}: {value:.2f}")

