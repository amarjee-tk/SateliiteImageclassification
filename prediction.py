import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Set image dimensions (should match the dimensions used during training)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Define the labels (replace these with your actual class names)
labels = ['Cloudy', 'Green_Area', 'Desert', 'Water']

def preprocess_single_image(image_path):
    """
    Preprocess a single image for prediction.
    """
    print(f"Loading image from: {image_path}")
    # Load the image
    img = Image.open(image_path)

    # Resize the image to the required dimensions
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert the image to an array
    img_array = np.array(img_resized)

    # Check if the image has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        raise ValueError("Input image must have 3 color channels (RGB).")

    # Normalize the image to the range [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 128, 128, 3)

    print(f"Processed image shape: {img_array.shape}")
    return img_array

def predict_single_image(image_path, model, labels):
    """
    Predict the class of a single image using the trained model.
    """
    # Preprocess the image
    processed_image = preprocess_single_image(image_path)

    print("Making prediction...")
    # Make a prediction
    prediction = model.predict(processed_image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction)

    # Convert the index to class name
    predicted_class_name = labels[predicted_class_index]

    print(f"Prediction: {prediction}")
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted class name: {predicted_class_name}")

    return predicted_class_name, prediction

# Example usage
if __name__ == "__main__":
    # Load your trained model (replace 'your_model.h5' with your actual model file)
    model_path = 'satellite_image_classification_model.h5'  # Update with the actual path to your model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    # Path to the image you want to predict (update with your image path)
    image_path = 'Forest_96.jpg'  # Update with the actual path to your image

    # Check if the image exists
    if os.path.exists(image_path):
        # Get the prediction
        predicted_class, prediction = predict_single_image(image_path, model, labels)

        # Print the prediction results
        print(f"Predicted class: {predicted_class}")
        print(f"Prediction probabilities: {prediction}")
    else:
        print(f"Image not found at path: {image_path}")
