import tensorflow as tf
from PIL import Image
import numpy as np
import streamlit as st

# Load your trained model
model = tf.keras.models.load_model('satellite_image_classification_model.h5')

# Set image dimensions (should match the dimensions used during training)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Define the labels (replace these with your actual class names)
labels = ['Cloudy', 'Green_Area', 'Desert', 'Water']

def preprocess_single_image(image):
    """
    Preprocess a single image for prediction.
    """
    print("Preprocessing image...")
    
    # Resize the image to the required dimensions
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert the image to an array
    img_array = np.array(img_resized)

    # # Check if the image has 3 channels (RGB)
    # if img_array.shape[-1] != 3:
    #     raise ValueError("Input image must have 3 color channels (RGB).")

    # Normalize the image to the range [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 128, 128, 3)

    print(f"Processed image shape: {img_array.shape}")
    return img_array

def predict_single_image(image, model, labels):
    """
    Predict the class of a single image using the trained model.
    """
    # Preprocess the image
    processed_image = preprocess_single_image(image)

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

# Set up the Streamlit interface
st.title("Satellite Image Classification")

# Let the user upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make a prediction when the user clicks the button
    if st.button("Classify Image"):
        # Predict the class
        predicted_class, prediction = predict_single_image(image, model, labels)
        
        # Display the prediction
        st.write(f"Prediction: {predicted_class}")
        st.write(f"Prediction probabilities: {prediction}")