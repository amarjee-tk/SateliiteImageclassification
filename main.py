import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('satellite_image_classification_model.h5')



from PIL import Image
import numpy as np

# def preprocess_image(image):
#     # Resize the image to the model input size (change as per your model)
#     image = image.resize((128, 128))  
#     # Convert to numpy array and normalize
#     image = np.array(image) / 255.0
#     # Expand dimensions for model input (assuming the model expects batches)
#     image = np.expand_dims(image, axis=0)
#     return image




def preprocess_single_image(image_path):
    # Set image dimensions (same as used in training)
    IMG_HEIGHT = 128
    IMG_WIDTH = 128

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

    return img_array



def predict(image, model):
    # Preprocess the image
    processed_image = preprocess_single_image(image)
    # Make a prediction
    prediction = model.predict(processed_image)
    # Get the class with the highest probability
    class_names = ['Cloudy', 'Green Area', 'Desert', 'Water']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class


import streamlit as st

# # Set up the Streamlit interface
# st.title("Satellite Image Classification")

# # Let the user upload an image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

# if uploaded_file is not None:
#     # Open the uploaded image
#     image = Image.open(uploaded_file)
    
#     # Display the image
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     # Make a prediction when the user clicks the button
#     if st.button("Classify Image"):
#         # Predict the class
#         prediction = predict(image, model)
        
#         # Display the prediction
#         st.write(f"Prediction: {prediction}")

if uploaded_file is not None:
    # Preprocess the uploaded image for prediction
    img_array = preprocess_single_image(uploaded_file)

    # Make a prediction using your model
    prediction = model.predict(img_array)

    # Get the class names
    class_names = ['Cloudy', 'Green_Area', 'Desert', 'Water']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f'Prediction: {predicted_class}')


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

prediction = predict(uploaded_file, model)
print(prediction)

