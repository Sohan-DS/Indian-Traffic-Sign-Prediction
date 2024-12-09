import os
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 64
MODEL_PATH = r"E:\DLP\traffic_sign_model.keras"  # Update this path
DATASET_PATH = r"E:\DLP\Dataset\train"  # Update with your dataset path

# Function to retrieve class names dynamically
def get_class_names(dataset_path):
    return sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])

CLASS_NAMES = get_class_names(DATASET_PATH)

# Load the trained model
@st.cache_resource
def load_traffic_sign_model():
    return load_model(MODEL_PATH)

model = load_traffic_sign_model()

# Function to preprocess and predict the image
def predict_image(image, model):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH)).convert("RGB")  # Resize and ensure 3 channels
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return predicted_class

# Streamlit app layout
st.title("Traffic Sign Recognition")
st.write("Upload an image of a traffic sign to identify it.")

# File uploader
uploaded_file = st.file_uploader("Choose a traffic sign image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction checkbox
    if st.checkbox("Check"):
        # Predict the traffic sign
        predicted_class = predict_image(image, model)
        st.write(f"**Predicted Sign:** {predicted_class}")