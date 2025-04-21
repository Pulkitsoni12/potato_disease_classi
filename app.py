import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


BATCH_SIZE = 32
IMAGE_SIZE = 256

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "train\PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names


# Load pre-trained model (ensure the model file is in the same directory or specify the path)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")  # Change the path accordingly
    return model

model = load_model()

# Title of the web app
st.title("Image Classification App")

# File uploader
uploaded_file = st.file_uploader("Drag and drop an image or click to upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  # Adjust size as per model requirement
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction, axis=1)[0]]
    confidence = round(100 * (np.max(prediction[0])), 2)
    
    # Display prediction
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"confidence : {confidence}")


