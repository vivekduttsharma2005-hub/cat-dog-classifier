# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ------------------------------
# Load the lightweight SavedModel
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_model_small")  # folder, not .h5
    return model

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🐱🐶 Cat vs Dog Classifier (Lightweight Model)")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(image.resize((128,128)))/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    pred_class = "Dog" if prediction[0][0] > 0.5 else "Cat"

    st.success(f"Prediction: **{pred_class}**")