import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ------------------------------
# Load your SavedModel
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cat_dog_model_saved")  # folder path
    return model

model = load_model()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and the model will predict whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    pred_class = "Dog" if prediction[0][0] > 0.5 else "Cat"
    
    st.success(f"Prediction: **{pred_class}**")