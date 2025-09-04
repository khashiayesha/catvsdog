import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import requests

#  Konfigurasi 
MODEL_PATH = "cnn_best_final.h5"
HUGGINGFACE_URL = "https://huggingface.co/ayeshaca/catvsdog/resolve/main/cnn_best_final.h5"
IMG_SIZE = (128, 128)

# Download model
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Mengunduh model dari Hugging Face...")
        r = requests.get(HUGGINGFACE_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.write(" Model berhasil diunduh.")

# Load model 
@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Judul 
st.title("Klasifikasi Gambar Kucing vs Anjing 🐱🐶")

# Upload Gambar
uploaded_file = st.file_uploader("Upload gambar kucing atau anjing:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", use_container_width=True)

    # Preprocessing
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # Prediksi
    preds = model.predict(img_array)
    class_idx = int(preds[0][0] > 0.5)  # 0 = Cat, 1 = Dog
    label = "Dog 🐶" if class_idx == 1 else "Cat 🐱"
    confidence = preds[0][0] if class_idx == 1 else 1 - preds[0][0]

    st.subheader(f"Hasil Prediksi: {label}")
    st.write(f"Confidence: **{confidence:.2f}**")
