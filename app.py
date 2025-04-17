import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from skimage.feature import hog
from skimage import color
from PIL import Image
import hpelm

# Load model dari file
elm = hpelm.HPELM(0, 0)  # Placeholder dimensi, akan diganti oleh .load()
elm.load('model_elm.model')

# Label kelas
label_dict = {
    0: "Jinak",
    1: "Ganas",
    2: "Normal"
}

# Fungsi untuk ekstrak fitur HOG
def extract_hog_features(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_gray = color.rgb2gray(img_array)
    features, _ = hog(img_gray, 
                      pixels_per_cell=(8, 8), 
                      cells_per_block=(2, 2), 
                      visualize=True)
    return features

# Tampilan Streamlit
st.set_page_config(page_title="Deteksi Kanker Paru", layout="centered")
st.title("🫁 Deteksi Kanker Paru-paru dari Citra CT-scan")
st.write("Upload gambar CT-scan paru-paru untuk memprediksi statusnya.")

uploaded_file = st.file_uploader("Unggah gambar CT-scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # Ekstraksi fitur dan prediksi
    features = extract_hog_features(img)
    features = np.expand_dims(features, axis=0)

    prediction = elm.predict(features)
    predicted_class = np.argmax(prediction)

    st.markdown(f"### Hasil Prediksi: **{label_dict[predicted_class]}**")
