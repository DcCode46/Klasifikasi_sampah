import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Mapping subkelas ke kategori utama
kategori_mapping = {
    "Sisa_Buah_dan_Sayur": "Organik",
    "Sisa_Makanan": "Organik",
    "Alumunium": "Anorganik",
    "Kaca": "Anorganik",
    "Kardus": "Anorganik",
    "Karet": "Anorganik",
    "Kertas": "Anorganik",
    "Plastik": "Anorganik",
    "Styrofoam": "Anorganik",
    "Tekstil": "Anorganik",
    "Alat_Pembersih_Kimia": "B3",
    "Baterai": "B3",
    "Lampu_dan_Elektronik": "B3",
    "Minyak_dan_Oli_Bekas": "B3",
    "Obat_dan_Medis": "B3"
}

# Urutan label sesuai model
class_names = [
    "Alat_Pembersih_Kimia", "Alumunium", "Baterai", "Kaca", "Kardus",
    "Karet", "Kertas", "Lampu_dan_Elektronik", "Minyak_dan_Oli_Bekas",
    "Obat_dan_Medis", "Plastik", "Sisa_Buah_dan_Sayur", "Sisa_Makanan",
    "Styrofoam", "Tekstil"
]

# Load model
model = tf.keras.models.load_model("model_sampah.h5")

# Fungsi preprocessing gambar
def preprocess_image(image):
    img = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Halaman Tab
tab1, tab2 = st.tabs(["🏠 Home", "🔍 Prediksi"])

# ---------------- TAB HOME ----------------
with tab1:
    st.title("♻️ Klasifikasi Sampah")
    st.write("Aplikasi ini dapat memprediksi jenis sampah dan kategori utamanya.")

    st.subheader("Jenis Sampah yang Bisa Diprediksi")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("Sisa_Buah_dan_Sayur.jpg", caption="Sisa Buah & Sayur (Organik)", use_container_width=True)
        st.image("Sisa_Makanan.jpg", caption="Sisa Makanan (Organik)", use_container_width=True)
        st.image("Alumunium.jpg", caption="Alumunium (Anorganik)", use_container_width=True)
        st.image("Alat_Pembersih_Kimia.jpg", caption="Alat Pembersih Kimia (B3)", use_container_width=True)
        st.image("Karet.jpeg", caption="Karet (Anorganik)", use_container_width=True)

    with col2:
        st.image("Kaca.jpg", caption="Kaca (Anorganik)", use_container_width=True)
        st.image("Kardus.jpg", caption="Kardus (Anorganik)", use_container_width=True)
        st.image("Kertas.jpg", caption="Kertas (Anorganik)", use_container_width=True)
        st.image("Lampu_dan_Elektronik.jpeg", caption="Lampu & Elektronik (B3)", use_container_width=True)
        st.image("Minyak_dan_Oli_Bekas.jpeg", caption="Minyak & Oli Bekas (B3)", use_container_width=True)

    with col3:
        st.image("Plastik.jpg", caption="Plastik (Anorganik)", use_container_width=True)
        st.image("Baterai.jpg", caption="Baterai (B3)", use_container_width=True)
        st.image("Obat_dan_Medis.jpg", caption="Obat & Medis (B3)", use_container_width=True)
        st.image("Styrofoam.jpeg", caption="Styrofoam (Anorganik)", use_container_width=True)
        st.image("Tekstil.jpg", caption="Tekstil (Anorganik)", use_container_width=True)


    st.info("Pastikan gambar yang diunggah jelas agar hasil prediksi akurat.")

# ---------------- TAB PREDIKSI ----------------
with tab2:
    st.title("🔍 Prediksi Sampah")

    # Pilihan metode input
    pilihan_input = st.radio("Pilih metode input gambar:", ["📁 Upload", "📷 Kamera"])

    image = None
    if pilihan_input == "📁 Upload":
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    elif pilihan_input == "📷 Kamera":
        camera_file = st.camera_input("Ambil gambar dengan kamera")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")
            st.image(image, caption="Gambar dari kamera", use_container_width=True)

    # Tombol Prediksi
    if image is not None and st.button("Prediksi"):
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        pred_index = np.argmax(predictions)
        subkelas_pred = class_names[pred_index]
        kategori_pred = kategori_mapping[subkelas_pred]
        confidence = predictions[0][pred_index] * 100

        st.success(f"**Subkelas:** {subkelas_pred}")
        st.info(f"**Kategori Utama:** {kategori_pred}")
        st.write(f"**Confidence:** {confidence:.2f}%")
