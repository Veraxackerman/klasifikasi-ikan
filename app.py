import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ─── Konfigurasi Halaman ───────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Kesegaran Ikan",
    page_icon="🐟",
    layout="centered"
)

# ─── CSS Kustom ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & background */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Header utama */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 { font-size: 1.9rem; font-weight: 700; color: #1a1a2e; margin:0; }
    .main-header p  { color: #555; font-size: 0.95rem; margin-top: 0.3rem; }

    /* Kartu hasil */
    .result-card {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .result-fresh    { background: #e8f8f0; border: 2px solid #27ae60; }
    .result-notfresh { background: #fdf0f0; border: 2px solid #e74c3c; }
    .result-label    { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
    .result-conf     { font-size: 1rem; color: #444; }
    .label-fresh     { color: #27ae60; }
    .label-notfresh  { color: #e74c3c; }

    /* Info box kecil */
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: #333;
        margin-top: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }

    /* Sembunyikan menu Streamlit */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Muat model dari file .keras — sesuaikan path jika perlu."""
    model_path = "model_final.keras"          # ← sesuaikan path model kamu
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


# ─── Fungsi Prediksi ───────────────────────────────────────────
def predict(model, img: Image.Image):
    """Preprocess lalu prediksi kesegaran ikan."""
    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    # prob mendekati 1 → not fresh  |  prob mendekati 0 → fresh
    label = "not fresh" if prob >= 0.5 else "fresh"
    conf  = prob if prob >= 0.5 else 1.0 - prob
    return label, conf, prob


# ─── UI Utama ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐟 Klasifikasi Kesegaran Ikan</h1>
    <p>Sistem deteksi otomatis kesegaran ikan menggunakan CNN MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load model
model = load_model()

if model is None:
    st.error(
        "⚠️ **File model tidak ditemukan.**\n\n"
        "Pastikan file `model_final.keras` berada di folder yang sama dengan `app.py`. "
        "File model dihasilkan setelah menjalankan notebook pelatihan."
    )
    st.stop()

# Upload gambar
st.subheader("📤 Unggah Citra Ikan")
uploaded = st.file_uploader(
    "Pilih gambar ikan (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is not None:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Gambar yang Diunggah**")
        st.image(img, use_column_width=True)

    with col2:
        st.markdown("**Hasil Klasifikasi**")

        with st.spinner("Menganalisis citra..."):
            label, conf, prob = predict(model, img)

        # Kartu hasil
        if label == "fresh":
            st.markdown(f"""
            <div class="result-card result-fresh">
                <div class="result-label label-fresh">✅ FRESH</div>
                <div class="result-conf">Ikan dalam kondisi <b>segar</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-notfresh">
                <div class="result-label label-notfresh">❌ NOT FRESH</div>
                <div class="result-conf">Ikan <b>tidak segar</b> / sudah menurun kualitasnya</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Tingkat Kepercayaan Model**")
        st.progress(conf)
        st.caption(f"{conf * 100:.2f}% yakin → **{label.upper()}**")

        # Probabilitas kedua kelas
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probabilitas per Kelas**")
        prob_fresh    = 1.0 - prob
        prob_notfresh = prob

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🟢 Fresh",     f"{prob_fresh * 100:.1f}%")
        with col_b:
            st.metric("🔴 Not Fresh", f"{prob_notfresh * 100:.1f}%")

    # Info tambahan
    st.markdown(f"""
    <div class="info-box">
        🔍 <b>Info Gambar:</b> Ukuran asli {img.width} × {img.height} px —
        Diproses pada resolusi <b>224 × 224 px</b> sesuai input MobileNetV2.
    </div>
    """, unsafe_allow_html=True)

else:
    # Placeholder saat belum ada gambar
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem; background:#f9fafb;
                border-radius:12px; border: 2px dashed #d1d5db; color:#6b7280;">
        <div style="font-size:2.5rem;">📷</div>
        <div style="font-size:1rem; margin-top:0.5rem;">
            Unggah foto ikan untuk memulai analisis
        </div>
        <div style="font-size:0.85rem; margin-top:0.3rem; color:#9ca3af;">
            Format yang didukung: JPG, JPEG, PNG
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Panduan singkat
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("ℹ️ Cara Penggunaan"):
        st.markdown("""
        1. Klik tombol **Browse files** di atas (atau drag & drop gambar)
        2. Pilih foto ikan yang ingin diidentifikasi kesegarannya
        3. Sistem akan otomatis menganalisis dan menampilkan hasil
        4. Hasil berupa kategori **Fresh** (segar) atau **Not Fresh** (tidak segar)
        """)

    with st.expander("🐠 Jenis Ikan yang Didukung"):
        st.markdown("""
        Model ini dilatih menggunakan dataset yang terdiri dari tiga jenis ikan:
        - 🐟 **Ikan Nila**
        - 🐡 **Ikan Kembung**
        - 🐠 **Ikan Tuna**
        """)

# ─── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Sistem Klasifikasi Kesegaran Ikan · CNN MobileNetV2 · Skripsi 2025
</div>
""", unsafe_allow_html=True)
