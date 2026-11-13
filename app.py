import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = (128, 128)   

@st.cache_resource
def muat_model():
    return tf.keras.models.load_model('face_cnn.keras', compile=False)

model = muat_model()

st.title('Deteksi Wajah Real vs Fake')

file = st.file_uploader('Unggah foto wajah', type=['jpg', 'jpeg', 'png'])

if file is not None:
    img = Image.open(file).convert('RGB')
    st.image(img, width=300)

    arr = np.expand_dims(np.array(img.resize(IMG_SIZE, Image.BILINEAR), dtype=np.float32), axis=0)
    p_real = float(model.predict(arr, verbose=0)[0][0])  
    label = 'REAL' if p_real > 0.5 else 'FAKE'
    yakin = p_real if label == 'REAL' else 1 - p_real

    st.subheader(f'Prediksi: {label} ({yakin * 100:.1f}%)')
    st.progress(p_real)