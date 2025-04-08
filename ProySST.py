import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os
import asyncio
import tempfile

# Estilo CSS
st.markdown("""
    <style>
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        text-align: center;
        width: 90%;
        max-width: 300px;
        margin: 10px auto;
    }
    .card-title {
        font-size: 1.2em;
        color: black;
    }
    .card-image {
        width: 100%;
        height: auto;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar clases desde archivo
def cargar_clases():
    try:
        with open("clasesSST.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo clasesSST.txt.")
        return []

CLASES = cargar_clases()
detected_classes = set()

def create_card(title, image_url):
    return f"""
    <div class="card">
        <img class="card-image" src="{image_url}" alt="{title}">
        <div class="card-title">{title}</div>
    </div>
    """

def get_class_html(cls, detected_classes):
    active = 'background-color:#FF4B4B;color:white;' if cls in detected_classes else 'background-color:white;color:black;'
    return f'<span style="padding:4px 6px;border-radius:5px;margin:3px;{active} display:inline-block;">{cls}</span>'

# Cargar modelo YOLOv8
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Asegúrate de usar un modelo entrenado para SST

model = load_model()

# Interfaz principal
def main():
    st.title("🦺 Detección de Implementos de Seguridad")
    actividades = ["Principal", "Subir imagen"]
    choice = st.sidebar.selectbox("Selecciona una opción", actividades)
    st.sidebar.markdown("---")

    if choice == "Principal":
        st.markdown("### Aplicación para reconocer implementos de seguridad como casco, chaleco, botas, etc.")
        st.markdown(f"<div style='padding:6px; border:2px solid #FF4B4B; border-radius:10px;'><h4 style='text-align:center;'>Clases</h4><p style='text-align:center;'>{' '.join([get_class_html(cls, detected_classes) for cls in CLASES])}</p></div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.markdown(create_card("📸 Subir Imagen", "https://i.pinimg.com/736x/e1/91/5c/e1915cea845d5e31e1ec113a34b45fd8.jpg"), unsafe_allow_html=True)
        col2.markdown(create_card("🎥 Subir Video", "https://static.vecteezy.com/system/resources/previews/005/919/290/original/video-play-film-player-movie-solid-icon-illustration-logo-template-suitable-for-many-purposes-free-vector.jpg"), unsafe_allow_html=True)

    elif choice == "Subir imagen":
        confianza = st.sidebar.slider('Nivel de confianza', 0.0, 1.0, 0.3, 0.05)
        image = st.file_uploader("📂 Sube una imagen", type=['jpg', 'jpeg', 'png'])

        if image:
            col1, col2 = st.columns(2)
            col1.image(image, caption="📷 Imagen original", use_column_width=True)

            with st.spinner("🔍 Detectando..."):
                img = Image.open(image).convert("RGB")
                results = model(img, conf=confianza)
                result = results[0]

                if result:
                    annotated_img = result.plot()
                    col2.image(annotated_img, caption="🧠 Detección", use_column_width=True)

                    for r in result.boxes:
                        class_id = int(r.cls.cpu().numpy()[0])
                        conf = float(r.conf.cpu().numpy()[0])
                        label = CLASES[class_id] if class_id < len(CLASES) else f"ID {class_id}"
                        detected_classes.add(label)
                        st.markdown(f"<div style='background:#f0f0f0;padding:5px;margin:5px 0;border-radius:5px;'>🧷 <b>{label}</b> — Confianza: {conf:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No se detectaron objetos en la imagen.")

            # Mostrar clases detectadas
            st.markdown("<br><h5>Clases detectadas:</h5>", unsafe_allow_html=True)
            st.markdown(" ".join([get_class_html(cls, detected_classes) for cls in CLASES]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()



