import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

# --- Configuración de la aplicación ---
st.set_page_config(page_title="🦺 Detector de Seguridad Industrial", page_icon="🦺")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Cargar clases desde archivo ---
def cargar_clases():
    try:
        with open("clasesSST.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo clasesSST.txt.")
        return []

CLASES = cargar_clases()

# --- Cargar modelo ONNX ---
@st.cache_resource
def cargar_modelo_onnx():
    return ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])

session = cargar_modelo_onnx()
input_name = session.get_inputs()[0].name

# --- Preprocesamiento ---
def preprocesar_imagen(imagen):
    imagen = imagen.resize((640, 640))
    img = np.array(imagen).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(img, axis=0)

# --- Dibujar resultados y extraer clases ---
def procesar_detecciones(imagen, detecciones, clases, umbral=0.3):
    img_np = np.array(imagen).copy()
    h, w, _ = img_np.shape
    clases_detectadas = set()

    for d in detecciones:
        if len(d) != 6:
            continue
        x, y, ancho, alto, conf, clase_id = d
        if conf < umbral:
            continue

        x1 = int((x - ancho / 2) * w)
        y1 = int((y - alto / 2) * h)
        x2 = int((x + ancho / 2) * w)
        y2 = int((y + alto / 2) * h)
        clase_id = int(clase_id)
        nombre_clase = clases[clase_id] if clase_id < len(clases) else f"ID {clase_id}"
        clases_detectadas.add(nombre_clase)

        # Dibujar
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_np, f"{nombre_clase} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img_np, sorted(clases_detectadas)

# --- Interfaz de usuario ---
st.title("🦺 Detector de Implementos de Seguridad")
st.write("Detecta elementos como casco, chaleco, gafas, botas, etc., según lo entrenado en tu modelo YOLOv8.")

confianza = st.slider("🔍 Umbral mínimo de confianza", 0.0, 1.0, 0.3, 0.05)

# Entrada de imagen
img_input = st.camera_input("📸 Captura una imagen") or \
            st.file_uploader("📂 O carga una imagen", type=["jpg", "jpeg", "png"])

if not img_input:
    url = st.text_input("🌐 O pega el enlace a una imagen")
    if url:
        try:
            response = requests.get(url)
            img_input = BytesIO(response.content)
        except:
            st.error("❌ No se pudo cargar la imagen desde el enlace.")

# Procesamiento
if img_input:
    try:
        imagen = Image.open(img_input).convert("RGB")
        entrada = preprocesar_imagen(imagen)
        salida = session.run(None, {input_name: entrada})[0]

        imagen_resultado, etiquetas = procesar_detecciones(imagen, salida[0], CLASES, umbral=confianza)

        st.image(imagen_resultado, caption="🧠 Resultado de detección", use_container_width=True)

        if etiquetas:
            st.success("🛡️ Objetos detectados:")
            for clase in etiquetas:
                st.write(f"✔️ {clase}")
        else:
            st.warning("⚠️ No se detectaron objetos con el umbral seleccionado.")

    except Exception as e:
        st.error(f"❌ Error durante el procesamiento: {e}")
else:
    st.info("Sube una imagen, usa la cámara o pega un enlace para comenzar.")


