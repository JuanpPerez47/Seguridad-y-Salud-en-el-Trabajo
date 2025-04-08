import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import warnings
import requests
from io import BytesIO

warnings.filterwarnings("ignore")

# Configurar la página
st.set_page_config(
    page_title="Verificación de Seguridad Industrial",
    page_icon="🦺",
    initial_sidebar_state='expanded'
)

# Estilo CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

# Cargar modelo TFLite
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = cargar_modelo()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Preprocesamiento de imagen
def preprocesar(imagen):
    imagen = imagen.resize((640, 640))
    arr = np.array(imagen).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Dibujar detecciones
def dibujar_detecciones(imagen, salida, umbral=0.3):
    img_array = np.array(imagen)
    h, w, _ = img_array.shape
    detectados = []

    for fila in salida:
        if len(fila) != 6:
            continue

        x, y, ancho, alto, conf, clase_id = fila
        if conf < umbral:
            continue

        x1 = int((x - ancho / 2) * w)
        y1 = int((y - alto / 2) * h)
        x2 = int((x + ancho / 2) * w)
        y2 = int((y + alto / 2) * h)
        clase_id = int(clase_id)

        nombre_clase = CLASES[clase_id] if clase_id < len(CLASES) else f"ID {clase_id}"
        detectados.append(nombre_clase)

        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"{nombre_clase} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img_array, detectados

# Sidebar
with st.sidebar:
    st.image("safety_icon.jpg", use_column_width=True)
    st.title("Detección de Seguridad en el Trabajo")
    st.subheader("Reconocimiento de implementos de protección personal")
    confianza = st.slider("Nivel mínimo de confianza", 0.0, 1.0, 0.3, 0.05)

# Encabezado
st.title("🦺 Verificación de Implementos de Seguridad")
st.write("Esta aplicación identifica elementos de seguridad como casco, chaleco, botas, etc., en una imagen de un trabajador.")
st.markdown("#### Modelo YOLOv8n optimizado con TensorFlow Lite")

# Captura o carga de imagen
st.header("📸 Capture o cargue una imagen")
img_input = st.camera_input("Tome una foto") or \
            st.file_uploader("... o cargue una imagen", type=["jpg", "jpeg", "png"])

# Procesamiento
if img_input:
    try:
        imagen = Image.open(img_input)
        st.image(imagen, caption="📷 Imagen cargada", use_container_width=True)

        entrada = preprocesar(imagen)
        interpreter.set_tensor(input_index, entrada)
        interpreter.invoke()
        salida = interpreter.get_tensor(output_index)[0]

        imagen_detectada, objetos = dibujar_detecciones(imagen, salida, umbral=confianza)

        st.image(imagen_detectada, caption="🧠 Resultados de detección", use_container_width=True)

        if objetos:
            st.success("Implementos detectados:")
            st.write("✔️ " + ", ".join(set(objetos)))
        else:
            st.warning("No se detectaron implementos de seguridad con el nivel de confianza seleccionado.")
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
else:
    st.info("Por favor, capture o cargue una imagen.")


