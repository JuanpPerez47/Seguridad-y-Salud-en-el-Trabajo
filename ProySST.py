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

# Configuración de la página
st.set_page_config(page_title="Verificación de Seguridad", page_icon="🦺")

# Ocultar menú y pie de página
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
    except:
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

# Preprocesar imagen
def preprocesar(imagen):
    imagen = imagen.resize((640, 640))
    arr = np.array(imagen).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Dibujar resultados
def dibujar_detecciones(imagen, salida, umbral=0.3):
    img_np = np.array(imagen).copy()
    h, w, _ = img_np.shape
    objetos_detectados = []

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
        nombre = CLASES[clase_id] if clase_id < len(CLASES) else f"ID {clase_id}"
        objetos_detectados.append(nombre)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_np, f"{nombre} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img_np, objetos_detectados

# Título principal
st.title("🦺 Verificación de Implementos de Seguridad")
st.write("Sube una imagen, toma una foto o pega un enlace para detectar implementos como casco, chaleco, botas, etc.")

confianza = st.slider("Nivel mínimo de confianza", 0.0, 1.0, 0.3, 0.05)

# Entrada por cámara o archivo
img_input = st.camera_input("📸 Captura una imagen") or \
            st.file_uploader("... o sube una imagen", type=["jpg", "jpeg", "png"])

# Entrada por URL
if not img_input:
    url = st.text_input("🌐 O pega el enlace de una imagen")
    if url:
        try:
            response = requests.get(url)
            img_input = BytesIO(response.content)
        except:
            st.error("❌ No se pudo cargar la imagen desde el enlace.")

# Procesamiento
if img_input:
    try:
        imagen = Image.open(img_input)

        entrada = preprocesar(imagen)
        interpreter.set_tensor(input_index, entrada)
        interpreter.invoke()
        salida = interpreter.get_tensor(output_index)[0]

        imagen_con_detecciones, objetos = dibujar_detecciones(imagen, salida, umbral=confianza)

        st.image(imagen_con_detecciones, caption="🧠 Resultado de detección", use_container_width=True)

        if objetos:
            st.success("Implementos detectados:")
            st.write("✔️ " + ", ".join(set(objetos)))
        else:
            st.warning("No se detectaron implementos con el nivel de confianza seleccionado.")
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
else:
    st.info("Sube una imagen, usa la cámara o pega un enlace para comenzar.")
