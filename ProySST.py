import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(page_title="Seguridad en el Trabajo", page_icon="🦺")

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

# Cargar modelo YOLOv8n TFLite
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="yolov8n_float32.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = cargar_modelo()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Preprocesar imagen para el modelo
def preprocesar(imagen):
    imagen = imagen.resize((640, 640))
    img_array = np.array(imagen).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# Dibujar cajas y etiquetas
def dibujar_detecciones(imagen, resultados, umbral=0.3):
    imagen_np = np.array(imagen)
    h, w, _ = imagen_np.shape
    objetos_detectados = []

    for fila in resultados:
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
        objetos_detectados.append(nombre_clase)

        # Dibujo
        color = (0, 255, 0)  # verde
        cv2.rectangle(imagen_np, (x1, y1), (x2, y2), color, 3)
        cv2.putText(imagen_np, f"{nombre_clase} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return imagen_np, objetos_detectados

# Interfaz Streamlit
st.title("🦺 Verificación de Implementos de Seguridad")
st.write("Sube una imagen o usa la cámara para detectar casco, chaleco, botas, etc.")
confianza = st.slider("Nivel mínimo de confianza", 0.0, 1.0, 0.3, 0.05)

# Entrada de imagen
img_input = st.camera_input("📸 Captura una imagen") or \
            st.file_uploader("... o carga una imagen", type=["jpg", "jpeg", "png"])

if img_input:
    try:
        imagen = Image.open(img_input)

        # Inferencia
        entrada = preprocesar(imagen)
        interpreter.set_tensor(input_index, entrada)
        interpreter.invoke()
        salida = interpreter.get_tensor(output_index)[0]

        # Dibujar resultados
        imagen_detectada, objetos = dibujar_detecciones(imagen, salida, umbral=confianza)
        st.image(imagen_detectada, caption="🧠 Resultado con detecciones", channels="BGR", use_container_width=True)

        # Mostrar lista de objetos
        if objetos:
            st.success("Implementos detectados:")
            st.write("✔️ " + ", ".join(set(objetos)))
        else:
            st.warning("No se detectaron implementos con el nivel de confianza seleccionado.")
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
else:
    st.info("Por favor, sube o captura una imagen.")
