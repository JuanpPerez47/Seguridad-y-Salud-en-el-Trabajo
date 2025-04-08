import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import requests
from io import BytesIO

# Configurar la página
st.set_page_config(page_title="Verificación de Seguridad SST", page_icon="🦺")

# Leer clases desde clasesSST.txt
def cargar_clases():
    try:
        with open("clasesSST.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
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
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocesar imagen
def preprocesar(imagen):
    imagen = imagen.resize((640, 640))
    img_array = np.array(imagen).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Dibujar detecciones en imagen
def dibujar_detecciones(imagen, salida, umbral=0.3):
    imagen = np.array(imagen)
    h, w, _ = imagen.shape
    detectados = []

    for fila in salida:
        if len(fila) != 6:
            continue  # saltar filas inválidas

        x_center, y_center, ancho, alto, confianza, clase_id = fila

        if confianza > umbral:
            x1 = int((x_center - ancho / 2) * w)
            y1 = int((y_center - alto / 2) * h)
            x2 = int((x_center + ancho / 2) * w)
            y2 = int((y_center + alto / 2) * h)

            class_id = int(clase_id)
            nombre_clase = CLASES[class_id] if class_id < len(CLASES) else f"ID {class_id}"
            detectados.append(nombre_clase)

            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(imagen, f"{nombre_clase} {confianza:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return imagen, detectados

# Título
st.title("🦺 Verificación de Implementos de Seguridad")
st.write("Detecta si una persona porta casco, chaleco, gafas u otros implementos definidos en `clasesSST.txt`.")

# Entrada de imagen
st.subheader("📷 Fuente de imagen")

img_input = st.camera_input("Captura una imagen") or \
            st.file_uploader("O carga una imagen desde tu equipo", type=["jpg", "png", "jpeg"])

# URL como opción adicional
if not img_input:
    image_url = st.text_input("O pega el enlace a una imagen")
    if image_url:
        try:
            response = requests.get(image_url)
            img_input = BytesIO(response.content)
        except:
            st.error("❌ No se pudo cargar la imagen desde el enlace. Verifica la URL.")

# Procesamiento
if img_input:
    try:
        imagen = Image.open(img_input)
        st.image(imagen, caption="🖼 Imagen cargada", use_container_width=True)

        input_data = preprocesar(imagen)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        salida = interpreter.get_tensor(output_details[0]['index'])[0]  # (N, 6)
        imagen_salida, detectados = dibujar_detecciones(imagen, salida, umbral=0.3)

        st.image(imagen_salida, caption="🔍 Resultados de detección", use_container_width=True)

        if detectados:
            st.success("✅ Implementos detectados:")
            st.write(", ".join(set(detectados)))
        else:
            st.warning("⚠️ No se detectaron implementos de seguridad.")
    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")
else:
    st.info("Por favor, proporciona una imagen desde la cámara, tu equipo o un enlace.")



