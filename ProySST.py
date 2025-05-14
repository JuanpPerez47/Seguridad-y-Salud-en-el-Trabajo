import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import requests
from io import BytesIO
from ultralytics import YOLO
from gtts import gTTS

# ✅ Configuración de la página (debe ser la primera instrucción de Streamlit)
st.set_page_config(page_title="Sistema de Reconocimiento de Objetos", layout="wide")

# Cargar modelo
modelo_objetos = YOLO("best.pt")  # Reemplaza con tu modelo personalizado si es necesario
st.write("Clases del modelo:", modelo_objetos.names)

# Encabezado
st.image("Vogue Editors.jpeg", width=1200)
st.markdown(
    "<h2 style='text-align: center; color: #003366;'>Sistema de Detección de Objetos en Laboratorio</h2>",
    unsafe_allow_html=True,
)

# Barra lateral con controles
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen/video")
    st.subheader("Detección de objetos con Yolov8")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100

# Entradas
archivo_imagen = st.file_uploader("📁 Subir imagen", type=["jpg", "jpeg", "png"])
archivo_video = st.file_uploader("🎞️ Subir video", type=["mp4", "mov", "avi"])
captura = st.camera_input("📷 Capturar imagen desde cámara")
url = st.text_input("🌐 Ingresar URL de imagen")

# Procesar imagen
procesar = st.button("📤 Procesar")

if procesar:
    imagen_original = None
    if archivo_imagen:
        imagen_original = Image.open(archivo_imagen)
    elif captura:
        imagen_original = Image.open(captura)
    elif url:
        try:
            response = requests.get(url)
            imagen_original = Image.open(BytesIO(response.content))
        except:
            st.error("❌ No se pudo cargar la imagen desde la URL.")
    
    if imagen_original:
        st.subheader("📸 Imagen analizada")
        st.image(imagen_original, use_container_width=True)

        img_cv = np.array(imagen_original)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        resultados = modelo_objetos(img_cv, conf=confianza)[0]
        etiquetas_detectadas = [modelo_objetos.names[int(d.cls)] for d in resultados.boxes]

        for box in resultados.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = modelo_objetos.names[int(box.cls[0])]
            conf = float(box.conf[0])
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_cv, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(img_cv, caption="Resultado de detección", channels="BGR")

        # Texto a voz
        texto = "Objetos detectados: " + ", ".join(etiquetas_detectadas)
        tts = gTTS(text=texto, lang='es')
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        st.audio(temp_audio.name, format="audio/mp3")

    elif archivo_video:
        st.subheader("🎞️ Procesamiento de video")
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video_path.write(archivo_video.read())

        cap = cv2.VideoCapture(temp_video_path.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resultados = modelo_objetos(frame, conf=confianza)[0]
            for box in resultados.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = modelo_objetos.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            stframe.image(frame, channels="BGR")

        cap.release()

# Pie de página
st.markdown("---")
st.markdown("<center><sub>📌 Autor: Juan Pablo Pérez Bayona - UNAB 2025 ©️</sub></center>", unsafe_allow_html=True)

