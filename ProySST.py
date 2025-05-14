import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import requests
from io import BytesIO
from ultralytics import YOLO
from gtts import gTTS

# ESTA LÍNEA DEBE IR PRIMERO
st.set_page_config(page_title="Sistema de Reconocimiento de Objetos", layout="wide")

# Cargar modelos
modelo_objetos = YOLO("best.pt")
st.write("Clases del modelo:", modelo_objetos.names)

# Encabezado principal con tamaño ajustado
st.image("Vogue Editors.jpeg", width=1200)
st.markdown(
    "<h2 style='text-align: center; color: #003366;'>Sistema de Detección de Elementos de Protección Personal</h2>",
    unsafe_allow_html=True,
)

# Barra lateral con controles
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de imagen")
    st.subheader("Detección de objetos de seguridad en el trabajo con Yolov8")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50) / 100  # Normalizado 0 a 1

# Entradas de imagen
archivo = st.file_uploader("📁 Subir desde archivo", type=["jpg", "jpeg", "png"])
captura = st.camera_input("📷 Capturar desde cámara")
url = st.text_input("🌐 Ingresar URL de imagen")

# Botón de procesamiento
procesar = st.button("📤 Procesar imagen")

# Procesar entrada
imagen_original = None
if procesar:
    if archivo:
        imagen_original = Image.open(archivo)
    elif captura:
        imagen_original = Image.open(captura)
    elif url:
        try:
            response = requests.get(url)
            imagen_original = Image.open(BytesIO(response.content))
        except:
            st.error("❌ No se pudo cargar la imagen desde el enlace.")
    else:
        st.warning("⚠️ Por favor, selecciona una imagen antes de procesar.")

# Procesamiento de la imagen
if imagen_original:
    st.subheader("📸 Imagen analizada")
    st.image(imagen_original, use_container_width=True)

    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    resultados_personas = modelo_personas(img_cv, conf=confianza)[0]
    personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]
    st.success(f"👥 Personas detectadas: {len(personas_detectadas)}")

    for i, persona in enumerate(personas_detectadas, start=1):
        x1, y1, x2, y2, _, _ = map(int, persona[:6])
        persona_img = img_cv[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, persona_img)
            resultados_ppe = modelo_ppe(temp_file.name)[0]
            etiquetas_detectadas = [modelo_ppe.names[int(d.cls)] for d in resultados_ppe.boxes]

            for box in resultados_ppe.boxes:
                x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
                label = modelo_ppe.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(persona_img, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                cv2.putText(persona_img, f"{label} {conf:.2f}", (x1o, y1o - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            st.markdown(f"### 👤 Persona {i}")
            st.image(persona_img, caption="Detección de PPE", channels="BGR", width=350)
            st.markdown("🔍 **Elementos detectados:** " + ", ".join(etiquetas_detectadas))

            requeridos = {"casco", "chaleco", "botas"}
            presentes = set(etiquetas_detectadas)

            if requeridos.issubset(presentes):
                st.success("✅ Cumple con los requisitos para el ingreso 🏭")
            else:
                faltantes = requeridos - presentes
                st.error(f"🚨 No cumple con el PPE. Faltan: {', '.join(faltantes)}")

            # 🔊 Texto a voz con gTTS
            texto_prediccion = f"La persona {i} tiene los siguientes elementos: {', '.join(etiquetas_detectadas)}. "
            if requeridos.issubset(presentes):
                texto_prediccion += "Cumple con los requisitos de protección."
            else:
                texto_prediccion += f"No cumple con el equipo requerido. Faltan: {', '.join(faltantes)}."

            tts = gTTS(text=texto_prediccion, lang='es')
            temp_audio = tempfile.NamedTemporaryFile(delete=True, suffix=".mp3")
            tts.save(temp_audio.name)

            st.audio(temp_audio.name, format="audio/mp3")

# Pie de página
st.markdown("---")
st.markdown("<center><sub>📌 Autor: Juan Pablo Pérez Bayona - UNAB 2025 ©️</sub></center>", unsafe_allow_html=True)

