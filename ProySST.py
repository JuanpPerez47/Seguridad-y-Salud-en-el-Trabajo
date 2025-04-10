import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
import requests
from io import BytesIO

# Modelos
modelo_personas = YOLO("yolov8n.pt")
modelo_ppe = YOLO("best.pt")

# Configurar página
st.set_page_config(page_title="Sistema PPE Inteligente", layout="wide")
st.markdown("<style>h1, h2, h3 { color: #003366; } .stButton>button { background-color: #004080; color: white; } .video-container {margin-bottom: 20px}</style>", unsafe_allow_html=True)

# BANNER
st.image("imagen12.jpg", use_column_width=True)

# TÍTULO PRINCIPAL
st.markdown("<h1>Modelo de Identificación de Equipos de Protección Personal - Smart Regions Center</h1>", unsafe_allow_html=True)
st.markdown("Desarrollo del Proyecto de Ciencia de Datos con Redes Convolucionales - UNAB 2025")

# LAYOUT A 2 COLUMNAS
col1, col2 = st.columns([1, 2])

# COLUMNA IZQUIERDA (YouTube y controles)
with col1:
    st.markdown("### Reconocimiento de imagen")
    
    st.video("https://www.youtube.com/watch?v=dPVfJ6Gv1-0")  # Reemplaza con el link correcto

    st.markdown("#### Identificación de objetos con YOLOv8")
    confianza = st.slider("Seleccione el nivel de confianza", 0, 100, 50)

# COLUMNA DERECHA (detección)
with col2:
    st.markdown("### Captura una foto para identificar el uso de PPE")

    opcion = st.radio("📸 Método de carga de imagen", ["📁 Subir Imagen", "📷 Cámara", "🔗 URL de imagen"])
    imagen_original = None
    procesar = False

    if opcion == "📁 Subir Imagen":
        archivo = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
        if st.button("📤 Enviar Foto"):
            if archivo:
                imagen_original = Image.open(archivo)
                procesar = True
            else:
                st.warning("Por favor, sube una imagen.")

    elif opcion == "📷 Cámara":
        captura = st.camera_input("Toma una foto")
        if st.button("📤 Enviar Foto"):
            if captura:
                imagen_original = Image.open(captura)
                procesar = True
            else:
                st.warning("Por favor, toma una foto.")

    elif opcion == "🔗 URL de imagen":
        url = st.text_input("Pega aquí el enlace a la imagen")
        if st.button("📤 Enviar Foto"):
            if url:
                try:
                    response = requests.get(url)
                    imagen_original = Image.open(BytesIO(response.content))
                    procesar = True
                except:
                    st.error("❌ No se pudo cargar la imagen desde el enlace.")
            else:
                st.warning("Ingresa un enlace válido.")

    # PROCESAMIENTO DE IMAGEN
    if procesar and imagen_original:
        st.subheader("📸 Imagen analizada")
        st.image(imagen_original, use_container_width=True)

        img_cv = np.array(imagen_original)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Detección de personas
        resultados_personas = modelo_personas(img_cv, conf=confianza / 100)[0]
        personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]

        st.success(f"👥 Personas detectadas: {len(personas_detectadas)}")

        for i, persona in enumerate(personas_detectadas, start=1):
            x1, y1, x2, y2, conf, clase = map(int, persona[:6])
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

# Pie de página
st.markdown("---")
st.markdown("<center><sub>📌 Autor: Juan Pablo Pérez Bayona - UNAB 2025 ©️</sub></center>", unsafe_allow_html=True)


