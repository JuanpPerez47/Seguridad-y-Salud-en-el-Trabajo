import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
from ultralytics import YOLO
import tempfile
import requests
from io import BytesIO

# Cargar modelos
modelo_personas = YOLO("yolov8n.pt")     # Detección de personas
modelo_ppe = YOLO("best.pt")             # Detección de PPE

# Configurar página
st.set_page_config(page_title="Sistema PPE Inteligente", layout="wide")
st.markdown("<style>h1, h2, h3 { color: #003366; } .stButton>button { background-color: #004080; color: white; }</style>", unsafe_allow_html=True)

# Encabezado
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image("logo.jpg", width=90)
with col2:
    st.title("🛡️ Sistema Inteligente de uso de Equipos de Protección Personal (PPE)")

st.markdown("""
Bienvenido al sistema de verificación de uso de **PPE** (casco, chaleco, botas) mediante visión por computadora.  
Ideal para garantizar la seguridad antes de ingresar a entornos industriales.

---  
""")

# Instrucciones
with st.expander("📌 ¿Cómo funciona?"):
    st.markdown("""
    1. Selecciona cómo quieres subir la imagen: **archivo**, **cámara** o **enlace**.  
    2. Haz clic en **Enviar Foto**.  
    3. El sistema detectará personas y verificará el uso correcto del equipo PPE.  
    """)

# Opciones de carga
st.subheader("🎯 Selecciona el método de carga")

opcion = st.radio("Método de carga", ["📁 Subir Imagen", "📷 Cámara", "🔗 URL de imagen"])

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

# Procesamiento
if procesar and imagen_original:
    st.subheader("📸 Imagen analizada")
    st.image(imagen_original, use_container_width=True)

    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Detección de personas
    resultados_personas = modelo_personas(img_cv)[0]
    personas_detectadas = [r for r in resultados_personas.boxes.data.cpu().numpy() if int(r[5]) == 0]

    st.success(f"👥 Personas detectadas: {len(personas_detectadas)}")

    # Evaluación individual
    for i, persona in enumerate(personas_detectadas, start=1):
        x1, y1, x2, y2, conf, clase = map(int, persona[:6])
        persona_img = img_cv[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            cv2.imwrite(temp_file.name, persona_img)

            # Evaluar PPE
            resultados_ppe = modelo_ppe(temp_file.name)[0]
            etiquetas_detectadas = [modelo_ppe.names[int(d.cls)] for d in resultados_ppe.boxes]

            # Dibujar detecciones
            for box in resultados_ppe.boxes:
                x1o, y1o, x2o, y2o = map(int, box.xyxy[0])
                label = modelo_ppe.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(persona_img, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
                cv2.putText(persona_img, f"{label} {conf:.2f}", (x1o, y1o - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Mostrar resultados
            st.markdown(f"### 👤 Persona {i}")
            st.image(persona_img, caption="Detección de PPE", channels="BGR", width=350)
            st.markdown("🔍 **Elementos detectados:** " + ", ".join(etiquetas_detectadas))

            # Verificación de cumplimiento
            requeridos = {"casco", "chaleco", "botas"}
            presentes = set(etiquetas_detectadas)

            if requeridos.issubset(presentes):
                st.success("✅ Cumple con los requisitos para el ingreso 🏭")
            else:
                faltantes = requeridos - presentes
                st.error(f"🚨 No cumple con el PPE. Faltan: {', '.join(faltantes)}")

    st.markdown("---")
    st.markdown("<center><sub>📌 Autor: Alfredo Díaz - UNAB 2025 ©️</sub></center>", unsafe_allow_html=True)
