import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tempfile
import requests
from io import BytesIO
from ultralytics import YOLO

# Modelos
modelo_personas = YOLO("yolov8n.pt")
modelo_ppe = YOLO("best.pt")

# Configuración de la página
st.set_page_config(page_title="Sistema PPE Inteligente", layout="wide")

# Encabezado principal
st.image("imagen12.jpg", use_container_width=True)
st.markdown(
    "<h2 style='text-align: center; color: #003366;'>Sistema de Detección de Elementos de Protección Personal</h2>",
    unsafe_allow_html=True,
)

# Barra lateral
st.sidebar.markdown("## Opciones de Entrada")

# Confianza del modelo
confianza = st.sidebar.slider("Nivel de confianza", 0, 100, 50)

# Entrada: Subir imagen
st.sidebar.markdown("### 1. Subir imagen desde archivo")
archivo = st.sidebar.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

# Entrada: Cámara
st.sidebar.markdown("### 2. Capturar desde cámara")
captura = st.sidebar.camera_input("")

# Entrada: Enlace URL
st.sidebar.markdown("### 3. Usar URL de imagen")
url = st.sidebar.text_input("Pega el enlace aquí")

# Procesar imagen
imagen_original = None
procesar = st.sidebar.button("📤 Procesar imagen")

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
            st.sidebar.error("❌ No se pudo cargar la imagen desde el enlace.")
    else:
        st.sidebar.warning("Por favor, selecciona una imagen primero.")

# Procesamiento
if imagen_original:
    st.subheader("📸 Imagen analizada")
    st.image(imagen_original, use_container_width=True)

    img_cv = np.array(imagen_original)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

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




