import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import cv2
import urllib.request
from io import BytesIO
from gtts import gTTS
import base64
import onnxruntime as ort
import tempfile
from PIL import Image

# ------------------- CONFIGURACIÓN INICIAL -------------------

st.set_page_config(page_title="Detección de EPP", page_icon="🦺", layout="centered")

# Ocultar menú y footer de Streamlit
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- CARGA DE MODELO Y CLASES -------------------

@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov8n.onnx")

@st.cache_data
def load_classes():
    with open("clasesSST.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
CLASSES = load_classes()

# ------------------- UTILIDADES -------------------

def preprocess(image, size=640):
    img_resized = cv2.resize(image, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_input, axis=0), img_resized

def postprocess(preds, shape, conf_thres=0.3, iou_thres=0.45):
    if len(preds) == 0 or preds[0].ndim != 3:
        return []
    detections = []
    boxes, scores, class_ids = [], [], []
    for det in preds[0][0]:
        if det[4] <= conf_thres:
            continue
        cls_scores = det[5:]
        cls_id = np.argmax(cls_scores)
        score = cls_scores[cls_id] * det[4]
        if score > conf_thres:
            x_c, y_c, w, h = det[:4]
            x1 = int((x_c - w / 2) * shape[1])
            y1 = int((y_c - h / 2) * shape[0])
            x2 = int((x_c + w / 2) * shape[1])
            y2 = int((x_c + h / 2) * shape[0])
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    for i in indices.flatten() if len(indices) > 0 else []:
        if 0 <= class_ids[i] < len(CLASSES):
            detections.append((boxes[i], class_ids[i], scores[i]))
    return detections

def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

def generar_audio(texto):
    tts = gTTS(text=texto or "No se encontró información.", lang='es')
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def reproducir_audio(mp3_fp):
    audio_bytes = mp3_fp.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>
    """, unsafe_allow_html=True)

# ------------------- INTERFAZ -------------------

st.image("https://i.ibb.co/XpF6Scs/smartregionlab2.jpeg", use_column_width=True)
st.title("🧠 Detección Inteligente de EPP")
st.write("Sube una imagen, toma una foto o usa un enlace para detectar elementos de protección personal.")

st.sidebar.title("Opciones")
conf_threshold = st.sidebar.slider("Nivel de confianza", 30, 90, 50, step=1) / 100

# Entrada: Archivo, cámara o URL
col1, col2, col3 = st.columns(3)

with col1:
    file_img = st.file_uploader("📁 Desde archivo", type=["jpg", "jpeg", "png"])
with col2:
    cam_img = st.camera_input("📷 Desde cámara")
with col3:
    url_img = st.text_input("🌐 Desde URL")

img = None

# Cargar imagen desde la opción seleccionada
if file_img:
    img = cv2.imdecode(np.frombuffer(file_img.read(), np.uint8), 1)
elif cam_img:
    img = cv2.imdecode(np.frombuffer(cam_img.read(), np.uint8), 1)
elif url_img:
    try:
        response = urllib.request.urlopen(url_img)
        img = cv2.imdecode(np.frombuffer(response.read(), np.uint8), 1)
    except:
        st.error("❌ No se pudo cargar la imagen desde la URL.")

# ------------------- PREDICCIÓN -------------------

if img is not None:
    st.markdown("### 🖼️ Imagen original")
    st.image(img, channels="BGR", use_column_width=True)

    input_tensor, resized = preprocess(img)
    outputs = model.run(None, {"images": input_tensor})
    detections = postprocess(outputs, resized.shape, conf_thres=conf_threshold)

    result_img = draw_boxes(resized.copy(), detections)

    st.markdown("### ✅ Resultados")
    st.image(result_img, channels="BGR", use_container_width=True)

    if detections:
        clases_detectadas = []
        for _, cls_id, score in detections:
            clases_detectadas.append(f"{CLASSES[cls_id]} ({score:.2f})")
        lista = ", ".join(set(clases_detectadas))
        st.success(f"Se detectaron: {lista}")
        audio = generar_audio(f"Se detectaron: {lista}")
        reproducir_audio(audio)
    else:
        st.info("No se detectaron objetos con suficiente confianza.")
        audio = generar_audio("No se detectaron elementos.")
        reproducir_audio(audio)


