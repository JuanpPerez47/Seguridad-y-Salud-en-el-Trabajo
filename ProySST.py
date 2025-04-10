import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# ------------------------ Leer nombres de clases ------------------------ #
CLASES = []
try:
    with open("clasesSST.txt", "r", encoding="utf-8") as f:
        CLASES = [line.strip() for line in f if line.strip()]
    if not CLASES:
        st.error("❌ El archivo clasesSST.txt está vacío.")
except FileNotFoundError:
    st.error("❌ No se encontró el archivo clasesSST.txt.")

# ------------------------ Cargar modelo ONNX ------------------------ #
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_height, input_width = input_shape[2], input_shape[3]

# ------------------------ Preprocesamiento ------------------------ #
def preprocess_image(image):
    image_resized = image.resize((input_width, input_height))
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# ------------------------ Postprocesamiento ------------------------ #
def postprocess_output(output, orig_image, conf_threshold=0.3):
    image_width, image_height = orig_image.size
    detections = output[0][0]
    boxes, class_ids, scores = [], [], []

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            x_center, y_center, width, height = det[0], det[1], det[2], det[3]
            class_id = int(det[5])
            left = int((x_center - width / 2) * image_width)
            top = int((y_center - height / 2) * image_height)
            right = int((x_center + width / 2) * image_width)
            bottom = int((y_center + height / 2) * image_height)

            boxes.append((left, top, right, bottom))
            class_ids.append(class_id)
            scores.append(confidence)

    return boxes, class_ids, scores

# ------------------------ Obtener nombre de clase ------------------------ #
def get_class_name(class_id):
    if 0 <= class_id < len(CLASES):
        return CLASES[class_id]
    else:
        return f"Clase desconocida (id {class_id})"

# ------------------------ Dibujar detecciones ------------------------ #
def draw_detections(image, boxes, class_ids, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, cls_id, score in zip(boxes, class_ids, scores):
        label = get_class_name(cls_id)
        label_text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle(box, outline="red", width=3)
        draw.rectangle(
            [box[0], box[1] - text_height - 4, box[0] + text_width + 4, box[1]],
            fill="red"
        )
        draw.text((box[0] + 2, box[1] - text_height - 2), label_text, fill="white", font=font)
    return image

# ------------------------ Interfaz principal ------------------------ #

# Barra lateral
with st.sidebar:
    st.video("https://www.youtube.com/watch?v=xxUHCtHnVk8")
    st.title("Reconocimiento de Imagen")
    st.subheader("Detección con YOLOv8")
    confianza = st.slider("🔍 Nivel de confianza", 0, 100, 50) / 100

# Imagen decorativa
st.image("smartregionlab2.jpeg", use_column_width=True)
st.title("🦺 Detección de Seguridad con YOLOv8 (ONNX)")

# Entrada de imagen
entrada = (
    st.file_uploader("📁 Sube una imagen", type=["jpg", "jpeg", "png"])
    or st.camera_input("📷 O toma una foto")
    or st.text_input("🌐 O ingresa la URL de una imagen")
)

# Procesamiento
image = None
if entrada:
    try:
        if isinstance(entrada, str):  # URL
            response = requests.get(entrada)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(entrada).convert("RGB")

        st.image(image, caption="📷 Imagen Original", use_container_width=True)

        input_tensor = preprocess_image(image)
        output = session.run(None, {input_name: input_tensor})
        boxes, class_ids, scores = postprocess_output(output, image.copy(), conf_threshold=confianza)

        image_with_boxes = draw_detections(image.copy(), boxes, class_ids, scores)
        st.image(image_with_boxes, caption="🟥 Imagen con Detecciones", use_container_width=True)

        st.markdown("### ✅ Objetos detectados:")
        if boxes:
            for cls_id in set(class_ids):
                st.write(f"- {get_class_name(cls_id)}")
        else:
            st.warning("⚠️ No se detectaron elementos con suficiente confianza.")

    except Exception as e:
        st.error(f"❌ Error al procesar la imagen: {e}")

