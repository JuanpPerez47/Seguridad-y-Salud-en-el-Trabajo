import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# Cargar clases desde archivo
with open("clasesSST.txt", "r") as f:
    CLASES = [line.strip() for line in f.readlines()]

# Cargar modelo ONNX
onnx_model_path = "yolov8n.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, h, w]
input_height, input_width = input_shape[2], input_shape[3]

# Preprocesar imagen
def preprocess_image(image):
    image_resized = image.resize((input_width, input_height))
    image_array = np.array(image_resized).astype(np.float32) / 255.0  # Normalizar
    image_array = np.transpose(image_array, (2, 0, 1))  # HWC -> CHW
    image_array = np.expand_dims(image_array, axis=0)  # [1, 3, h, w]
    return image_array

# Postprocesamiento
def postprocess_output(output, orig_image, conf_threshold=0.3):
    image_width, image_height = orig_image.size
    detections = output[0][0]  # [N, 6]
    boxes, class_ids, scores = [], [], []

    for det in detections:
        confidence = det[4]
        if confidence > conf_threshold:
            x_center, y_center, width, height = det[0], det[1], det[2], det[3]
            class_id = int(det[5])

            # Coordenadas absolutas
            left = int((x_center - width / 2) * image_width)
            top = int((y_center - height / 2) * image_height)
            right = int((x_center + width / 2) * image_width)
            bottom = int((y_center + height / 2) * image_height)

            boxes.append((left, top, right, bottom))
            class_ids.append(class_id)
            scores.append(confidence)

    return boxes, class_ids, scores

# Obtener nombre seguro de clase
def get_class_name(class_id):
    if 0 <= class_id < len(CLASES):
        return CLASES[class_id]
    else:
        return f"Clase desconocida (id {class_id})"

# Dibujar detecciones en la imagen
def draw_detections(image, boxes, class_ids, scores):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, cls_id, score in zip(boxes, class_ids, scores):
        label = get_class_name(cls_id)
        label_text = f"{label} ({score:.2f})"

        # Calcular el tamaño del texto usando textbbox
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Dibujar el rectángulo de la caja
        draw.rectangle(box, outline="red", width=3)

        # Dibujar fondo para el texto
        text_background = [
            box[0],
            box[1] - text_height - 4,
            box[0] + text_width + 4,
            box[1]
        ]
        draw.rectangle(text_background, fill="red")

        # Dibujar texto
        draw.text((box[0] + 2, box[1] - text_height - 2), label_text, fill="white", font=font)

    return image


# Interfaz Streamlit
st.title("🦺 Detección de Seguridad con YOLOv8 (ONNX)")
uploaded_file = st.file_uploader("📷 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Imagen Original", use_column_width=True)

    # Inferencia
    input_tensor = preprocess_image(image)
    output = session.run(None, {input_name: input_tensor})
    boxes, class_ids, scores = postprocess_output(output, image.copy())

    # Dibujar resultados en la imagen
    image_with_boxes = image.copy()
    image_with_boxes = draw_detections(image_with_boxes, boxes, class_ids, scores)
    st.image(image_with_boxes, caption="🟥 Imagen con Detecciones", use_column_width=True)

    # Mostrar etiquetas detectadas
    st.markdown("### ✅ Objetos detectados:")
    if boxes:
        for cls_id in set(class_ids):
            st.write(f"- {get_class_name(cls_id)}")
    else:
        st.write("⚠️ No se detectaron elementos con suficiente confianza.")
