import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import urllib.request

# ---------------------------
# CONFIGURACIÓN Y CARGA
# ---------------------------

CLASSES = [line.strip() for line in open("clasesSST.txt", "r")]

@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov8n.onnx")

def preprocess(image, size=640):
    img_resized = cv2.resize(image, (size, size))
    img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img_input, axis=0), img_resized

def postprocess(preds, shape, conf_thres=0.3, iou_thres=0.45):
    if len(preds) == 0 or preds[0].ndim != 3:
        return []
    boxes, scores, class_ids, detections = [], [], [], []

    for det in preds[0][0]:
        if det[4] <= conf_thres:
            continue
        cls_scores = det[5:]
        if len(cls_scores) == 0:
            continue
        cls_id = np.argmax(cls_scores)
        score = cls_scores[cls_id] * det[4]
        if score > conf_thres:
            x_c, y_c, w, h = det[:4]
            x1 = int((x_c - w / 2) * shape[1])
            y1 = int((y_c - h / 2) * shape[0])
            x2 = int((x_c + w / 2) * shape[1])
            y2 = int((y_c + h / 2) * shape[0])
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        for i in indices.flatten():
            if 0 <= class_ids[i] < len(CLASSES):
                detections.append((boxes[i], class_ids[i], scores[i]))
    return detections

def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# ---------------------------
# INTERFAZ STREAMLIT
# ---------------------------

st.set_page_config(page_title="Detección EPP - YOLOv8", layout="centered")
st.markdown("<h1 style='text-align:center;'>🦺 Detección de Equipos de Protección Personal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detecta automáticamente <strong>casco, botas, guantes, chaleco y personas</strong> desde una imagen.</p>", unsafe_allow_html=True)
st.markdown("---")

# Columnas para cargar imagen
col1, col2, col3 = st.columns(3)
image = None

with col1:
    st.subheader("📁 Subir Imagen")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

with col2:
    st.subheader("📷 Cámara")
    camera_image = st.camera_input("")
    if camera_image:
        image = cv2.imdecode(np.frombuffer(camera_image.read(), np.uint8), 1)

with col3:
    st.subheader("🌐 URL")
    image_url = st.text_input("", placeholder="Pega una URL de imagen")
    if image_url:
        try:
            resp = urllib.request.urlopen(image_url)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, 1)
        except:
            st.error("No se pudo cargar la imagen desde el enlace.")

# ---------------------------
# PREDICCIÓN
# ---------------------------

if image is not None:
    st.markdown("### 🖼️ Imagen original")
    st.image(image, channels="BGR", use_container_width=True)

    model = load_model()
    input_tensor, resized_img = preprocess(image)
    outputs = model.run(None, {"images": input_tensor})
    detections = postprocess(outputs, resized_img.shape)

    result_img = draw_boxes(resized_img.copy(), detections)
    st.markdown("### 🟩 Imagen con detecciones")
    st.image(result_img, channels="BGR", use_container_width=True)

    # Mostrar tabla con resultados
    if detections:
        st.markdown("### 📋 Clases detectadas:")
        detected_classes = [
            {"Clase": CLASSES[cls_id], "Confianza": f"{score:.2f}"}
            for _, cls_id, score in detections
        ]
        st.table(detected_classes)
    else:
        st.info("✅ No se detectaron objetos con suficiente confianza.")

