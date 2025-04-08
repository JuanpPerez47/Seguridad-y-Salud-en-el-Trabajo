import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import urllib.request

# Cargar clases
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Cargar modelo
@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov8n.onnx")

# Preprocesamiento
def preprocess(image, size=640):
    img_resized = cv2.resize(image, (size, size))
    img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # YOLOv8 espera RGB
    img_input = img_input.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, img_resized

# Postprocesamiento con validaciones
def postprocess(preds, shape, conf_thres=0.3, iou_thres=0.5):
    if len(preds) == 0 or preds[0].ndim != 3:
        return []

    detections = []
    boxes, scores, class_ids = [], [], []
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

# Dibujar cajas sin cambiar color original (BGR)
def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Detección EPP - YOLOv8", layout="centered")
st.markdown("<h1 style='text-align: center;'>🦺 Detección de Equipos de Protección Personal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube una imagen, toma una foto o pega un enlace para detectar: <strong>botas, guantes, casco, humanos y chaleco</strong>.</p>", unsafe_allow_html=True)

# Elegir fuente de imagen
source = st.radio("Selecciona fuente de imagen:", ["📁 Subir imagen", "📷 Cámara", "🌐 URL"])

image = None

if source == "📁 Subir imagen":
    file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if file:
        bytes_data = file.read()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), 1)

elif source == "📷 Cámara":
    cam_image = st.camera_input("Toma una foto")
    if cam_image:
        image = cv2.imdecode(np.frombuffer(cam_image.read(), np.uint8), 1)

elif source == "🌐 URL":
    url = st.text_input("Pega el enlace de la imagen:")
    if url:
        try:
            resp = urllib.request.urlopen(url)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(img_array, 1)
        except:
            st.error("❌ No se pudo cargar la imagen desde el enlace.")

# Procesar imagen
if image is not None:
    st.image(image, caption="Imagen original", channels="BGR", use_container_width=True)

    model = load_model()
    input_tensor, resized_img = preprocess(image)
    outputs = model.run(None, {"images": input_tensor})
    detections = postprocess(outputs, resized_img.shape)

    if detections:
        result_img = draw_boxes(resized_img.copy(), detections)
        st.image(result_img, caption="🟩 Resultado con detecciones", channels="BGR", use_container_width=True)
    else:
        st.info("✅ No se detectaron objetos con confianza suficiente.")

