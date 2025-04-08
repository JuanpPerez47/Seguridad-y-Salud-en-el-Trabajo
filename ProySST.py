import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort

# Cargar clases desde archivo
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Cargar modelo
@st.cache_resource
def load_model():
    return ort.InferenceSession("yolov8n.onnx")

# Preprocesamiento para YOLOv8
def preprocess(image, size=640):
    img_resized = cv2.resize(image, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    return img_input, img_resized

# Postprocesamiento
def postprocess(preds, shape, conf_thres=0.3, iou_thres=0.5):
    detections = []
    boxes, scores, class_ids = [], [], []

    for det in preds[0]:
        conf = det[4]
        if conf > conf_thres:
            cls_scores = det[5:]
            cls_id = np.argmax(cls_scores)
            score = cls_scores[cls_id] * conf
            if score > conf_thres:
                x_c, y_c, w, h = det[0:4]
                x1 = int((x_c - w/2) * shape[1])
                y1 = int((y_c - h/2) * shape[0])
                x2 = int((x_c + w/2) * shape[1])
                y2 = int((y_c + h/2) * shape[0])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        for i in indices.flatten():
            if class_ids[i] < len(CLASSES):
                detections.append((boxes[i], class_ids[i], scores[i]))
    return detections

# Dibujar cajas
def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return image

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Detección de EPP con YOLOv8", layout="centered")

st.title("🦺 Detección de Equipos de Protección Personal")
st.write("Este sistema detecta elementos de seguridad como **casco**, **chaleco**, **guantes**, **botas** y **personas**.")

uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Imagen original", use_container_width=True)

    model = load_model()
    input_tensor, resized_image = preprocess(image)
    outputs = model.run(None, {"images": input_tensor})
    detections = postprocess(outputs, resized_image.shape)

    if detections:
        result_img = draw_boxes(resized_image.copy(), detections)
        st.image(result_img, caption="🟩 Detecciones", use_container_width=True)
    else:
        st.info("No se detectaron objetos.")

