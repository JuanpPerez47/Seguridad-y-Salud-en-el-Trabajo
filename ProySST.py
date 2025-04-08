import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
import urllib.request

# Cargar clases
CLASSES = ["boots", "gloves", "helmet", "human", "vest"]
VALID_CLASS_IDS = list(range(len(CLASSES)))  # [0, 1, 2, 3, 4]

# Cargar modelo ONNX
@st.cache_resource
def load_model(model_path="yolov8n.onnx"):
    return ort.InferenceSession(model_path)

# Preprocesamiento para YOLOv8
def preprocess(image, input_size=640):
    image_resized = cv2.resize(image, (input_size, input_size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    image_input = np.expand_dims(image_input, axis=0)
    return image_input, image_resized

# Postprocesamiento con NMS
def postprocess(outputs, image_shape, conf_thres=0.3, iou_thres=0.45):
    predictions = outputs[0]
    boxes, scores, class_ids = [], [], []

    for pred in predictions[0]:
        conf = pred[4]
        if conf > conf_thres:
            cls_scores = pred[5:]
            cls_id = np.argmax(cls_scores)
            score = cls_scores[cls_id] * conf
            if score > conf_thres:
                x_center, y_center, w, h = pred[:4]
                x1 = int((x_center - w / 2) * image_shape[1])
                y1 = int((y_center - h / 2) * image_shape[0])
                x2 = int((x_center + w / 2) * image_shape[1])
                y2 = int((y_center + h / 2) * image_shape[0])
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))
                class_ids.append(int(cls_id))

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    detections = []

    if len(indices) > 0:
        for idx in indices:
            i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
            if class_ids[i] in VALID_CLASS_IDS:
                detections.append((boxes[i], class_ids[i], scores[i]))
    return detections

# Dibujar cajas en imagen
def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return image

# Interfaz Streamlit
st.title("🦺 Detección de Equipos de Protección Personal (EPP) - YOLOv8")

# Elegir origen de imagen
option = st.radio("Selecciona el origen de la imagen:", ("📁 Subir archivo", "📷 Cámara", "🌐 Desde URL"))

image = None

if option == "📁 Subir archivo":
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

elif option == "📷 Cámara":
    camera_image = st.camera_input("Toma una foto")
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

elif option == "🌐 Desde URL":
    image_url = st.text_input("Pega la URL de la imagen:")
    if image_url:
        try:
            resp = urllib.request.urlopen(image_url)
            image_data = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_data, 1)
        except:
            st.error("❌ No se pudo cargar la imagen desde la URL.")

# Procesar imagen si está disponible
if image is not None:
    st.image(image, caption="Imagen original", use_container_width=True)

    model = load_model()
    input_image, resized_img = preprocess(image)
    outputs = model.run(None, {"images": input_image})

    detections = postprocess(outputs, resized_img.shape)
    img_with_boxes = draw_boxes(resized_img.copy(), detections)

    st.image(img_with_boxes, caption="Imagen con detecciones", use_container_width=True)

    if len(detections) == 0:
        st.info("✅ No se detectaron objetos con suficiente confianza.")
