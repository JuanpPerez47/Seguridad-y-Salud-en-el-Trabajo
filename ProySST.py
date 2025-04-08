import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort

# Cargar clases
with open("clasesSST.txt", "r") as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Función para cargar el modelo ONNX
def load_model(model_path):
    return ort.InferenceSession(model_path)

# Preprocesamiento de imagen
def preprocess(image, input_size=640):
    image_resized = cv2.resize(image, (input_size, input_size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_input = image_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    image_input = np.expand_dims(image_input, axis=0)
    return image_input, image_resized

# Postprocesamiento seguro
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
            detections.append((boxes[i], class_ids[i], scores[i]))
    return detections

# Dibujar resultados
def draw_boxes(image, detections):
    for box, cls_id, score in detections:
        x1, y1, x2, y2 = box
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return image

# Streamlit UI
st.title("🦺 Detección de objetos de seguridad (SST) con YOLOv8")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Imagen original", use_container_width=True)

    model = load_model("yolov8n.onnx")
    input_image, resized_img = preprocess(img)
    outputs = model.run(None, {"images": input_image})

    detections = postprocess(outputs, resized_img.shape)
    img_with_boxes = draw_boxes(resized_img.copy(), detections)

    st.image(img_with_boxes, caption="Imagen con detecciones", use_container_width=True)


