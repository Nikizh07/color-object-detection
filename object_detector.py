# object_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import sys
import streamlit as st


MODEL_PATH = "yolov8m.pt"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KMEANS_CLUSTERS = 3
CONFIDENCE_THRESHOLD = 0.4


@st.cache_resource
def load_yolo_model(model_path):
    try:
        print(f"Loading model: {model_path}...")
        model = YOLO(model_path)
        print("Model loaded successfully.")
        class_names = model.names
        print(f"Model classes: {list(class_names.values())}")
        return model, class_names
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        sys.exit(1)


def get_dominant_color(image, k=KMEANS_CLUSTERS):
    if image is None or image.size == 0: return np.array([0, 0, 0])
    if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] != 3: return np.array([0, 0, 0])
    try:
        pixels = image.reshape((-1, 3))
    except ValueError: return np.array([0, 0, 0])
    pixels = np.float32(pixels)
    try:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(pixels)
        counts = np.bincount(kmeans.labels_)
        if not counts.size > 0: return np.array([0, 0, 0])
        dominant_bgr = kmeans.cluster_centers_[np.argmax(counts)]
        return np.uint8(dominant_bgr)
    except Exception: return np.array([0, 0, 0])


def classify_color(bgr_color):
    if not isinstance(bgr_color, (np.ndarray, list, tuple)) or len(bgr_color) != 3: return "unknown"
    pixel = np.uint8([[bgr_color]])
    try:
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
    except Exception: return "unknown"
    if v < 60: return "black"
    if s < 50: return "white" if v > 190 else "gray"
    if h < 10 or h >= 170: return "red"
    elif 10 <= h < 25: return "orange" if s > 100 and v > 100 else "brown"
    elif 25 <= h < 35: return "yellow"
    elif 35 <= h < 85: return "green"
    elif 85 <= h < 100: return "cyan"
    elif 100 <= h < 135: return "blue"
    elif 135 <= h < 160: return "purple"
    elif 160 <= h < 170: return "pink"
    else: return "unknown"


# Removed capture_frame_from_stream as it's URL specific

def analyze_image(image, model, class_names_map):
    if image is None:
        print("Error: No image provided for analysis.")
        return None, [], defaultdict(int)

    individual_detections = []
    object_counts = defaultdict(int)
    annotated_image = image.copy()

    results = model(image, stream=True, conf=CONFIDENCE_THRESHOLD)
    detection_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            detection_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = class_names_map.get(cls, "Unknown Class")

            pad = 5
            crop_y1 = max(0, y1 - pad)
            crop_y2 = min(annotated_image.shape[0], y2 + pad)
            crop_x1 = max(0, x1 - pad)
            crop_x2 = min(annotated_image.shape[1], x2 + pad)
            cropped_object = image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped_object.size == 0:
                continue

            dominant_bgr = get_dominant_color(cropped_object)
            color_name = classify_color(dominant_bgr)

            detection_info = {
                "Color": color_name.capitalize(),
                "Object": class_name.capitalize(),
                "Confidence": f"{conf:.2f}"
            }
            individual_detections.append(detection_info)

            object_counts[(color_name, class_name)] += 1

            box_color_bgr = (int(dominant_bgr[0]), int(dominant_bgr[1]), int(dominant_bgr[2]))
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color_bgr, 2)

    print(f"Analysis complete. Processed {detection_count} potential objects.")
    if detection_count == 0:
        print("No objects detected meeting the confidence threshold.")

    return annotated_image, individual_detections, object_counts
