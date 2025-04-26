from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import sys
import streamlit as st
import logging # Added for better error reporting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
MODEL_PATH = "yolo11n.pt" # Using YOLOv11 model path
FRAME_WIDTH = 640 # Example value, might not be strictly enforced here
FRAME_HEIGHT = 480 # Example value, might not be strictly enforced here
KMEANS_CLUSTERS = 3 # Number of clusters for dominant color analysis
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence for detecting an object
# --- End Configuration ---


@st.cache_resource # Cache the loaded model for efficiency
def load_yolo_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        logging.info(f"Attempting to load model: {model_path}...")
        model = YOLO(model_path)
        # Perform a dummy inference to check compatibility fully (optional but recommended)
        # _ = model(np.zeros((640, 640, 3)), verbose=False)
        logging.info(f"Model '{model_path}' loaded successfully.")
        class_names = model.names
        logging.info(f"Model classes: {list(class_names.values())}")
        # Check if it looks like a detection model
        if not hasattr(model, 'predict') or not class_names:
             logging.error(f"Model loaded from {model_path} might not be a detection model or is missing class names.")
             st.error(f"Failed to initialize detection model from {model_path}. Check model compatibility.")
             sys.exit(1)
        return model, class_names
    except FileNotFoundError:
        logging.error(f"Error: Model file not found at {model_path}. Please ensure the file exists.")
        st.error(f"Fatal Error: Model file not found at {model_path}. Download or place the file correctly.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading YOLO model from {model_path}: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not load YOLO model from {model_path}. Check compatibility and file integrity. Error: {e}")
        sys.exit(1)


def get_dominant_color(image, k=KMEANS_CLUSTERS):
    """Finds the dominant color in an image using K-Means clustering."""
    if image is None or image.size == 0:
        logging.warning("get_dominant_color received empty image.")
        return np.array([0, 0, 0]) # Return black for empty image

    # Ensure image is BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] != 3:
        logging.warning(f"get_dominant_color received image with unexpected shape: {image.shape}")
        return np.array([0, 0, 0]) # Return black for unexpected format

    try:
        # Reshape for K-Means (handle potential empty dimension)
        pixels = image.reshape((-1, 3))
        if pixels.shape[0] == 0:
            logging.warning("Image became empty after reshape in get_dominant_color.")
            return np.array([0,0,0])

        pixels = np.float32(pixels)

        # Perform K-Means
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(pixels) # Use 'auto' for n_init in newer sklearn
        counts = np.bincount(kmeans.labels_)

        if not counts.size > 0:
             logging.warning("KMeans resulted in empty counts.")
             return np.array([0, 0, 0])

        # Get the most frequent cluster center
        dominant_bgr = kmeans.cluster_centers_[np.argmax(counts)]
        return np.uint8(dominant_bgr)

    except cv2.error as cv_err:
         logging.error(f"OpenCV error in get_dominant_color: {cv_err}", exc_info=True)
         return np.array([0, 0, 0])
    except Exception as e:
         logging.error(f"Error in K-Means clustering for dominant color: {e}", exc_info=True)
         return np.array([0, 0, 0]) # Return black on error


def classify_color(bgr_color):
    """Classifies a BGR color into a common color name using HSV thresholds."""
    if not isinstance(bgr_color, (np.ndarray, list, tuple)) or len(bgr_color) != 3:
        logging.warning(f"Invalid input to classify_color: {bgr_color}")
        return "unknown"

    pixel = np.uint8([[bgr_color]]) # Create 1x1 pixel image
    try:
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
    except Exception as e:
        logging.error(f"Error converting BGR to HSV in classify_color: {e}")
        return "unknown"

    # Color thresholds based on HSV values
    if v < 60: return "black"
    if s < 50:
        return "white" if v > 190 else "gray"
    # Chromatic colors
    if h < 10 or h >= 170: return "red"        # Includes wrap-around hue for red
    elif 10 <= h < 25: return "orange"       # Adjusted threshold
    elif 25 <= h < 35: return "yellow"
    elif 35 <= h < 85: return "green"
    elif 85 <= h < 100: return "cyan"        # Adjusted threshold
    elif 100 <= h < 135: return "blue"
    elif 135 <= h < 160: return "purple"      # Adjusted threshold
    elif 160 <= h < 170: return "pink"        # Adjusted threshold
    else: return "unknown" # Should not happen with valid H range (0-179)


def analyze_image(image, model, class_names_map):
    """Analyzes an image using the YOLO model to detect objects and their colors."""
    if image is None:
        logging.error("analyze_image received None image.")
        return None, [], defaultdict(int)

    individual_detections = []
    object_counts = defaultdict(int)
    annotated_image = image.copy() # Work on a copy

    logging.info(f"Starting analysis with confidence threshold: {CONFIDENCE_THRESHOLD}")
    detection_count_total = 0
    detection_count_threshold = 0

    try:
        # Perform inference
        results = model(image, stream=True, conf=CONFIDENCE_THRESHOLD, verbose=False) # Set verbose=False for cleaner logs

        # Process results stream
        for r in results:
            boxes = r.boxes
            detection_count_total += len(boxes) # Count all potential boxes before filtering by conf (though filtering happens in model call)

            for box in boxes: # These boxes should already meet the confidence threshold
                detection_count_threshold += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                conf = float(box.conf[0])            # Confidence score
                cls = int(box.cls[0])                # Class ID
                class_name = class_names_map.get(cls, f"Unknown Class ({cls})") # Get class name

                # Define padding for cropping
                pad = 5
                crop_y1 = max(0, y1 - pad)
                crop_y2 = min(annotated_image.shape[0], y2 + pad)
                crop_x1 = max(0, x1 - pad)
                crop_x2 = min(annotated_image.shape[1], x2 + pad)

                # Crop the detected object
                cropped_object = image[crop_y1:crop_y2, crop_x1:crop_x2]

                if cropped_object.size == 0:
                    logging.warning(f"Skipping empty crop for {class_name} at [{x1},{y1},{x2},{y2}]")
                    continue # Skip if crop is empty

                # Get dominant color and classify it
                dominant_bgr = get_dominant_color(cropped_object)
                color_name = classify_color(dominant_bgr)

                # Store detection info
                detection_info = {
                    "Color": color_name.capitalize(),
                    "Object": class_name.capitalize(),
                    "Confidence": f"{conf:.2f}"
                }
                individual_detections.append(detection_info)

                # Update counts
                object_counts[(color_name, class_name)] += 1

                # Draw bounding box on the annotated image using the dominant color
                box_color_bgr = (int(dominant_bgr[0]), int(dominant_bgr[1]), int(dominant_bgr[2]))
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color_bgr, 2)

                # Add label (optional)
                label = f"{color_name.capitalize()} {class_name.capitalize()} ({conf:.2f})"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10 # Position label above or below box
                cv2.rectangle(annotated_image, (x1, label_y - h - 5) , (x1 + w + 5, label_y), box_color_bgr, -1) # Background
                cv2.putText(annotated_image, label, (x1 + 5, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA) # White text

        logging.info(f"Analysis complete. Found {detection_count_threshold} objects meeting threshold (out of {detection_count_total} potentials).")
        if detection_count_threshold == 0:
            logging.info("No objects detected meeting the confidence threshold.")

        return annotated_image, individual_detections, object_counts

    except Exception as e:
        logging.error(f"Error during image analysis: {e}", exc_info=True)
        # Return original image and empty results on failure
        return image, [], defaultdict(int)
