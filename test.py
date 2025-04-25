from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import math
from sklearn.cluster import KMeans

# Constants
MODEL_PATH = "yolov8m.pt"
VIDEO_SOURCE = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# COCO_CLASSES = [
#     "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#     "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#     "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#     "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#     "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#     "teddy bear", "hair drier", "toothbrush", "shoe", "hand", "face", "eye", "ear", "nose",
#     "mouth", "head", "neck", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot", "pen", "calculator" , "fan" 
# ]



# Load YOLOv8 model
model = YOLO(MODEL_PATH)

COCO_CLASSES = model.names  # Get class names from the model

def get_dominant_color(image, k=5):
    """Get the dominant color of an image using KMeans clustering with improved filtering."""
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    
    # Skip if image is too small
    if pixels.shape[0] < 10:
        return np.array([0, 0, 0])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
    
    # Get counts of each cluster
    counts = np.bincount(kmeans.labels_)
    
    # Sort clusters by size (largest first)
    sorted_indices = np.argsort(counts)[::-1]
    
    # Get centers of the largest clusters
    centers = kmeans.cluster_centers_[sorted_indices]
    
    # Filter out near-black and near-white colors (often background)
    for center in centers:
        b, g, r = center
        # Skip blacks and whites (common backgrounds)
        if (r > 30 and g > 30 and b > 30) and not (r > 240 and g > 240 and b > 240):
            return center
    
    # If all clusters were filtered, return the largest one
    return kmeans.cluster_centers_[np.argmax(counts)]

def classify_color(bgr):
    """Classify a color based on its BGR values using HSV color space."""
    # Convert from BGR to HSV for better color classification
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    
    # For black, white, and gray - check saturation and value first
    if v < 50:
        return "black"
    if s < 30 and v > 200:
        return "white"
    if s < 30:
        return "gray"
    
    # Color ranges in HSV
    if h < 10 or h > 170:
        return "red"
    elif 10 <= h < 25:
        return "orange"
    elif 25 <= h < 35:
        return "yellow"
    elif 35 <= h < 80:
        return "green"
    elif 80 <= h < 100:
        return "turquoise"
    elif 100 <= h < 130:
        return "blue"
    elif 130 <= h < 155:
        return "purple"
    elif 155 <= h < 170:
        return "pink"
    else:
        return "unknown"

def capture_image(video_source, width, height):
    """Capture an image from the video source."""
    cap = cv2.VideoCapture(video_source)


    print("Press SPACE to capture image or ESC to exit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Press SPACE to Capture", frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE key
            captured_image = frame.copy()
            break
        elif key == 27:  # ESC key
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

def analyze_image(image, model, class_names):
    """Analyze the image using the YOLO model and classify objects."""
    results = model(image, stream=True)
    object_counts = defaultdict(int)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            class_name = class_names[cls]

            # Crop object
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            # Get dominant color
            dominant_color = get_dominant_color(cropped)
            color_name = classify_color(dominant_color)

            # Apply fruit-specific corrections based on color
            if class_name == "orange" and color_name == "yellow":
                class_name = "lemon"  # Reclassify yellow oranges as lemons
            
            # Add more fruit corrections as needed
            # For example:
            # if class_name == "apple" and color_name == "yellow":
            #     class_name = "golden apple"

            # Update count
            object_counts[(color_name, class_name)] += 1

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{color_name} {class_name}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image, object_counts

def main():
    # Capture image
    captured_image = capture_image(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    # Analyze image
    labeled_image, object_counts = analyze_image(captured_image, model, COCO_CLASSES)

    # Show final labeled image
    cv2.imshow("Detected Image", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary
    print("\nFinal Results:")
    for (color, name), count in object_counts.items():
        if color != "unknown":
            print(f"{count} {color} {name}")

if __name__ == "__main__":
    main()
