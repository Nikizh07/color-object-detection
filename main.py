from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import math
from sklearn.cluster import KMeans

# Constants
MODEL_PATH = "yolov8m.pt"
VIDEO_SOURCE = "http://10.10.172.86:4747/video"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "shoe", "hand", "face", "eye", "ear", "nose",
    "mouth", "head", "neck", "shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot", "pen", "calculator" , "fan"
]

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

def get_dominant_color(image, k=3):
    """Get the dominant color of an image using KMeans clustering."""
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
    dominant = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant

def classify_color(rgb):
    """Classify a color based on its RGB values."""
    r, g, b = rgb[2], rgb[1], rgb[0]  # Convert BGR to RGB
    if r > 150 and g < 100 and b < 100:
        return "red"
    elif b > 150 and g < 100 and r < 100:
        return "blue"
    elif g > 150 and r < 100 and b < 100:
        return "green"
    elif r > 180 and g > 180 and b < 100:
        return "yellow"
    elif r > 200 and g > 200 and b > 200:
        return "white"
    elif r < 80 and g < 80 and b < 80:
        return "black"
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
