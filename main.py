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
    bgr = np.uint8([[rgb]])  # Convert to OpenCV format
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if v < 50:
        return "black"
    if s < 50:
        if v > 200:
            return "white"
        else:
            return "gray"

    if h < 15 or h > 160:
        return "red"
    elif 15 <= h < 35:
        return "orange"
    elif 35 <= h < 85:
        return "yellow"
    elif 85 <= h < 170:
        return "green"
    elif 170 <= h < 200:
        return "cyan"
    elif 200 <= h < 260:
        return "blue"
    elif 260 <= h < 300:
        return "purple"
    elif 300 <= h <= 345:
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

            # Update count
            object_counts[(color_name, class_name)] += 1

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{color_name} {class_name}"
            cv2.putText(image, "", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image, object_counts

def main():
    # Capture image
    captured_image = capture_image(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT)

    # Analyze image
    labeled_image, object_counts = analyze_image(captured_image, model, model.names)

    # Show final labeled image
    cv2.imshow("Detected Image", labeled_image)
    # cv2.imshow("Detected Image", captured_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print summary
    print("\nFinal Results:")
    for (color, name), count in object_counts.items():
        if color != "unknown":
            print(f"{count} {color} {name}")

if __name__ == "__main__":
    main()
