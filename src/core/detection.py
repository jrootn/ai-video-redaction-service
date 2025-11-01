import os
import cv2
import numpy as np
from ultralytics import YOLO

# Get confidence score from environment variable, with a default of 0.5
min_confidence = float(os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5))

# Initialize YOLOv12 model
model = YOLO("models/yolov12n-face.pt")

def detect_objects(image: np.ndarray) -> list:
    """
    Detects faces in an image using YOLOv12 Face Detection.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A list of bounding boxes for the detected faces.
        Format: [x_min, y_min, x_max, y_max, confidence, class_id]
    """
    # Run detection
    results = model(image, conf=min_confidence)

    detected_boxes = []
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            detected_boxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

    return detected_boxes
