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
    Detects and tracks faces in an image using YOLOv12 Face Detection and ByteTrack.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A list of bounding boxes for the detected faces.
        Format: [x_min, y_min, x_max, y_max, track_id]
    """
    # Run tracking
    results = model.track(image, conf=min_confidence, tracker="bytetrack.yaml", persist=True)

    tracked_boxes = []
    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            track_id = int(box.id[0])
            tracked_boxes.append([x_min, y_min, x_max, y_max, track_id])

    return tracked_boxes
