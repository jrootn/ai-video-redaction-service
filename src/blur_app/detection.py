import os
import cv2
import numpy as np
import mediapipe as mp

# Get confidence score from environment variable, with a default of 0.5
min_confidence = float(os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5))

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_confidence)

def detect_objects(image: np.ndarray) -> list:
    """
    Detects faces in an image using MediaPipe Face Detection.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A list of bounding boxes for the detected faces.
        Format: [x_min, y_min, x_max, y_max, confidence, class_id]
    """
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find faces
    results = face_detection.process(rgb_image)

    detected_boxes = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            x_max = x_min + width
            y_max = y_min + height
            
            confidence = detection.score[0]
            class_id = 0  # Class ID 0 for faces

            detected_boxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

    return detected_boxes
