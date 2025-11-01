import numpy as np
from boxmot import ByteTrack

# Initialize the tracker
tracker = ByteTrack()

def track_objects(image: np.ndarray, detected_boxes: list) -> list:
    """
    Tracks detected objects using ByteTrack.

    Args:
        image: The input image as a NumPy array.
        detected_boxes: A list of bounding boxes from the detection model.
                        Format: [x_min, y_min, x_max, y_max, confidence, class_id]

    Returns:
        A list of tracked objects with their IDs and bounding boxes.
        Format: [x_min, y_min, x_max, y_max, track_id]
    """
    if not detected_boxes:
        return []

    # Convert detections to the format expected by the tracker
    # [x_min, y_min, x_max, y_max, confidence, class_id]
    detections = np.array(detected_boxes)

    # Update the tracker with the new detections
    tracked_objects = tracker.update(detections, image)

    # Format the output
    if len(tracked_objects) == 0:
        return []
        
    # [x_min, y_min, x_max, y_max, track_id, confidence, class_id, feature]
    tracked_boxes = []
    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, track_id, _, _, _ = obj
        tracked_boxes.append([int(x_min), int(y_min), int(x_max), int(y_max), int(track_id)])

    return tracked_boxes
