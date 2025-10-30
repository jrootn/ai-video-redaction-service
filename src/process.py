import os
import cv2
import pandas as pd
import hashlib
import numpy as np
from blur_app.detection import detect_objects

def non_max_suppression(boxes, scores, threshold):
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes to (x1, y1, x2, y2) format
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

def process_video():
    """
    Main function to process the video, detect faces using a tiling strategy, and extract metadata.
    """
    # Get configuration from environment variables
    input_video_path = os.environ.get("INPUT_VIDEO_PATH")
    output_metadata_path = os.environ.get("OUTPUT_METADATA_PATH")
    output_thumbnail_dir = os.environ.get("OUTPUT_THUMBNAIL_DIR")

    print(f"Starting video processing for: {input_video_path}")

    if not all([input_video_path, output_metadata_path, output_thumbnail_dir]):
        print("Error: Missing one or more environment variables.")
        return

    os.makedirs(output_thumbnail_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return

    frame_number = 0
    all_detections = []
    saved_tracking_ids = set()

    # Tiling parameters
    tile_size = 640
    overlap = 540

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        frame_boxes = []
        frame_scores = []

        # Create overlapping tiles
        for y in range(0, frame_height, tile_size - overlap):
            for x in range(0, frame_width, tile_size - overlap):
                tile = frame[y:min(y + tile_size, frame_height), x:min(x + tile_size, frame_width)]
                
                if tile.shape[0] == 0 or tile.shape[1] == 0:
                    continue

                # Detect objects in the tile
                detected_in_tile = detect_objects(tile)

                for box in detected_in_tile:
                    x_min, y_min, x_max, y_max, confidence, _ = box
                    # Convert tile coordinates to frame coordinates
                    global_x_min = x + x_min
                    global_y_min = y + y_min
                    global_x_max = x + x_max
                    global_y_max = y + y_max
                    
                    frame_boxes.append([global_x_min, global_y_min, global_x_max, global_y_max])
                    frame_scores.append(confidence)

        # Apply Non-Maximum Suppression to the detections for the current frame
        if frame_boxes:
            boxes_np = np.array(frame_boxes)
            scores_np = np.array(frame_scores)
            keep_indices = non_max_suppression(boxes_np, scores_np, threshold=0.4)
            final_boxes = boxes_np[keep_indices]

            for box in final_boxes:
                x_min, y_min, x_max, y_max = box

                box_str = f"{frame_number}{x_min}{y_min}{x_max}{y_max}"
                tracking_id = hashlib.sha256(box_str.encode()).hexdigest()[:8]

                all_detections.append({
                    "frame_number": frame_number,
                    "tracking_id": tracking_id,
                    "bbox": [x_min, y_min, x_max, y_max]
                })

                if tracking_id not in saved_tracking_ids:
                    safe_x_min = max(0, x_min)
                    safe_y_min = max(0, y_min)
                    safe_x_max = min(frame_width, x_max)
                    safe_y_max = min(frame_height, y_max)

                    if safe_y_max > safe_y_min and safe_x_max > safe_x_min:
                        thumbnail = frame[safe_y_min:safe_y_max, safe_x_min:safe_x_max]
                        thumbnail_path = os.path.join(output_thumbnail_dir, f"id_{tracking_id}.jpg")
                        cv2.imwrite(thumbnail_path, thumbnail)
                        saved_tracking_ids.add(tracking_id)

        frame_number += 1

    cap.release()

    if all_detections:
        df = pd.DataFrame(all_detections)
        df.to_parquet(output_metadata_path)
        print(f"Metadata saved to {output_metadata_path}")

    print("Video processing complete.")

if __name__ == "__main__":
    process_video()
