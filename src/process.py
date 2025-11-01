import os
import cv2
import pandas as pd
from blur_app.detection import detect_objects
from blur_app.tracking import track_objects

def process_video():
    """
    Main function to process a video, detect and track faces, and extract metadata.
    """
    import numpy as np
    # Get configuration from environment variables
    input_video_path = os.environ.get("INPUT_VIDEO_PATH")
    output_metadata_path = os.environ.get("OUTPUT_METADATA_PATH")
    output_thumbnail_dir = os.environ.get("OUTPUT_THUMBNAIL_DIR")
    output_video_path = os.environ.get("OUTPUT_VIDEO_PATH")

    print(f"Starting video processing for: {input_video_path}")

    if not all([input_video_path, output_metadata_path, output_thumbnail_dir, output_video_path]):
        print("Error: Missing one or more environment variables.")
        return

    os.makedirs(output_thumbnail_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    all_tracked_objects = []
    thumbnail_candidates = {}
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect objects in the current frame
        detected_boxes = detect_objects(frame)

        # 2. Track the detected objects
        tracked_objects = track_objects(frame, detected_boxes)

        for obj in tracked_objects:
            x_min, y_min, x_max, y_max, tracking_id = obj

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tracking_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate center
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Update history
            if tracking_id not in track_history:
                track_history[tracking_id] = []
            track_history[tracking_id].append((center_x, center_y))

            # Draw tracking line
            points = np.array(track_history[tracking_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)
            
            all_tracked_objects.append({
                "frame_number": frame_number,
                "tracking_id": tracking_id,
                "bbox": [x_min, y_min, x_max, y_max]
            })

            # Store thumbnail candidates
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_area = bbox_width * bbox_height

            if tracking_id not in thumbnail_candidates or bbox_area > thumbnail_candidates[tracking_id]['area']:
                thumbnail_candidates[tracking_id] = {
                    'frame': frame.copy(),
                    'bbox': [x_min, y_min, x_max, y_max],
                    'area': bbox_area
                }

        # Write the frame to the output video
        out.write(frame)

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames...")

    cap.release()
    out.release()

    # Save the largest thumbnail for each tracking ID
    for tracking_id, data in thumbnail_candidates.items():
        frame = data['frame']
        x_min, y_min, x_max, y_max = data['bbox']
        
        frame_height, frame_width, _ = frame.shape
        
        # Add a margin to the bounding box
        margin = 0.2
        x_margin = int((x_max - x_min) * margin)
        y_margin = int((y_max - y_min) * margin)

        safe_x_min = max(0, x_min - x_margin)
        safe_y_min = max(0, y_min - y_margin)
        safe_x_max = min(frame_width, x_max + x_margin)
        safe_y_max = min(frame_height, y_max + y_margin)

        if safe_y_max > safe_y_min and safe_x_max > safe_x_min:
            thumbnail = frame[safe_y_min:safe_y_max, safe_x_min:safe_x_max]
            thumbnail_path = os.path.join(output_thumbnail_dir, f"id_{tracking_id}.jpg")
            cv2.imwrite(thumbnail_path, thumbnail)

    if all_tracked_objects:
        df = pd.DataFrame(all_tracked_objects)
        df.to_parquet(output_metadata_path)
        df.to_csv(output_metadata_path.replace('.parquet', '.csv'), index=False)
        print(f"Metadata saved to {output_metadata_path}")

    print("Video processing complete.")

if __name__ == "__main__":
    process_video()
