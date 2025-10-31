import os
import cv2
import pandas as pd
from blur_app.detection import detect_objects
from blur_app.tracking import track_objects

def process_video():
    """
    Main function to process a video, detect and track faces, and extract metadata.
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
    all_tracked_objects = []
    saved_tracking_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect objects in the current frame
        detected_boxes = detect_objects(frame)

        # 2. Track the detected objects
        tracked_objects = track_objects(frame, detected_boxes)

        frame_height, frame_width, _ = frame.shape

        for obj in tracked_objects:
            x_min, y_min, x_max, y_max, tracking_id = obj
            
            all_tracked_objects.append({
                "frame_number": frame_number,
                "tracking_id": tracking_id,
                "bbox": [x_min, y_min, x_max, y_max]
            })

            # Save a thumbnail for each new tracked object
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
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames...")

    cap.release()

    if all_tracked_objects:
        df = pd.DataFrame(all_tracked_objects)
        df.to_parquet(output_metadata_path)
        print(f"Metadata saved to {output_metadata_path}")

    print("Video processing complete.")

if __name__ == "__main__":
    process_video()
