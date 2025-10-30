import os
import cv2
import mediapipe as mp
import numpy as np
import json
import pandas as pd

def process_video():
    """
    Main function to process the video, detect faces, and extract metadata.
    """
    # Get configuration from environment variables
    input_video_path = os.environ.get("INPUT_VIDEO_PATH")
    output_metadata_path = os.environ.get("OUTPUT_METADATA_PATH")
    output_thumbnail_dir = os.environ.get("OUTPUT_THUMBNAIL_DIR")

    print(f"Starting video processing for: {input_video_path}")

    if not all([input_video_path, output_metadata_path, output_thumbnail_dir]):
        print("Error: Missing one or more environment variables.")
        return

    # Ensure output directory for thumbnails exists
    os.makedirs(output_thumbnail_dir, exist_ok=True)

    # Initialize MediaPipe FaceDetector
    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='models/blaze_face_short_range.tflite'),
        running_mode=VisionRunningMode.VIDEO
    )

    with FaceDetector.create_from_options(options) as detector:
        # Open the video file
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_video_path}")
            return

        frame_number = 0
        all_detections = []
        saved_tracking_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize-Detect-Scale Optimization
            original_height, original_width, _ = frame.shape
            resized_frame = cv2.resize(frame, (1280, 720))
            
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform face detection on the resized frame
            face_detector_result = detector.detect_for_video(mp_image, frame_number)

            if face_detector_result.detections:
                for detection in face_detector_result.detections:
                    bbox = detection.bounding_box
                    
                    # Scale bounding box back to original frame size
                    scaled_x_min = int(bbox.origin_x * original_width / 1280)
                    scaled_y_min = int(bbox.origin_y * original_height / 720)
                    scaled_x_max = int((bbox.origin_x + bbox.width) * original_width / 1280)
                    scaled_y_max = int((bbox.origin_y + bbox.height) * original_height / 720)

                    # Get the tracking ID
                    if detection.categories:
                        tracking_id = detection.categories[0].index
                    else:
                        print(f"Warning: No categories found for detection in frame {frame_number}")
                        print(detection)
                        tracking_id = None

                    if tracking_id is not None:
                        all_detections.append({
                            "frame_number": frame_number,
                            "tracking_id": tracking_id,
                            "bbox": [scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max]
                        })

                        # Save thumbnail for new tracking IDs
                        if tracking_id not in saved_tracking_ids:
                            thumbnail = frame[scaled_y_min:scaled_y_max, scaled_x_min:scaled_x_max]
                            thumbnail_path = os.path.join(output_thumbnail_dir, f"id_{tracking_id}.jpg")
                            cv2.imwrite(thumbnail_path, thumbnail)
                            saved_tracking_ids.add(tracking_id)

            frame_number += 1

        cap.release()

        # Save metadata to Parquet file
        if all_detections:
            df = pd.DataFrame(all_detections)
            df.to_parquet(output_metadata_path)
            print(f"Metadata saved to {output_metadata_path}")

    print("Video processing complete.")

if __name__ == "__main__":
    process_video()
