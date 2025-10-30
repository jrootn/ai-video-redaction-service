import os
import cv2
import mediapipe as mp
import numpy as np
import json
import pandas as pd
import hashlib

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

    # Initialize MediaPipe FaceLandmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='models/face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=10,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
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

            # Perform face landmarking on the resized frame
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_number)

            if face_landmarker_result.face_landmarks:
                for face_landmarks in face_landmarker_result.face_landmarks:
                    # This is a simplified bounding box calculation
                    x_min = min([lm.x for lm in face_landmarks])
                    y_min = min([lm.y for lm in face_landmarks])
                    x_max = max([lm.x for lm in face_landmarks])
                    y_max = max([lm.y for lm in face_landmarks])

                    # Scale bounding box back to original frame size
                    scaled_x_min = int(x_min * original_width)
                    scaled_y_min = int(y_min * original_height)
                    scaled_x_max = int(x_max * original_width)
                    scaled_y_max = int(y_max * original_height)

                    # Create a simple hash of the landmarks to serve as a tracking ID
                    landmarks_str = "".join([f"{lm.x}{lm.y}{lm.z}" for lm in face_landmarks])
                    tracking_id = hashlib.sha256(landmarks_str.encode()).hexdigest()[:8]


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
