import os
import cv2
import pandas as pd

def visualize_detections():
    """
    Reads video and metadata to produce a video with bounding boxes.
    """
    # Get configuration from environment variables
    input_video_path = os.environ.get("INPUT_VIDEO_PATH")
    output_metadata_path = os.environ.get("OUTPUT_METADATA_PATH")
    output_video_path = "output/video_with_detections.mp4"

    print(f"Starting visualization for: {input_video_path}")

    if not all([input_video_path, output_metadata_path]):
        print("Error: Missing one or more environment variables.")
        return

    # Read metadata
    try:
        df = pd.read_parquet(output_metadata_path)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at: {output_metadata_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_video_path}")
        return

    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for the current frame
        detections = df[df["frame_number"] == frame_number]

        for _, row in detections.iterrows():
            bbox = row["bbox"]
            tracking_id = row["tracking_id"]
            x_min, y_min, x_max, y_max = bbox

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put tracking ID on the bounding box
            cv2.putText(frame, str(tracking_id), (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

    print(f"Visualization complete. Video saved to: {output_video_path}")

if __name__ == "__main__":
    visualize_detections()
