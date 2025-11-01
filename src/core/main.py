import cv2
from ..video_processing.processing import VideoProcessor
from ..anonymization.strategies import SimpleBlurAnonymizer
from . import detection, tracking, blurring, config

def main(video_path: str):
    """
    Main function to run the face and number plate blurring application.
    """
    processor = VideoProcessor(video_path)
    metadata = processor.get_video_metadata()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"{config.OUTPUT_DIR}/blurred_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'], (metadata['width'], metadata['height']))

    anonymizer = SimpleBlurAnonymizer()

    print("Starting the blurring application...")
    for frame in processor.get_frames():
        detected_boxes = detection.detect_objects(frame)
        tracked_objects = tracking.track_objects(frame, detected_boxes)
        blurred_frame = blurring.blur_objects(frame, tracked_objects, anonymizer)
        out.write(blurred_frame)

    processor.release()
    out.release()
    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    # This should be replaced with a proper CLI argument parser
    main("data/sample.mp4")
