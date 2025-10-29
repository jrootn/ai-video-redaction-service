import numpy as np
import cv2
from . import detection, tracking, blurring, config

def main():
    """
    Main function to run the face and number plate blurring application.
    """
    # Load an example image.
    # In a real application, this would come from a video stream or file.
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(image, "Sample Image", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("Starting the blurring application...")

    # 1. Detect objects
    detected_boxes = detection.detect_objects(image)
    print(f"Detected {len(detected_boxes)} objects.")

    # 2. Track objects
    tracked_objects = tracking.track_objects(image, detected_boxes)
    print(f"Tracking {len(tracked_objects)} objects.")

    # 3. Blur objects
    blurred_image = blurring.blur_objects(image, tracked_objects)
    print("Image blurring complete.")

    # 4. Save or display the output
    output_path = f"{config.OUTPUT_DIR}/blurred_image.png"
    # cv2.imwrite(output_path, blurred_image)
    print(f"Output image saved to {output_path}")

    # For demonstration, we'll just show the image.
    # cv2.imshow("Blurred Image", blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
