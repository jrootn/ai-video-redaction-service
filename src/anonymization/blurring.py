import numpy as np
from .strategies import Anonymizer

def blur_objects(image: np.ndarray, tracked_objects: list, anonymizer: Anonymizer) -> np.ndarray:
    """
    Blurs the detected and tracked objects in an image using a specified anonymizer.

    Args:
        image: The input image as a NumPy array.
        tracked_objects: A list of tracked objects with their bounding boxes.
        anonymizer: An instance of a class that implements the Anonymizer interface.

    Returns:
        The image with the specified objects blurred.
    """

    print("Blurring objects in the image...")
    faces_to_blur = [obj[:4] for obj in tracked_objects]
    anonymized_image = anonymizer.anonymize(image.copy(), faces_to_blur)
    return anonymized_image
