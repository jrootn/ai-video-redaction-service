import unittest
import numpy as np
import cv2
from src.anonymization.strategies import SimpleBlurAnonymizer

class TestAnonymization(unittest.TestCase):
    def test_simple_blur_anonymizer(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = [[10, 10, 50, 50]]
        anonymizer = SimpleBlurAnonymizer()
        anonymized_image = anonymizer.anonymize(image.copy(), faces)
        
        # Check if the face area is blurred
        face_roi = anonymized_image[10:50, 10:50]
        original_roi = image[10:50, 10:50]
        
        self.assertFalse(np.array_equal(face_roi, original_roi))

if __name__ == '__main__':
    unittest.main()
