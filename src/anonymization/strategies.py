from abc import ABC, abstractmethod
import numpy as np
import cv2

class Anonymizer(ABC):
    @abstractmethod
    def anonymize(self, image: np.ndarray, faces: list) -> np.ndarray:
        pass

class SimpleBlurAnonymizer(Anonymizer):
    def anonymize(self, image: np.ndarray, faces: list) -> np.ndarray:
        for face in faces:
            x1, y1, x2, y2 = face
            face_roi = image[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            image[y1:y2, x1:x2] = blurred_face
        return image
