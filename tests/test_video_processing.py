import unittest
import os
import cv2
import numpy as np
from src.video_processing.processing import VideoProcessor

class TestVideoProcessing(unittest.TestCase):
    def setUp(self):
        # Create a dummy video file for testing
        self.video_path = "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, 20.0, (640, 480))
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        os.remove(self.video_path)

    def test_video_processor(self):
        processor = VideoProcessor(self.video_path)
        metadata = processor.get_video_metadata()
        
        self.assertEqual(metadata['frame_count'], 10)
        self.assertEqual(metadata['width'], 640)
        self.assertEqual(metadata['height'], 480)
        
        frames = list(processor.get_frames())
        self.assertEqual(len(frames), 10)
        
        processor.release()

if __name__ == '__main__':
    unittest.main()
