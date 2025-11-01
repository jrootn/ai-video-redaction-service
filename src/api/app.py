from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from ..video_processing.processing import VideoProcessor
from ..core.detection import detect_objects
from ..core.tracking import track_objects
from ..anonymization.strategies import SimpleBlurAnonymizer
from ..anonymization import blurring

app = Flask(__name__)

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    processor = VideoProcessor(video_path)
    
    unique_faces = {}
    for frame in processor.get_frames():
        detected_boxes = detect_objects(frame)
        tracked_objects = track_objects(frame, detected_boxes)
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            if track_id not in unique_faces:
                face_roi = frame[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.jpg', face_roi)
                unique_faces[track_id] = base64.b64encode(buffer).decode('utf-8')

    processor.release()
    return jsonify({"unique_faces": unique_faces})

@app.route('/blur_faces', methods=['POST'])
def blur_faces():
    data = request.get_json()
    if 'video' not in request.files or 'faces_to_blur' not in data:
        return jsonify({"error": "Missing video file or faces_to_blur"}), 400

    video_file = request.files['video']
    video_path = "temp_video.mp4"
    video_file.save(video_path)

    faces_to_blur = data['faces_to_blur']

    processor = VideoProcessor(video_path)
    metadata = processor.get_video_metadata()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = f"output/blurred_video.mp4"
    out = cv2.VideoWriter(output_path, fourcc, metadata['fps'], (metadata['width'], metadata['height']))

    anonymizer = SimpleBlurAnonymizer()

    for frame in processor.get_frames():
        detected_boxes = detect_objects(frame)
        tracked_objects = track_objects(frame, detected_boxes)
        
        objects_to_blur = [obj for obj in tracked_objects if obj[4] in faces_to_blur]
        
        blurred_frame = blurring.blur_objects(frame, objects_to_blur, anonymizer)
        out.write(blurred_frame)

    processor.release()
    out.release()

    return jsonify({"message": "Video processed successfully", "output_path": output_path})


if __name__ == '__main__':
    app.run(debug=True)
