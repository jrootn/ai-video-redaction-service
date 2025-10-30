# AI Video Redaction Service (YOLO-Face Detector)

This service uses the **YOLO-Face** model for accurate and efficient face detection in videos. It's a container-ready Python application that processes video files, detects faces, and outputs structured metadata.

## How it Works

The application processes a video file frame by frame, using a pre-trained YOLO-Face model to identify faces. For each detected face, it records the frame number, a unique tracking ID, and the bounding box coordinates.

This implementation includes several YOLOv12 models (`yolov12n-face.pt`, `yolov12s-face.pt`, `yolov12m-face.pt`, and `yolov12l-face.pt`) in the `models` directory. You can easily switch between these models by editing the `src/blur_app/detection.py` file to test their performance.

## Running Locally

To run the `detector-service` locally, follow these steps:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Service:**
    ```bash
    bash run_local.sh
    ```

This will process the sample video in the `tests` directory and output the following to the `output` directory:

*   `metadata.parquet`: A Parquet file containing the structured metadata for all detected faces.
*   `thumbnails/`: A directory containing a thumbnail image for each unique face detected.

You can adjust the sensitivity of the detector by editing the `MIN_DETECTION_CONFIDENCE` environment variable in the `run_local.sh` script.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

## References

*   **YOLO-Face Repository:** [https://github.com/YapaLab/yolo-face](https://github.com/YapaLab/yolo-face)
*   **Pre-trained Models:** The models used in this project were trained on the WIDERFace dataset and are provided by the YOLO-Face repository.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── requirements.txt
├── run_local.sh
├── src
│   └── process.py
└── tests
    └── sample_video.mp4
