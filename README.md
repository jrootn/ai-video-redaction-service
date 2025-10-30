# AI Video Redaction Service

This project is the first part of a larger AI Video Redaction Service. This component, the `detector-service`, is a stateless, container-ready Python application designed to perform high-performance face tracking on video files.

## Project Intent

The primary goal of this service is to analyze a video and produce structured metadata that identifies the location of all faces in every frame. This metadata can then be used by other services to perform actions like blurring or redacting faces.

This service is designed with MLOps best practices in mind, ensuring that it is efficient, scalable, and ready for deployment in a cloud-native environment like Google Kubernetes Engine (GKE).

## How it Works

The application processes a video file frame by frame, using the MediaPipe `FaceDetector` to identify faces. For each detected face, it records the frame number, a unique tracking ID, and the bounding box coordinates.

### "Resize-Detect-Scale" Optimization

To ensure high performance and reduce computational cost, this service implements a "Resize-Detect-Scale" optimization:

1.  **Resize:** Each frame is resized to a smaller, standard resolution (1280x720).
2.  **Detect:** Face detection is performed on the smaller, resized frame.
3.  **Scale:** The resulting bounding box coordinates are scaled back up to match the original frame's resolution.

This approach significantly reduces the processing load, making the service faster and more cost-effective, especially when dealing with high-resolution videos.

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

*   `metadata.json`: A Parquet file containing the structured metadata for all detected faces.
*   `thumbnails/`: A directory containing a thumbnail image for each unique face detected.

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
```

*   **`Dockerfile`**: Defines the container image for the application.
*   **`README.md`**: This file.
*   **`requirements.txt`**: A list of the Python dependencies required to run the service.
*   **`run_local.sh`**: A script to run the service locally.
*   **`src/process.py`**: The main application logic.
*   **`tests/sample_video.mp4`**: A sample video for testing the service.
