# Face and Number Plate Blurring

This project is a Python application for detecting, tracking, and blurring faces and number plates in images and videos. It uses the YOLOX model for object detection and the ByteTrack algorithm for object tracking.

## Features

-   **Face Detection:** Detects human faces in images.
-   **Number Plate Detection:** Detects vehicle number plates.
-   **Object Tracking:** Tracks detected objects across video frames.
-   **Efficient Blurring:** Applies a strong, irreversible blur to the detected regions.
-   **Configurable:** Easily configure model paths and detection thresholds.

## Project Structure

The project is organized as follows:

```
.
├── .venv/                  # Python virtual environment
├── data/                   # Input/output data (ignored by Git)
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Utility scripts
├── src/
│   └── blur_app/           # Main application source code
│       ├── __init__.py
│       ├── blurring.py     # Blurring algorithm
│       ├── config.py       # Configuration settings
│       ├── detection.py    # YOLOX detection logic
│       ├── main.py         # Main application entry point
│       └── tracking.py     # ByteTrack tracking logic
├── tests/                  # Unit and integration tests
├── .dockerignore           # Docker ignore file
├── .gitignore              # Git ignore file
├── pyproject.toml          # Project dependencies and metadata
└── README.md               # This file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create and activate the virtual environment:**
    This project uses `uv` to manage the virtual environment.
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    The required dependencies are listed in `pyproject.toml`. Install them using `uv`:
    ```bash
    uv pip install -e .
    ```

## Usage

To run the application, execute the `main.py` script:

```bash
python -m src.blur_app.main
```

The script will process a sample image, detect and track objects, apply blurring, and save the output to the `data/output` directory.

## Configuration

You can configure the application by modifying the `src/blur_app/config.py` file. This file allows you to set the paths to the YOLOX and ByteTrack models, as well as the confidence threshold for object detection.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss any changes.
