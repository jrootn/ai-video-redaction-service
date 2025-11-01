# AI Video Redaction Service

This project is an AI-powered video redaction service that automatically detects and anonymizes faces in videos. It provides a flexible and extensible pipeline for video processing, face detection, tracking, and anonymization.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Future Work](#future-work)

## Features

- **Video Processing**: Handles various video formats and extracts metadata.
- **Face Detection and Tracking**: Accurately detects and tracks faces throughout a video.
- **Extensible Anonymization**: Supports multiple anonymization strategies, starting with simple blurring. New strategies like AI avatars can be easily added.
- **Face Selection API**: Allows users to select which faces to anonymize through a simple REST API.

## Architecture

The project is structured into the following modules:

- `src/api`: Contains the Flask application for the face selection API.
- `src/anonymization`: Implements the anonymization strategies (e.g., blurring).
- `src/core`: The core application logic, including detection, tracking, and the main processing pipeline.
- `src/video_processing`: Handles video input and frame extraction.
- `tests`: Contains unit tests for the different modules.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` and `virtualenv`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jrootn/ai-video-redaction-service.git
   cd ai-video-redaction-service
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API

To start the face selection API, run the following command:

```bash
./run_api.sh
```

The API will be available at `http://127.0.0.1:5000`.

### API Endpoints

#### 1. Process Video and Get Unique Faces

- **URL**: `/process_video`
- **Method**: `POST`
- **Form Data**:
  - `video`: The video file to process.
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "unique_faces": {
        "1": "base64_encoded_image",
        "2": "base64_encoded_image"
      }
    }
    ```

#### 2. Blur Selected Faces

- **URL**: `/blur_faces`
- **Method**: `POST`
- **Form Data**:
  - `video`: The video file to process.
- **JSON Payload**:
  ```json
  {
    "faces_to_blur": [1, 2]
  }
  ```
- **Success Response**:
  - **Code**: 200
  - **Content**:
    ```json
    {
      "message": "Video processed successfully",
      "output_path": "output/blurred_video.mp4"
    }
    ```

## Testing

To run the unit tests, use the following command:

```bash
./run_tests.sh
```

## Future Work

- **Add more anonymization strategies**:
  - AI-generated avatars
  - Gender-aware avatars
- **Improve face detection and tracking accuracy.**
- **Add a web interface for easier user interaction.**
- **Deploy the service using Docker and Kubernetes.**
