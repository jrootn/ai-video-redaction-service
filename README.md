# AI Video Redaction Service

This project is a video redaction service that uses AI to detect and blur faces in videos.

## Detector Service

The `detector-service` is a stateless, container-ready Python application that consumes a video, performs high-performance face tracking, and produces structured metadata.

### Running Locally

To run the `detector-service` locally, use the following command:

```bash
bash run_local.sh
```

This will process the sample video in the `tests` directory and output the metadata and thumbnails to the `output` directory.
