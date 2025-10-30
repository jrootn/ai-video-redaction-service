# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final Image
FROM python:3.10-slim

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu

# Copy application code
COPY src/process.py .
COPY src/blur_app /app/blur_app
COPY models /app/models

# Set environment variables for the application
ENV INPUT_VIDEO_PATH="tests/sample_video.mp4"
ENV OUTPUT_METADATA_PATH="output/metadata.parquet"
ENV OUTPUT_THUMBNAIL_DIR="output/thumbnails"

# Set the entrypoint
CMD ["python", "process.py"]
