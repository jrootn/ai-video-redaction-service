#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create the output directory if it doesn't exist
mkdir -p output

# Install local dependencies
echo "Installing local dependencies..."
pip install --no-cache-dir -r requirements.txt

# Run the processing script locally to generate metadata
echo "Running the processing script locally..."
# You can change the confidence score here (e.g., 0.2 for high sensitivity, 0.5 for balanced)
export MIN_DETECTION_CONFIDENCE="0.5"
INPUT_VIDEO_PATH="tests/sample_video.mp4" \
OUTPUT_METADATA_PATH="output/metadata.parquet" \
OUTPUT_THUMBNAIL_DIR="output/thumbnails" \
OUTPUT_VIDEO_PATH="output/output.mp4" \
python3 src/process.py

echo "Local processing complete."
