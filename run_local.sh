#!/bin/bash
export INPUT_VIDEO_PATH="./tests/sample_video.mp4"
export OUTPUT_METADATA_PATH="./output/metadata.json"
export OUTPUT_THUMBNAIL_DIR="./output/thumbnails/"

# Ensure output directories exist
mkdir -p ./output/thumbnails/

# Run the worker
python3 src/process.py
