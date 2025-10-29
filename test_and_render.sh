#!/bin/bash
# Automated test and render script for DiffSHEG realtime wrapper
# This script runs the test, generates waypoints, and renders a video

set -e  # Exit on error

echo "========================================================================"
echo "DiffSHEG Realtime Wrapper - Test and Render Pipeline"
echo "========================================================================"

# Step 1: Run the realtime wrapper test
echo ""
echo "Step 1: Running realtime wrapper test..."
echo "------------------------------------------------------------------------"
python test_realtime_wrapper.py

# Check if test completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Test failed!"
    exit 1
fi

# Step 2: Check if output files exist
echo ""
echo "Step 2: Verifying output files..."
echo "------------------------------------------------------------------------"

BVH_FILE="results/realtime_test/bvh/realtime_output.bvh"
JSON_FILE="results/realtime_test/face_json/realtime_output.json"
AUDIO_FILE="results/realtime_test/2_scott_0_3_3.wav"
OUTPUT_VIDEO="results/realtime_test/output_video.mp4"

if [ ! -f "$BVH_FILE" ]; then
    echo "ERROR: BVH file not found: $BVH_FILE"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "ERROR: JSON file not found: $JSON_FILE"
    exit 1
fi

if [ ! -f "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    exit 1
fi

echo "✓ BVH file found: $BVH_FILE"
echo "✓ JSON file found: $JSON_FILE"
echo "✓ Audio file found: $AUDIO_FILE"

# Step 3: Render video with Blender
echo ""
echo "Step 3: Rendering video with Blender (headless mode)..."
echo "------------------------------------------------------------------------"
echo "This may take a few minutes depending on video duration..."

blender -b assets/beat_visualize.blend --python render_realtime_output.py -- \
    --bvh "$BVH_FILE" \
    --json "$JSON_FILE" \
    --audio "$AUDIO_FILE" \
    --output "$OUTPUT_VIDEO"

# Check if rendering completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Rendering failed!"
    exit 1
fi

# Step 4: Verify output video
echo ""
echo "========================================================================"
echo "Pipeline Complete!"
echo "========================================================================"

if [ -f "$OUTPUT_VIDEO" ]; then
    VIDEO_SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
    echo "✓ Video rendered successfully!"
    echo ""
    echo "Output video: $(realpath $OUTPUT_VIDEO)"
    echo "Video size: $VIDEO_SIZE"
    echo ""
    echo "You can download this file to view it:"
    echo "  scp user@server:$(realpath $OUTPUT_VIDEO) ."
else
    echo "WARNING: Video file not found at expected location"
    echo "Check Blender output above for errors"
fi

echo "========================================================================"
