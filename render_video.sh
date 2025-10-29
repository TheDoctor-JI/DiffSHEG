#!/bin/bash
# Quick render script - renders video from existing BVH/JSON files

# Default paths
BVH_FILE="${1:-results/realtime_test/bvh/realtime_output.bvh}"
JSON_FILE="${2:-results/realtime_test/face_json/realtime_output.json}"
AUDIO_FILE="${3:-results/realtime_test/2_scott_0_3_3.wav}"
OUTPUT_VIDEO="${4:-results/realtime_test/output_video.mp4}"

echo "========================================================================"
echo "Rendering Video with Blender (Headless)"
echo "========================================================================"
echo "BVH:    $BVH_FILE"
echo "JSON:   $JSON_FILE"
echo "Audio:  $AUDIO_FILE"
echo "Output: $OUTPUT_VIDEO"
echo "========================================================================"

# Check if files exist
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

# Run Blender in headless mode
blender -b assets/beat_visualize.blend --python render_realtime_output.py -- \
    --bvh "$BVH_FILE" \
    --json "$JSON_FILE" \
    --audio "$AUDIO_FILE" \
    --output "$OUTPUT_VIDEO"

# Check result
if [ $? -eq 0 ] && [ -f "$OUTPUT_VIDEO" ]; then
    echo ""
    echo "========================================================================"
    echo "✓ Rendering complete!"
    echo "Video saved to: $(realpath $OUTPUT_VIDEO)"
    echo "Size: $(du -h $OUTPUT_VIDEO | cut -f1)"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ Rendering failed!"
    echo "Check Blender output above for errors"
    echo "========================================================================"
    exit 1
fi
