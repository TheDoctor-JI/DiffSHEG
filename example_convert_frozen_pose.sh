#!/bin/bash
# Example: Convert a frozen skeleton pose to custom neutral positions
#
# USAGE:
#   ./example_convert_frozen_pose.sh frozen_skeleton_1234567890.yaml
#
# This will:
#   1. Convert the frozen pose to normalized space
#   2. Output custom_neutral_positions.yaml
#   3. Print instructions for adding to config

if [ $# -eq 0 ]; then
    echo "Usage: $0 <frozen_skeleton_file.yaml>"
    echo ""
    echo "Example:"
    echo "  $0 frozen_skeleton_1234567890.yaml"
    echo ""
    echo "This will convert the frozen pose to custom neutral positions"
    echo "that can be used in embodiment_manager_config.yaml"
    exit 1
fi

FROZEN_FILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if file exists
if [ ! -f "$FROZEN_FILE" ]; then
    echo "Error: File not found: $FROZEN_FILE"
    exit 1
fi

echo "Converting frozen skeleton pose to custom neutral positions..."
echo "Input file: $FROZEN_FILE"
echo ""

# Run conversion script
python3 "$SCRIPT_DIR/convert_frozen_pose.py" \
    --input "$FROZEN_FILE" \
    --cache_name beat_4english_15_141 \
    --output custom_neutral_positions.yaml

echo ""
echo "Done! Next steps:"
echo "  1. Open embodiment_manager_config.yaml"
echo "  2. Add custom_neutral_positions to co_speech_gestures section"
echo "  3. Restart the server"
