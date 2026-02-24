#!/bin/bash
#
# Convenience Wrapper for Offline Keypoint Analysis
#
# Activates the virtual environment and runs analyze_keypoints.py with provided arguments.
#
# Usage:
#   bash run_offline.sh --keypoints-2d <path> --keypoints-3d <path> [--video <path>] [--output-dir <path>]
#
# Example:
#   bash run_offline.sh \
#     --keypoints-2d outputs/job_id/keypoints_2d.json \
#     --keypoints-3d outputs/job_id/keypoints_3d.json \
#     --video input_video.mp4
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo ""
    echo "Please run setup first:"
    echo "  bash setup_venv.sh"
    exit 1
fi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Run the analysis script with all provided arguments
python "${SCRIPT_DIR}/analyze_keypoints.py" "$@"
