#!/bin/bash
#
# Setup Python Virtual Environment for Offline Keypoint Analysis
#
# This script creates a lightweight Python virtual environment with only the
# dependencies needed for running analysis on pre-computed keypoints.
# No GPU, pose estimation, or job queue dependencies are included.
#
# Usage:
#   bash setup_venv.sh
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="${SCRIPT_DIR}/.venv"

echo "=========================================="
echo "Offline Keypoint Analysis - venv Setup"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.9 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Using: $PYTHON_VERSION"
echo ""

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo "✓ Created new virtual environment"
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Created virtual environment at: $VENV_DIR"
fi

echo ""
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Ensure pip is up to date
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ Pip upgraded"

echo ""
echo "Installing dependencies from requirements-minimum.txt..."
pip install -r "${SCRIPT_DIR}/requirements-minimum.txt"

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo ""
echo "  source ${SCRIPT_DIR}/.venv/bin/activate"
echo ""
echo "Then you can run the offline analysis:"
echo ""
echo "  python analyze_keypoints.py --keypoints-2d <path> --keypoints-3d <path> [--video <path>] [--output-dir <path>]"
echo ""
echo "To deactivate, simply run:"
echo ""
echo "  deactivate"
echo ""
