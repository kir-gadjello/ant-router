#!/bin/bash
set -e

# --- Configuration ---
# You can edit these default values
MODEL_A_DEFAULT="claude-3-5-sonnet-20241022"
MODEL_B_DEFAULT="claude-3-opus-20240229"
COMMAND_DEFAULT="claude code 'write a hello world in python'"
ROUTER_BIN="./target/debug/anthropic-bridge"

MODEL_A="${MODEL_A:-$MODEL_A_DEFAULT}"
MODEL_B="${MODEL_B:-$MODEL_B_DEFAULT}"
COMMAND="${COMMAND:-$COMMAND_DEFAULT}"

# --- Setup ---
VENV_DIR=".venv"
REQUIREMENTS_FILE="tools/requirements.txt"

if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed."
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f "$REQUIREMENTS_FILE" ]; then
    uv pip install -q -r "$REQUIREMENTS_FILE"
fi

# --- Build ---
echo "Building Router..."
cargo build --quiet

# --- Run ---
echo "Running Model Diff..."
echo "Model A: $MODEL_A"
echo "Model B: $MODEL_B"
echo "Command: $COMMAND"
echo "---------------------------------------------------"

# Pass arguments to the python script
# Note: we need to handle the command argument carefully if it contains spaces
python tools/model_diff.py \
    --model-a "$MODEL_A" \
    --model-b "$MODEL_B" \
    --router-bin "$ROUTER_BIN" \
    $COMMAND
