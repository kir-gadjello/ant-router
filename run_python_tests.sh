#!/bin/bash
set -e

# Configuration
VENV_DIR=".venv"
REQUIREMENTS_FILE="tools/requirements.txt"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it (e.g., 'curl -LsSf https://astral.sh/uv/install.sh | sh')."
    exit 1
fi

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    uv pip install -r "$REQUIREMENTS_FILE"
else
    # Fallback if requirements file missing (though we created it)
    uv pip install requests rich deepdiff flask
fi

# Run Tests
echo "Running Python tests..."
# Assuming tests are discoverable or specific files.
# The user mentioned "run python tests". Let's run all .py files in tests/ that look like tests.
# Or use pytest if installed. Let's add pytest to deps if we want robust testing.
# For now, just executing specific known E2E scripts or a pattern.

if [ -f "tests/e2e_tool_calling.py" ]; then
    python tests/e2e_tool_calling.py
else
    echo "No specific python test file found at tests/e2e_tool_calling.py"
fi

# Add other test runners here if needed
