#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate the environment
source .venv/bin/activate

# 3. Upgrade pip and install requirements
echo "Installing dependencies..."
uv pip install --upgrade pip
uv pip install -e .


# Compile C++ Pybinds using setup.py
echo "Installing build-time dependencies..."
uv pip install pybind11 setuptools wheel

echo "Compiling C++ extensions..."
python3 setup.py build_ext --inplace

mkdir -p database/storage

echo "Setup complete."