#!/bin/bash
# Script to force install all dependencies for the QNN project

echo "Forcing installation of all required packages..."

# Check if virtual environment exists
if [ ! -d "qnn_env" ]; then
    echo "Virtual environment 'qnn_env' not found. Creating it..."
    python3.11 -m venv qnn_env
fi

# Activate the virtual environment
source qnn_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install packages one by one
echo "Installing cirq..."
pip install cirq --force-reinstall

echo "Installing numpy..."
pip install numpy --force-reinstall

echo "Installing matplotlib..."
pip install matplotlib --force-reinstall

echo "Installing sympy..."
pip install sympy --force-reinstall

echo "Installing scipy..."
pip install scipy --force-reinstall

echo "Installing openfermion..."
pip install openfermion --force-reinstall

# Optional packages
echo "Installing PyQt5 (optional for GUI)..."
pip install PyQt5 --force-reinstall

echo "All dependencies have been forcefully reinstalled."
echo "Try reloading VS Code and see if the import errors are resolved." 