#!/bin/bash
# Script to set up Python virtual environment for QNN project

set -e  # Exit immediately if a command exits with a non-zero status

echo "==================================================================="
echo "Setting up Python virtual environment for Quantum Neural Network"
echo "==================================================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Detected: $python_version"

# Extract major and minor version
python_major=$(echo $python_version | cut -d' ' -f2 | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d' ' -f2 | cut -d'.' -f2)

# Check if Python version is at least 3.8
if [[ $python_major -lt 3 || ($python_major -eq 3 && $python_minor -lt 8) ]]; then
    echo "Error: This project requires Python 3.8 or higher."
    echo "Please install a compatible Python version and try again."
    exit 1
fi

# Create the virtual environment if it doesn't exist
if [ ! -d "qnn_env" ]; then
    echo "Creating virtual environment 'qnn_env'..."
    python3 -m venv qnn_env
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please make sure python3-venv is installed on your system."
        echo "On Ubuntu/Debian: sudo apt-get install python3-venv"
        echo "On macOS: pip3 install virtualenv"
        exit 1
    fi
else
    echo "Virtual environment 'qnn_env' already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source qnn_env/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != *"qnn_env"* ]]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies with verbose output
echo "Installing dependencies..."
pip install -v -r requirements.txt

# Verify key packages are installed
echo "Verifying installations..."
MISSING_PACKAGES=()

check_package() {
    python -c "import $1" 2>/dev/null || MISSING_PACKAGES+=("$1")
}

# Check core packages
check_package cirq
check_package numpy
check_package matplotlib
check_package sympy
check_package scipy
check_package openfermion

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "WARNING: The following packages could not be imported:"
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "  - $pkg"
    done
    echo "You may need to install these packages manually."
else
    echo "All core packages verified successfully!"
fi

# Create a .env file for VSCode to recognize the environment
echo "Creating VSCode environment settings..."
cat > .env << EOL
PYTHONPATH=${VIRTUAL_ENV}/lib/python3.11/site-packages:${PYTHONPATH}
EOL

# Make scripts executable
echo "Making scripts executable..."
chmod +x run_qnn.sh
chmod +x force_install_dependencies.sh
chmod +x run_tests.sh
chmod +x qnn_launcher.py

# Create pyrightconfig.json if it doesn't exist
if [ ! -f "pyrightconfig.json" ]; then
    echo "Creating Pylance configuration file..."
    cat > pyrightconfig.json << EOF
{
    "venvPath": ".",
    "venv": "qnn_env",
    "extraPaths": [
        "qnn_env/lib/python3.8/site-packages",
        "qnn_env/lib/python3.9/site-packages",
        "qnn_env/lib/python3.10/site-packages",
        "qnn_env/lib/python3.11/site-packages"
    ],
    "reportMissingImports": false,
    "reportMissingModuleSource": false
}
EOF
fi

# Create .vscode directory and settings.json if they don't exist
if [ ! -d ".vscode" ]; then
    mkdir -p .vscode
fi

if [ ! -f ".vscode/settings.json" ]; then
    echo "Creating VS Code settings..."
    cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "qnn_env/bin/python",
    "python.analysis.extraPaths": [
        "qnn_env/lib/python3.8/site-packages",
        "qnn_env/lib/python3.9/site-packages",
        "qnn_env/lib/python3.10/site-packages",
        "qnn_env/lib/python3.11/site-packages"
    ]
}
EOF
fi

echo "==================================================================="
echo "Setup complete! To activate the environment, run:"
echo "source qnn_env/bin/activate"
echo "===================================================================" 