#!/bin/bash
# Script to run the QNN application

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment if it exists
if [ -d "qnn_env" ]; then
    echo "Activating virtual environment..."
    source qnn_env/bin/activate
fi

# Parse command line arguments
case "$1" in
    --cli)
        echo "Running CLI interface..."
        python cli_interface.py
        ;;
    --gui)
        echo "Running GUI interface..."
        python gui_interface.py
        ;;
    --ansatz-comparison)
        echo "Running Ansatz Comparison GUI..."
        python ansatz_comparison_gui.py
        ;;
    --help)
        echo "Usage: $0 [OPTION]"
        echo "Run the Quantum Neural Network application."
        echo ""
        echo "Options:"
        echo "  --cli                Run the command-line interface (default)"
        echo "  --gui                Run the graphical user interface"
        echo "  --ansatz-comparison  Run the ansatz comparison GUI"
        echo "  --help               Display this help message"
        ;;
    *)
        # Default is CLI interface
        echo "Running CLI interface..."
        python cli_interface.py
        ;;
esac 