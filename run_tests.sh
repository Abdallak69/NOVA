#!/bin/bash

# Script to run the test suite for the QNN project

# Exit immediately if a command exits with a non-zero status
set -e

# Activate the virtual environment if it exists
if [ -d "qnn_env" ]; then
    echo "Activating virtual environment..."
    source qnn_env/bin/activate
fi

# Ensure dependencies are installed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Run the test suite
echo "Running tests..."
python -m unittest test_ansatz.py

# Check if we should create test coverage report
if [ "$1" == "--coverage" ]; then
    echo "Generating coverage report..."
    pip install -q coverage
    coverage run -m unittest test_ansatz.py
    coverage report -m
    coverage html
    echo "Coverage report generated in htmlcov/ directory"
fi

echo "Tests completed successfully!" 