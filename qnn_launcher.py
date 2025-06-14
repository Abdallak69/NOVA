#!/usr/bin/env python3
"""
Launcher script for the Quantum Neural Network project.

This script provides a simple graphical interface to launch different components
of the QNN project, including the CLI, GUI, and Ansatz Comparison tools.
"""

import os
import platform
import sys

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class QNNLauncher(QMainWindow):
    """Main launcher window for the QNN project."""

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("QNN Project Launcher")
        self.setGeometry(100, 100, 800, 600)

        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create header
        self.setup_header()

        # Create the launcher options
        self.setup_launcher_options()

        # Create the description area
        self.setup_description_area()

        # Create the launch button
        self.setup_launch_button()

        # Set the default description
        self.update_description(0)

        # Initialize process
        self.process = None

        self.show()

    def setup_header(self):
        """Set up the header with title and description."""
        header_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Quantum Neural Network for Molecular Energy Estimation")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("Select a component to launch")
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle_label)

        self.main_layout.addLayout(header_layout)

    def setup_launcher_options(self):
        """Set up the launcher options."""
        self.options_group = QGroupBox("Available Components")
        options_layout = QVBoxLayout()

        # Create radio buttons for each option
        self.button_group = QButtonGroup(self)

        # CLI Option
        self.cli_radio = QRadioButton("Command-Line Interface")
        self.cli_radio.setChecked(True)  # Default selection
        self.button_group.addButton(self.cli_radio, 0)
        options_layout.addWidget(self.cli_radio)

        # GUI Option
        self.gui_radio = QRadioButton("Graphical User Interface")
        self.button_group.addButton(self.gui_radio, 1)
        options_layout.addWidget(self.gui_radio)

        # Ansatz Comparison Option
        self.ansatz_radio = QRadioButton("Ansatz Comparison Tool")
        self.button_group.addButton(self.ansatz_radio, 2)
        options_layout.addWidget(self.ansatz_radio)

        # Run Tests Option
        self.tests_radio = QRadioButton("Run Test Suite")
        self.button_group.addButton(self.tests_radio, 3)
        options_layout.addWidget(self.tests_radio)

        self.options_group.setLayout(options_layout)
        self.main_layout.addWidget(self.options_group)

        # Connect signals
        self.button_group.buttonClicked[int].connect(self.update_description)

    def setup_description_area(self):
        """Set up the description area."""
        self.description_group = QGroupBox("Description")
        description_layout = QVBoxLayout()

        self.description_text = QTextEdit()
        self.description_text.setReadOnly(True)
        description_layout.addWidget(self.description_text)

        self.description_group.setLayout(description_layout)
        self.main_layout.addWidget(self.description_group)

    def setup_launch_button(self):
        """Set up the launch button."""
        button_layout = QHBoxLayout()

        self.launch_button = QPushButton("Launch")
        self.launch_button.setMinimumHeight(50)
        launch_font = QFont()
        launch_font.setPointSize(12)
        launch_font.setBold(True)
        self.launch_button.setFont(launch_font)
        self.launch_button.clicked.connect(self.launch_selected)

        button_layout.addWidget(self.launch_button)
        self.main_layout.addLayout(button_layout)

    def update_description(self, button_id):
        """Update the description based on the selected option."""
        descriptions = {
            0: """<h3>Command-Line Interface</h3>
                <p>The CLI provides a text-based interface for running the QNN model. It's a good choice if you're familiar with command-line tools or if you're running on a system without graphical capabilities.</p>
                
                <p>Features:</p>
                <ul>
                    <li>Create and train QNN models for different molecules</li>
                    <li>Set bond lengths and other molecular parameters</li>
                    <li>Choose from different ansatz types</li>
                    <li>View energy convergence during training</li>
                    <li>Compare different ansatz types</li>
                </ul>
                
                <p>The CLI is the most reliable way to run the QNN as it has minimal dependencies.</p>""",
            1: """<h3>Graphical User Interface</h3>
                <p>The GUI provides a user-friendly interface with interactive controls and visualizations. It's ideal for exploring the QNN model if you prefer graphical interfaces.</p>
                
                <p>Features:</p>
                <ul>
                    <li>Interactive molecule creation and parameter setting</li>
                    <li>Real-time visualization of energy convergence</li>
                    <li>Circuit visualization tools</li>
                    <li>Save and load trained models</li>
                    <li>Export results and figures</li>
                </ul>
                
                <p>Note: The GUI requires PyQt5 to be properly installed.</p>""",
            2: """<h3>Ansatz Comparison Tool</h3>
                <p>This specialized tool helps you compare the performance of different ansatz circuits for molecular energy estimation. It's useful for understanding which ansatz works best for specific molecules.</p>
                
                <p>Features:</p>
                <ul>
                    <li>Direct comparison of multiple ansatz types</li>
                    <li>Visualization of energy convergence for each ansatz</li>
                    <li>Circuit visualization and parameter counts</li>
                    <li>Performance metrics for each ansatz</li>
                    <li>Export comparison results and figures</li>
                </ul>
                
                <p>Note: This tool requires PyQt5 to be properly installed.</p>""",
            3: """<h3>Run Test Suite</h3>
                <p>The test suite runs unit tests to verify that all components of the QNN project are functioning correctly, particularly the ansatz circuit implementations.</p>
                
                <p>What it tests:</p>
                <ul>
                    <li>Correct creation of different ansatz circuits</li>
                    <li>Parameter assignment and retrieval</li>
                    <li>Circuit building and validation</li>
                    <li>Factory function operation</li>
                </ul>
                
                <p>Use this option when you want to make sure everything is working properly, especially after making changes to the code.</p>""",
        }

        self.description_text.setHtml(descriptions[button_id])

    def launch_selected(self):
        """Launch the selected component."""
        button_id = self.button_group.checkedId()

        # Check if a process is already running
        if self.process and self.process.state() == QProcess.Running:
            QMessageBox.warning(
                self,
                "Process Running",
                "A component is already running. Please wait for it to finish.",
            )
            return

        try:
            if button_id == 0:  # CLI
                self.launch_cli()
            elif button_id == 1:  # GUI
                self.launch_gui()
            elif button_id == 2:  # Ansatz Comparison
                self.launch_ansatz_comparison()
            elif button_id == 3:  # Tests
                self.launch_tests()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch: {str(e)}")

    def launch_cli(self):
        """Launch the CLI interface."""
        self.launch_script("cli_interface.py")

    def launch_gui(self):
        """Launch the GUI interface."""
        self.launch_script("gui_interface.py")

    def launch_ansatz_comparison(self):
        """Launch the ansatz comparison tool."""
        self.launch_script("ansatz_comparison_gui.py")

    def launch_tests(self):
        """Launch the test suite."""
        if platform.system() == "Windows":
            script = "run_tests.bat"
        else:
            script = "./run_tests.sh"

        self.launch_script(script)

    def launch_script(self, script):
        """Launch a script using QProcess."""
        # Initialize QProcess if needed
        if not self.process:
            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self.handle_output)
            self.process.readyReadStandardError.connect(self.handle_error)
            self.process.finished.connect(self.handle_finished)

        # Determine Python interpreter path
        if os.path.exists("qnn_env"):
            if platform.system() == "Windows":
                python_path = os.path.join("qnn_env", "Scripts", "python.exe")
            else:
                python_path = os.path.join("qnn_env", "bin", "python")
        else:
            python_path = sys.executable

        # Start the process
        if script.endswith(".py"):
            self.process.start(python_path, [script])
            QMessageBox.information(self, "Launching", f"Launching {script}...")
        else:
            # For shell scripts
            if platform.system() == "Windows":
                self.process.start(script)
            else:
                self.process.start("/bin/bash", [script])
            QMessageBox.information(self, "Launching", f"Launching {script}...")

    def handle_output(self):
        """Handle standard output from the process."""
        output = self.process.readAllStandardOutput().data().decode()
        print(output, end="")

    def handle_error(self):
        """Handle standard error from the process."""
        error = self.process.readAllStandardError().data().decode()
        print(error, end="")

    def handle_finished(self, exit_code, exit_status):
        """Handle process completion."""
        if exit_code == 0:
            print("Process completed successfully.")
        else:
            print(f"Process exited with code {exit_code}.")


def main():
    """Main function to run the launcher."""
    app = QApplication(sys.argv)
    window = QNNLauncher()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
