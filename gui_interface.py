#!/usr/bin/env python3

"""
GUI Interface for Quantum Neural Network Molecular Energy Estimation

This module provides a graphical user interface for testing and visualizing results
from the Quantum Neural Network (QNN) for molecular energy estimation.
"""

# Set matplotlib backend explicitly before any other imports
import matplotlib

matplotlib.use("Qt5Agg")

import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Import the QNN implementation
from qnn_molecular_energy import MolecularQNN


class MplCanvas(FigureCanvas):
    """
    Canvas for embedding Matplotlib plots in the Qt GUI.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the canvas for matplotlib plots.

        Args:
            parent: The parent widget
            width (int): Figure width in inches
            height (int): Figure height in inches
            dpi (int): Dots per inch (resolution)
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class TrainingThread(QThread):
    """
    Thread for running QNN training in the background.

    This prevents the GUI from freezing during training.
    """

    # Signals to communicate with the main thread
    progress_signal = pyqtSignal(int)
    update_plot_signal = pyqtSignal(list)
    training_complete_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, qnn, iterations, method="BFGS"):
        """
        Initialize the training thread.

        Args:
            qnn (MolecularQNN): The QNN instance to train
            iterations (int): Maximum number of optimization iterations
            method (str): Optimization method
        """
        super().__init__()
        self.qnn = qnn
        self.iterations = iterations
        self.method = method
        self.current_iteration = 0
        self.energy_history = []

    def run(self):
        """Run the training process in a separate thread."""
        try:
            self.log_signal.emit("Starting training...")

            # Define a callback function to track progress
            def callback(params):
                try:
                    energy = self.qnn._energy_expectation(params)
                    self.energy_history.append(energy)
                    self.current_iteration += 1

                    # Update progress bar (approximately)
                    progress = min(
                        100, int(100 * self.current_iteration / self.iterations)
                    )
                    self.progress_signal.emit(progress)

                    # Update plot every 5 iterations
                    if len(self.energy_history) % 5 == 0:
                        self.update_plot_signal.emit(self.energy_history.copy())
                        self.log_signal.emit(
                            f"Iteration {len(self.energy_history)}: Energy = {energy:.6f} Hartree"
                        )
                except Exception as e:
                    # Log the error but don't stop optimization
                    self.log_signal.emit(f"Warning in callback: {str(e)}")

            # Train the QNN
            from scipy.optimize import minimize

            try:
                result = minimize(
                    self.qnn._energy_expectation,
                    self.qnn.params,
                    method=self.method,
                    options={"maxiter": self.iterations},
                    callback=callback,
                )
            except ValueError as e:
                if "Unknown solver" in str(e):
                    self.error_signal.emit(
                        f"Invalid optimization method '{self.method}'. Try using BFGS, COBYLA, or Nelder-Mead."
                    )
                    return
                else:
                    raise

            # Update the QNN with the results
            self.qnn.optimal_params = result.x
            self.qnn.final_energy = result.fun
            self.qnn.energy_history = self.energy_history
            self.qnn.params = result.x
            self.qnn.param_resolver = dict(zip(self.qnn.symbols, self.qnn.params))

            # Emit signal with the results
            results = {
                "energy": self.qnn.final_energy,
                "params": self.qnn.optimal_params,
                "energy_history": self.energy_history,
                "success": result.success,
                "iterations": len(self.energy_history),
            }

            # Add convergence status message
            if not result.success:
                self.log_signal.emit(
                    f"Warning: Optimization may not have fully converged. Message: {result.message}"
                )

            self.training_complete_signal.emit(results)

        except KeyboardInterrupt:
            self.error_signal.emit("Training was interrupted")

        except MemoryError:
            self.error_signal.emit(
                "Out of memory. Try using fewer qubits or a simpler ansatz."
            )

        except Exception as e:
            # Get more detailed error information
            import traceback

            error_details = traceback.format_exc()

            # Log the full error to console for debugging
            print(f"Training error: {str(e)}\n{error_details}")

            # Send a user-friendly message to the UI
            if "simulator too large" in str(e).lower():
                self.error_signal.emit(
                    "The quantum circuit is too large for simulation. Try using fewer qubits or a simpler ansatz."
                )
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                self.error_signal.emit(
                    "Network error or timeout when connecting to quantum hardware. Please check your connection and try again."
                )
            else:
                self.error_signal.emit(f"Training error: {str(e)}")


class QNNApp(QMainWindow):
    """
    Main application window for the QNN GUI.
    """

    def __init__(self):
        """Initialize the application window and UI components."""
        super().__init__()

        # Set window title and size
        self.setWindowTitle("Quantum Neural Network for Molecular Energy Estimation")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize QNN instance
        self.qnn = None
        self.training_thread = None

        # Set up the UI
        self.init_ui()

    def init_ui(self):
        """Set up the user interface components."""
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Setup control panels
        self.setup_molecule_controls(left_layout)
        self.setup_training_controls(left_layout)
        self.setup_visualization_controls(right_layout)
        self.setup_status_panel(left_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)  # Give right panel more space

        # Set main widget
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Initial state setup
        self.update_ui_state(is_trained=False)

    def setup_molecule_controls(self, parent_layout):
        """
        Set up controls for molecule selection and parameters.

        Args:
            parent_layout (QLayout): The parent layout to add controls to
        """
        molecule_group = QGroupBox("Molecule Configuration")
        form_layout = QFormLayout()

        # Molecule selection
        self.molecule_combo = QComboBox()
        self.molecule_combo.addItems(["H2", "LiH", "H2O"])
        self.molecule_combo.currentTextChanged.connect(self.on_molecule_changed)
        form_layout.addRow("Molecule:", self.molecule_combo)

        # Bond length
        self.bond_length_spin = QDoubleSpinBox()
        self.bond_length_spin.setRange(0.1, 5.0)
        self.bond_length_spin.setSingleStep(0.01)
        self.bond_length_spin.setValue(0.74)  # Default for H2
        self.bond_length_spin.setDecimals(2)
        form_layout.addRow("Bond Length (Å):", self.bond_length_spin)

        # Circuit depth
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 10)
        self.depth_spin.setValue(2)
        form_layout.addRow("Circuit Depth:", self.depth_spin)

        # Create QNN button
        self.create_qnn_btn = QPushButton("Create Quantum Neural Network")
        self.create_qnn_btn.clicked.connect(self.on_create_qnn)

        # Add to form layout
        molecule_group.setLayout(form_layout)
        parent_layout.addWidget(molecule_group)
        parent_layout.addWidget(self.create_qnn_btn)

    def setup_training_controls(self, parent_layout):
        """
        Set up controls for QNN training parameters.

        Args:
            parent_layout (QLayout): The parent layout to add controls to
        """
        training_group = QGroupBox("Training Configuration")
        form_layout = QFormLayout()

        # Iterations
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 1000)
        self.iterations_spin.setValue(100)
        self.iterations_spin.setSingleStep(10)
        form_layout.addRow("Max Iterations:", self.iterations_spin)

        # Optimization method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell"])
        form_layout.addRow("Optimization Method:", self.method_combo)

        # Training controls
        training_btn_layout = QHBoxLayout()

        self.train_btn = QPushButton("Train Network")
        self.train_btn.clicked.connect(self.on_train)
        self.train_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self.on_stop_training)
        self.stop_btn.setEnabled(False)

        training_btn_layout.addWidget(self.train_btn)
        training_btn_layout.addWidget(self.stop_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Add to form layout
        training_group.setLayout(form_layout)
        parent_layout.addWidget(training_group)
        parent_layout.addLayout(training_btn_layout)
        parent_layout.addWidget(self.progress_bar)

        # Save/Load section
        io_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save Model")
        self.save_btn.clicked.connect(self.on_save_model)
        self.save_btn.setEnabled(False)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.on_load_model)

        io_layout.addWidget(self.save_btn)
        io_layout.addWidget(self.load_btn)

        parent_layout.addLayout(io_layout)

    def setup_visualization_controls(self, parent_layout):
        """
        Set up visualization tabs and plots.

        Args:
            parent_layout (QLayout): The parent layout to add controls to
        """
        # Create tabs for different visualizations
        self.tabs = QTabWidget()

        # Tab 1: Energy convergence plot
        self.energy_plot_tab = QWidget()
        energy_plot_layout = QVBoxLayout()

        # Energy plot
        self.energy_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        energy_plot_layout.addWidget(self.energy_canvas)

        # Results panel
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()

        self.current_energy_label = QLabel("N/A")
        self.exact_energy_label = QLabel("N/A")
        self.error_label = QLabel("N/A")
        self.training_time_label = QLabel("N/A")

        results_layout.addRow("Current Energy:", self.current_energy_label)
        results_layout.addRow("Exact Energy:", self.exact_energy_label)
        results_layout.addRow("Error:", self.error_label)
        results_layout.addRow("Training Time:", self.training_time_label)

        results_group.setLayout(results_layout)
        energy_plot_layout.addWidget(results_group)

        self.energy_plot_tab.setLayout(energy_plot_layout)

        # Tab 2: Circuit visualization
        self.circuit_tab = QWidget()
        circuit_layout = QVBoxLayout()

        self.circuit_text = QTextEdit()
        self.circuit_text.setReadOnly(True)
        self.circuit_text.setFont(QFont("Courier New", 10))
        circuit_layout.addWidget(self.circuit_text)

        self.circuit_tab.setLayout(circuit_layout)

        # Add tabs to the tab widget
        self.tabs.addTab(self.energy_plot_tab, "Energy Convergence")
        self.tabs.addTab(self.circuit_tab, "Quantum Circuit")

        # Add to parent layout
        parent_layout.addWidget(self.tabs)

    def setup_status_panel(self, parent_layout):
        """
        Set up status and log panel at the bottom.

        Args:
            parent_layout (QLayout): The parent layout to add controls to
        """
        status_group = QGroupBox("Log")
        status_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)

        status_layout.addWidget(self.log_text)
        status_group.setLayout(status_layout)

        parent_layout.addWidget(status_group)

    def update_ui_state(self, is_trained=False, is_loaded=False, is_training=False):
        """
        Update the enabled/disabled state of UI elements based on application state.

        Args:
            is_trained (bool): Whether a model has been trained
            is_loaded (bool): Whether a model has been loaded
            is_training (bool): Whether training is in progress
        """
        # Molecule controls
        self.molecule_combo.setEnabled(not is_training)
        self.bond_length_spin.setEnabled(not is_training)
        self.depth_spin.setEnabled(not is_training)
        self.create_qnn_btn.setEnabled(not is_training and not is_loaded)

        # Training controls
        self.iterations_spin.setEnabled(not is_training)
        self.method_combo.setEnabled(not is_training)
        self.train_btn.setEnabled((self.qnn is not None) and not is_training)
        self.stop_btn.setEnabled(is_training)

        # Save/Load
        self.save_btn.setEnabled((is_trained or is_loaded) and not is_training)
        self.load_btn.setEnabled(not is_training)

    def on_molecule_changed(self, molecule):
        """
        Handle molecule selection change by updating default bond length.

        Args:
            molecule (str): Selected molecule name
        """
        # Update default bond length based on molecule
        if molecule == "H2":
            self.bond_length_spin.setValue(0.74)
        elif molecule == "LiH":
            self.bond_length_spin.setValue(1.60)
        elif molecule == "H2O":
            self.bond_length_spin.setValue(0.96)

    def on_create_qnn(self):
        """Create a new QNN instance based on the UI parameters."""
        try:
            # Get parameters from UI
            molecule = self.molecule_combo.currentText()
            bond_length = self.bond_length_spin.value()
            depth = self.depth_spin.value()

            # Get ansatz type
            ansatz_type = (
                self.ansatz_combo.currentText()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            # Prepare ansatz kwargs based on type
            ansatz_kwargs = {}
            if ansatz_type == "hardware_efficient":
                rotation_gates = self.rotation_combo.currentText()
                entangle_pattern = self.entangle_combo.currentText().lower()
                ansatz_kwargs = {
                    "rotation_gates": rotation_gates,
                    "entangle_pattern": entangle_pattern,
                }

            # Create the QNN
            self.log_message(
                f"Creating QNN for {molecule} with {ansatz_type} ansatz..."
            )

            # Create QNN instance
            self.qnn = MolecularQNN(
                molecule=molecule,
                bond_length=bond_length,
                depth=depth,
                ansatz_type=ansatz_type,
                ansatz_kwargs=ansatz_kwargs,
            )

            # Update UI
            self.log_message("QNN created successfully!")
            self.log_message(f"Number of qubits: {self.qnn.n_qubits}")
            self.log_message(f"Number of parameters: {len(self.qnn.params)}")

            # Update UI state
            self.update_ui_state(is_loaded=True)

            # Update circuit visualization
            self.update_circuit_visualization()

            # Update exact energy if available
            self.update_exact_energy()

        except ValueError as e:
            self.show_error(f"Invalid parameter: {str(e)}")
        except TypeError as e:
            self.show_error(
                f"Type error: {str(e)}\nCheck the parameter types and try again."
            )
        except Exception as e:
            import traceback

            print(f"Error creating QNN: {str(e)}")
            print(traceback.format_exc())
            self.show_error(f"Error creating QNN: {str(e)}")

    def on_train(self):
        """Start the training process in a background thread."""
        if not self.qnn:
            self.show_error("No QNN model created yet. Please create a model first.")
            return

        try:
            # Get training parameters
            iterations = self.iterations_spin.value()
            method = self.optimizer_combo.currentText()

            # Disable training button and update UI
            self.update_ui_state(is_training=True)
            self.log_message(
                f"Starting training with {method} optimizer for {iterations} iterations..."
            )

            # Create and start training thread
            self.training_thread = TrainingThread(self.qnn, iterations, method)
            self.training_thread.progress_signal.connect(self.update_progress)
            self.training_thread.update_plot_signal.connect(self.update_plot)
            self.training_thread.training_complete_signal.connect(
                self.on_training_complete
            )
            self.training_thread.error_signal.connect(self.show_error)
            self.training_thread.log_signal.connect(self.log_message)

            # Start the thread
            self.training_thread.start()

            # Add stop button functionality
            self.stop_button.clicked.connect(self.on_stop_training)

        except Exception as e:
            self.update_ui_state(is_training=False)
            self.show_error(f"Error starting training: {str(e)}")

    def on_stop_training(self):
        """Stop the training process."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.log_message("Training stopped by user")
            self.update_ui_state(is_trained=True)

    def on_training_complete(self, results):
        """Handle training completion."""
        try:
            # Update UI state
            self.update_ui_state(is_trained=True, is_training=False)

            # Log results
            energy = results["energy"]
            success = results["success"]
            iterations = results["iterations"]

            status = "completed successfully" if success else "completed with warnings"
            self.log_message(f"Training {status} after {iterations} iterations")
            self.log_message(f"Final energy: {energy:.6f} Hartree")

            # Update energy history plot with final values
            energy_history = results["energy_history"]
            self.update_plot(energy_history)

            # Compute and show error if exact energy is available
            if hasattr(self.qnn, "exact_energy") and self.qnn.exact_energy is not None:
                exact = self.qnn.exact_energy
                error = abs(energy - exact)
                error_percent = 100 * error / abs(exact)
                self.log_message(f"Exact energy: {exact:.6f} Hartree")
                self.log_message(
                    f"Absolute error: {error:.6f} Hartree ({error_percent:.4f}%)"
                )

                # Add reference line to plot
                self.energy_canvas.axes.axhline(
                    y=exact, color="r", linestyle="--", label=f"Exact: {exact:.6f}"
                )
                self.energy_canvas.axes.legend()
                self.energy_canvas.draw()

        except Exception as e:
            self.show_error(f"Error processing training results: {str(e)}")

    def update_progress(self, value):
        """
        Update the progress bar.

        Args:
            value (int): Progress percentage (0-100)
        """
        self.progress_bar.setValue(value)

    def update_plot(self, energy_history):
        """
        Update the energy convergence plot.

        Args:
            energy_history (list): History of energy values
        """
        # Clear the plot
        self.energy_canvas.axes.clear()

        # Plot energy history
        self.energy_canvas.axes.plot(energy_history, "o-", color="blue", alpha=0.7)
        self.energy_canvas.axes.set_xlabel("Iteration", fontsize=12)
        self.energy_canvas.axes.set_ylabel("Energy (Hartree)", fontsize=12)
        self.energy_canvas.axes.set_title(
            f"QNN Energy Convergence for {self.qnn.molecule}", fontsize=14
        )
        self.energy_canvas.axes.grid(True, linestyle="--", alpha=0.7)

        # Add exact energy reference line if available
        exact_energies = {"H2": -1.137, "LiH": -8.000, "H2O": -76.00}
        if self.qnn.molecule in exact_energies:
            exact = exact_energies[self.qnn.molecule]
            self.energy_canvas.axes.axhline(
                y=exact,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Exact: {exact:.3f} Hartree",
            )
            self.energy_canvas.axes.legend()

        # Update current energy label
        if energy_history:
            self.current_energy_label.setText(f"{energy_history[-1]:.6f} Hartree")

        # Redraw the canvas
        self.energy_canvas.draw()

    def update_circuit_visualization(self):
        """Update the quantum circuit visualization."""
        if not self.qnn:
            return

        try:
            # Get circuit with resolved parameters
            circuit = self.qnn.get_circuit(resolved=True)

            # Clear previous figure
            self.circuit_canvas.axes.clear()

            # Draw circuit using matplotlib
            try:
                import cirq
                from cirq.contrib.svg import SVGCircuit

                # For now, just show a text representation in the plot
                self.circuit_canvas.axes.text(
                    0.5,
                    0.5,
                    str(circuit),
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=self.circuit_canvas.axes.transAxes,
                    fontsize=8,
                )
                self.circuit_canvas.axes.axis("off")
                self.circuit_canvas.draw()
            except ImportError:
                self.log_message(
                    "Warning: cirq.contrib.svg not available for circuit visualization"
                )

        except Exception as e:
            self.log_message(
                f"Warning: Could not update circuit visualization: {str(e)}"
            )

    def update_exact_energy(self):
        """Update the exact energy reference label."""
        if self.qnn is None:
            return

        exact_energies = {"H2": -1.137, "LiH": -8.000, "H2O": -76.00}

        if self.qnn.molecule in exact_energies:
            exact = exact_energies[self.qnn.molecule]
            self.exact_energy_label.setText(f"{exact:.6f} Hartree")
        else:
            self.exact_energy_label.setText("N/A")

    def on_save_model(self):
        """Save the current model to a file."""
        if not self.qnn:
            self.show_error("No model to save. Please create and train a model first.")
            return

        try:
            # Get save file path
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save QNN Model",
                "",
                "Pickle Files (*.pkl);;All Files (*)",
                options=options,
            )

            if file_path:
                # Add .pkl extension if not present
                if not file_path.endswith(".pkl"):
                    file_path += ".pkl"

                # Save model
                self.qnn.save_model(file_path)
                self.log_message(f"Model saved to {file_path}")

        except PermissionError:
            self.show_error(
                "Permission denied when saving file. Check your file permissions."
            )
        except Exception as e:
            self.show_error(f"Error saving model: {str(e)}")

    def on_load_model(self):
        """Load a model from a file."""
        try:
            # Get load file path
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load QNN Model",
                "",
                "Pickle Files (*.pkl);;All Files (*)",
                options=options,
            )

            if file_path:
                try:
                    # Load model
                    self.log_message(f"Loading model from {file_path}...")
                    self.qnn = MolecularQNN.load_model(file_path)

                    # Update UI
                    self.log_message("Model loaded successfully.")
                    self.log_message(f"Molecule: {self.qnn.molecule}")
                    self.log_message(f"Bond length: {self.qnn.bond_length} Å")
                    self.log_message(f"Ansatz type: {self.qnn.ansatz_type}")
                    self.log_message(f"Number of qubits: {self.qnn.n_qubits}")

                    if hasattr(self.qnn, "final_energy"):
                        self.log_message(f"Energy: {self.qnn.final_energy:.6f} Hartree")

                    # Update UI controls to match loaded model
                    try:
                        # Set molecule combo box
                        index = self.molecule_combo.findText(self.qnn.molecule)
                        if index >= 0:
                            self.molecule_combo.setCurrentIndex(index)

                        # Set bond length
                        self.bond_length_spin.setValue(self.qnn.bond_length)

                        # Set depth
                        if hasattr(self.qnn, "depth"):
                            self.depth_spin.setValue(self.qnn.depth)

                        # Set ansatz type
                        ansatz_display = self.qnn.ansatz_type.replace("_", " ").title()
                        index = self.ansatz_combo.findText(
                            ansatz_display, Qt.MatchContains
                        )
                        if index >= 0:
                            self.ansatz_combo.setCurrentIndex(index)
                    except Exception as e:
                        self.log_message(
                            f"Warning: Could not update all UI controls: {str(e)}"
                        )

                    # Update energy plot if available
                    if hasattr(self.qnn, "energy_history") and self.qnn.energy_history:
                        self.update_plot(self.qnn.energy_history)

                    # Update circuit visualization
                    self.update_circuit_visualization()

                    # Update exact energy if available
                    self.update_exact_energy()

                    # Update UI state
                    self.update_ui_state(is_loaded=True, is_trained=True)

                except Exception as e:
                    self.show_error(
                        f"Error loading model: {str(e)}\nThe file might be corrupted or incompatible."
                    )

        except Exception as e:
            self.show_error(f"Error selecting file: {str(e)}")

    def log_message(self, message):
        """
        Add a message to the log.

        Args:
            message (str): The message to log
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def show_error(self, message):
        """
        Show an error message.

        Args:
            message (str): The error message
        """
        self.log_message(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)
        # If we were training, reset UI state
        if (
            hasattr(self, "training_thread")
            and self.training_thread
            and self.training_thread.isRunning()
        ):
            self.update_ui_state(is_training=False)
            if hasattr(self, "qnn") and self.qnn:
                self.update_ui_state(is_loaded=True)
                if hasattr(self.qnn, "energy_history") and self.qnn.energy_history:
                    self.update_ui_state(is_trained=True)


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = QNNApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
