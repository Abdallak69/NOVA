#!/usr/bin/env python3
"""
Dedicated GUI for comparing different ansatz circuits in the QNN project.

This module provides a graphical interface for visualizing and comparing
the performance of different ansatz circuits for molecular energy estimation.
"""

import os
import sys

import cirq
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the QNN module
try:
    from nova.ansatz.ansatz_circuits import create_ansatz
    from nova.core.qnn_molecular_energy import MolecularQNN

    print("Successfully imported QNN modules")
except ImportError as e:
    print(f"Error importing QNN modules: {e}")
    raise


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding plots in the PyQt GUI."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class TrainingThread(QThread):
    """Thread for running QNN training in the background."""

    # Signal to update progress and results
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(self, molecule, bond_length, methods, depth, iterations):
        super().__init__()
        self.molecule = molecule
        self.bond_length = bond_length
        self.methods = methods
        self.depth = depth
        self.iterations = iterations

    def run(self):
        """Run the QNN training and comparison."""
        try:
            # Create the QNN
            qnn = MolecularQNN(
                molecule=self.molecule,
                bond_length=self.bond_length,
                ansatz_type="hardware_efficient",  # Default type, will be changed during comparison
                ansatz_kwargs={"depth": self.depth},
            )

            # Run the comparison
            results, _ = qnn.compare_ansatz_types(
                iterations=self.iterations,
                methods=self.methods,
                callback=self.update_progress,
            )

            # Emit finished signal with results
            self.finished_signal.emit(results)

        except Exception as e:
            print(f"Error during training: {e}")
            self.finished_signal.emit({"error": str(e)})

    def update_progress(self, progress_data):
        """Update the progress of the training."""
        self.update_signal.emit(progress_data)


class AnsatzComparisonGUI(QMainWindow):
    """Main window for the Ansatz Comparison GUI."""

    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Quantum Neural Network - Ansatz Comparison")
        self.setGeometry(100, 100, 1200, 800)

        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Create a splitter for control panel and results
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Create the control panel
        self.setup_control_panel()

        # Create the results panel
        self.setup_results_panel()

        # Add panels to splitter
        self.splitter.addWidget(self.control_panel)
        self.splitter.addWidget(self.results_panel)
        self.splitter.setSizes([400, 800])

        # Initialize training thread
        self.training_thread = None

        # Available methods
        self.available_methods = [
            "hardware_efficient",
            "ucc",
            "chea",
            "symmetry_preserving",
            "hva",
        ]

        # Method display names
        self.method_names = {
            "hardware_efficient": "Hardware-Efficient Ansatz",
            "ucc": "Unitary Coupled Cluster",
            "chea": "Custom Hardware-Efficient",
            "symmetry_preserving": "Symmetry-Preserving",
            "hva": "Hamiltonian Variational",
        }

        # Populate molecule selector
        self.molecule_selector.addItems(["H2", "LiH", "H2O"])

        # Populate method checkboxes
        for method in self.available_methods:
            checkbox = QCheckBox(self.method_names[method])
            checkbox.setChecked(True)  # Default to all methods selected
            self.method_checkboxes.append(checkbox)
            self.methods_layout.addWidget(checkbox)

        # Connect signals
        self.run_button.clicked.connect(self.start_comparison)
        self.stop_button.clicked.connect(self.stop_comparison)
        self.save_button.clicked.connect(self.save_results)
        self.clear_button.clicked.connect(self.clear_results)

        self.show()

    def setup_control_panel(self):
        """Set up the control panel with options for comparison."""
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)

        # Molecule selection group
        self.molecule_group = QGroupBox("Molecule")
        self.molecule_layout = QFormLayout()
        self.molecule_selector = QComboBox()
        self.bond_length = QDoubleSpinBox()
        self.bond_length.setRange(0.1, 5.0)
        self.bond_length.setValue(0.74)
        self.bond_length.setSingleStep(0.01)
        self.molecule_layout.addRow("Molecule:", self.molecule_selector)
        self.molecule_layout.addRow("Bond Length (Å):", self.bond_length)
        self.molecule_group.setLayout(self.molecule_layout)
        self.control_layout.addWidget(self.molecule_group)

        # Circuit options group
        self.circuit_group = QGroupBox("Circuit Options")
        self.circuit_layout = QFormLayout()
        self.circuit_depth = QSpinBox()
        self.circuit_depth.setRange(1, 10)
        self.circuit_depth.setValue(2)
        self.iterations = QSpinBox()
        self.iterations.setRange(10, 1000)
        self.iterations.setValue(50)
        self.circuit_layout.addRow("Circuit Depth:", self.circuit_depth)
        self.circuit_layout.addRow("Training Iterations:", self.iterations)
        self.circuit_group.setLayout(self.circuit_layout)
        self.control_layout.addWidget(self.circuit_group)

        # Methods group
        self.methods_group = QGroupBox("Ansatz Methods to Compare")
        self.methods_layout = QVBoxLayout()
        self.method_checkboxes = []
        self.methods_group.setLayout(self.methods_layout)
        self.control_layout.addWidget(self.methods_group)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Comparison")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.button_layout.addWidget(self.run_button)
        self.button_layout.addWidget(self.stop_button)
        self.control_layout.addLayout(self.button_layout)

        # Status
        self.status_group = QGroupBox("Status")
        self.status_layout = QVBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_layout.addWidget(self.status_label)
        self.status_group.setLayout(self.status_layout)
        self.control_layout.addWidget(self.status_group)

        # Add stretch to push everything up
        self.control_layout.addStretch()

    def setup_results_panel(self):
        """Set up the results panel with plots and data."""
        self.results_panel = QWidget()
        self.results_layout = QVBoxLayout(self.results_panel)

        # Create tabs for different views
        self.results_tabs = QTabWidget()

        # Energy convergence plot tab
        self.convergence_tab = QWidget()
        self.convergence_layout = QVBoxLayout(self.convergence_tab)
        self.convergence_canvas = MplCanvas(self, width=10, height=6)
        self.convergence_layout.addWidget(self.convergence_canvas)
        self.results_tabs.addTab(self.convergence_tab, "Energy Convergence")

        # Final energy comparison tab
        self.comparison_tab = QWidget()
        self.comparison_layout = QVBoxLayout(self.comparison_tab)
        self.comparison_canvas = MplCanvas(self, width=10, height=6)
        self.comparison_layout.addWidget(self.comparison_canvas)
        self.results_tabs.addTab(self.comparison_tab, "Energy Comparison")

        # Circuit visualization tab
        self.circuit_tab = QWidget()
        self.circuit_layout = QVBoxLayout(self.circuit_tab)
        self.circuit_selector = QComboBox()
        self.circuit_text = QTextEdit()
        self.circuit_text.setReadOnly(True)
        self.circuit_layout.addWidget(QLabel("Select Ansatz:"))
        self.circuit_layout.addWidget(self.circuit_selector)
        self.circuit_layout.addWidget(self.circuit_text)
        self.results_tabs.addTab(self.circuit_tab, "Circuit Details")

        # Results data tab
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        self.data_layout.addWidget(self.data_text)
        self.results_tabs.addTab(self.data_tab, "Raw Data")

        # Add the tabs to the results layout
        self.results_layout.addWidget(self.results_tabs)

        # Buttons for saving and clearing results
        self.results_button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Results")
        self.clear_button = QPushButton("Clear Results")
        self.results_button_layout.addWidget(self.save_button)
        self.results_button_layout.addWidget(self.clear_button)
        self.results_layout.addLayout(self.results_button_layout)

    def start_comparison(self):
        """Start the ansatz comparison."""
        # Get the selected molecule
        molecule = self.molecule_selector.currentText()
        bond_length = self.bond_length.value()

        # Get the selected methods
        selected_methods = []
        for i, checkbox in enumerate(self.method_checkboxes):
            if checkbox.isChecked():
                selected_methods.append(self.available_methods[i])

        if not selected_methods:
            QMessageBox.warning(
                self,
                "No Methods Selected",
                "Please select at least one ansatz method to compare.",
            )
            return

        # Get circuit options
        depth = self.circuit_depth.value()
        iterations = self.iterations.value()

        # Update status
        self.status_label.setText(f"Running comparison for {molecule}...")

        # Disable the run button and enable stop button
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Clear existing plots
        self.convergence_canvas.axes.clear()
        self.comparison_canvas.axes.clear()

        # Create and start the training thread
        self.training_thread = TrainingThread(
            molecule=molecule,
            bond_length=bond_length,
            methods=selected_methods,
            depth=depth,
            iterations=iterations,
        )
        self.training_thread.update_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.finish_comparison)
        self.training_thread.start()

    def update_progress(self, progress_data):
        """Update the progress of the comparison."""
        # Update the convergence plot
        self.convergence_canvas.axes.clear()

        # Plot the energy convergence for each method
        for method, data in progress_data.items():
            if method != "iteration":
                iterations = progress_data.get("iteration", [])
                energies = data.get("energies", [])
                if iterations and energies:
                    self.convergence_canvas.axes.plot(
                        iterations,
                        energies,
                        label=self.method_names.get(method, method),
                    )

        # Add exact energy if available
        if "exact_energy" in progress_data:
            exact_energy = progress_data["exact_energy"]
            max_iter = max(progress_data.get("iteration", [0]))
            if max_iter > 0:
                self.convergence_canvas.axes.axhline(
                    y=exact_energy, color="r", linestyle="--", label="Exact Energy"
                )

        # Format the plot
        self.convergence_canvas.axes.set_xlabel("Iteration")
        self.convergence_canvas.axes.set_ylabel("Energy (Hartree)")
        self.convergence_canvas.axes.set_title(
            f"Energy Convergence for {self.molecule_selector.currentText()} "
            f"(Bond Length: {self.bond_length.value()} Å)"
        )
        self.convergence_canvas.axes.legend()
        self.convergence_canvas.axes.grid(True)

        # Redraw the canvas
        self.convergence_canvas.draw()

        # Update status
        current_iteration = max(progress_data.get("iteration", [0]))
        total_iterations = self.iterations.value()
        self.status_label.setText(
            f"Running comparison... Iteration {current_iteration}/{total_iterations}"
        )

    def finish_comparison(self, results):
        """Handle the completion of the comparison."""
        # Re-enable the run button and disable stop button
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Check for errors
        if "error" in results:
            QMessageBox.critical(
                self, "Error", f"An error occurred: {results['error']}"
            )
            self.status_label.setText("Error during comparison")
            return

        # Update status
        self.status_label.setText("Comparison completed")

        # Plot final energy comparison
        self.plot_energy_comparison(results)

        # Update circuit details selector
        self.circuit_selector.clear()
        for method in results:
            if method != "exact_energy":
                self.circuit_selector.addItem(
                    self.method_names.get(method, method), method
                )

        # Connect circuit selector
        self.circuit_selector.currentIndexChanged.connect(self.show_circuit_details)

        # Show circuit details for the first method
        if self.circuit_selector.count() > 0:
            self.show_circuit_details(0)

        # Update raw data
        self.update_raw_data(results)

    def plot_energy_comparison(self, results):
        """Plot the final energy comparison."""
        self.comparison_canvas.axes.clear()

        methods = []
        final_energies = []

        for method, data in results.items():
            if method != "exact_energy" and "final_energy" in data:
                methods.append(self.method_names.get(method, method))
                final_energies.append(data["final_energy"])

        # Create bar chart
        bars = self.comparison_canvas.axes.bar(methods, final_energies)

        # Add exact energy if available
        if "exact_energy" in results:
            exact_energy = results["exact_energy"]
            self.comparison_canvas.axes.axhline(
                y=exact_energy, color="r", linestyle="--", label="Exact Energy"
            )

        # Format the plot
        self.comparison_canvas.axes.set_xlabel("Ansatz Method")
        self.comparison_canvas.axes.set_ylabel("Final Energy (Hartree)")
        self.comparison_canvas.axes.set_title(
            f"Final Energy Comparison for {self.molecule_selector.currentText()} "
            f"(Bond Length: {self.bond_length.value()} Å)"
        )
        self.comparison_canvas.axes.legend()

        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            self.comparison_canvas.axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.5f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

        # Redraw the canvas
        self.comparison_canvas.draw()

    def show_circuit_details(self, index):
        """Show details of the selected circuit."""
        if index < 0:
            return

        method = self.circuit_selector.itemData(index)

        try:
            # Create an example circuit
            qubits = 4  # Use 4 qubits for visualization
            depth = self.circuit_depth.value()

            if method == "hva":
                # For HVA, we need to provide Hamiltonian terms
                from cirq import LineQubit, Z

                q = LineQubit.range(qubits)
                ham_terms = [Z(q[i]) * Z(q[i + 1]) for i in range(qubits - 1)]
                ansatz = create_ansatz(
                    method, q, depth=depth, hamiltonian_terms=ham_terms
                )
            else:
                ansatz = create_ansatz(
                    method, cirq.LineQubit.range(qubits), depth=depth
                )

            circuit = ansatz.build_circuit()

            # Display circuit details
            circuit_details = f"## {self.method_names.get(method, method)}\n\n"
            circuit_details += f"Number of qubits: {qubits}\n"
            circuit_details += f"Circuit depth: {depth}\n"
            circuit_details += f"Parameter count: {ansatz.param_count()}\n\n"
            circuit_details += "Circuit structure:\n\n"
            circuit_details += str(circuit)

            self.circuit_text.setPlainText(circuit_details)

        except Exception as e:
            self.circuit_text.setPlainText(f"Error displaying circuit: {e}")

    def update_raw_data(self, results):
        """Update the raw data display."""
        data_text = "# Comparison Results\n\n"

        # Add molecule info
        data_text += f"## Molecule: {self.molecule_selector.currentText()}\n"
        data_text += f"Bond Length: {self.bond_length.value()} Å\n"
        data_text += f"Circuit Depth: {self.circuit_depth.value()}\n"
        data_text += f"Training Iterations: {self.iterations.value()}\n\n"

        # Add exact energy if available
        if "exact_energy" in results:
            data_text += f"Exact Energy: {results['exact_energy']:.8f} Hartree\n\n"

        # Add results for each method
        data_text += "## Method Results\n\n"
        for method, data in results.items():
            if method != "exact_energy":
                data_text += f"### {self.method_names.get(method, method)}\n"
                if "final_energy" in data:
                    data_text += f"Final Energy: {data['final_energy']:.8f} Hartree\n"
                    if "exact_energy" in results:
                        error = abs(data["final_energy"] - results["exact_energy"])
                        data_text += f"Absolute Error: {error:.8f} Hartree\n"
                if "param_count" in data:
                    data_text += f"Parameter Count: {data['param_count']}\n"
                if "iteration_time" in data:
                    data_text += f"Average Iteration Time: {data['iteration_time']:.4f} seconds\n"
                data_text += "\n"

        self.data_text.setPlainText(data_text)

    def stop_comparison(self):
        """Stop the current comparison."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.training_thread.wait()
            self.status_label.setText("Comparison stopped")

            # Re-enable the run button and disable stop button
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def save_results(self):
        """Save the results to a file."""
        # Get the current tab
        current_tab = self.results_tabs.currentWidget()

        if current_tab == self.convergence_tab or current_tab == self.comparison_tab:
            # Save the current plot
            canvas = (
                self.convergence_canvas
                if current_tab == self.convergence_tab
                else self.comparison_canvas
            )
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
            )
            if filename:
                canvas.fig.savefig(filename, dpi=300, bbox_inches="tight")
                QMessageBox.information(self, "Success", f"Plot saved to {filename}")
        elif current_tab == self.data_tab:
            # Save the raw data
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Data", "", "Text Files (*.txt);;Markdown Files (*.md)"
            )
            if filename:
                with open(filename, "w") as f:
                    f.write(self.data_text.toPlainText())
                QMessageBox.information(self, "Success", f"Data saved to {filename}")
        elif current_tab == self.circuit_tab:
            # Save the circuit details
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save Circuit Details",
                "",
                "Text Files (*.txt);;Markdown Files (*.md)",
            )
            if filename:
                with open(filename, "w") as f:
                    f.write(self.circuit_text.toPlainText())
                QMessageBox.information(
                    self, "Success", f"Circuit details saved to {filename}"
                )

    def clear_results(self):
        """Clear the current results."""
        # Clear plots
        self.convergence_canvas.axes.clear()
        self.convergence_canvas.draw()
        self.comparison_canvas.axes.clear()
        self.comparison_canvas.draw()

        # Clear circuit details
        self.circuit_selector.clear()
        self.circuit_text.clear()

        # Clear raw data
        self.data_text.clear()

        # Reset status
        self.status_label.setText("Ready")


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    _ = AnsatzComparisonGUI()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
