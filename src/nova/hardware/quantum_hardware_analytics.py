#!/usr/bin/env python3
"""
Quantum Hardware Analytics Module

This module provides tools for analyzing and comparing quantum hardware providers,
including benchmarking, visualization, and device selection capabilities.

Features:
- Benchmarking tools to compare different hardware providers
- Visualization of device properties, connectivity, and noise characteristics
- Device selection tools based on circuit requirements
"""

import logging
import time
from typing import Any, Dict, List, Optional

import cirq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Conditionally import Qiskit
try:
    import qiskit
    from qiskit import transpile as qiskit_transpile

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Import our quantum hardware interface and circuit transpiler
try:
    from nova.hardware.quantum_hardware_interface import (
        QuantumBackend as QuantumHardwareInterface,
        hardware_manager,
    )
    from nova.transpiler.quantum_circuit_transpiler import (
        CircuitTranspiler,
        OptimizationLevel,
        create_circuit_transpiler,
    )

    HARDWARE_INTERFACE_AVAILABLE = True
except ImportError:
    HARDWARE_INTERFACE_AVAILABLE = False

    # Create minimal versions for standalone usage
    class DummyHardwareManager:
        def get_interface(self, name=None):
            return None

    hardware_manager = DummyHardwareManager()
    HARDWARE_INTERFACE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# ===================== Benchmarking Tools =====================


class HardwareBenchmark:
    """
    Tool for benchmarking quantum hardware providers using standard test circuits.

    This class provides methods to run benchmark circuits on different hardware
    backends and compare their performance metrics.
    """

    def __init__(self):
        """Initialize the hardware benchmark tool."""
        self.results = {}
        self.benchmark_circuits = self._create_benchmark_circuits()

    def _create_benchmark_circuits(self) -> Dict[str, Any]:
        """
        Create a set of standard benchmark circuits.

        Returns:
            Dictionary of named benchmark circuits.
        """
        circuits = {}

        # Bell state circuit
        qubits = cirq.LineQubit.range(2)
        bell_circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key="result"),
        )
        circuits["bell_state"] = bell_circuit

        # GHZ state circuit (3 qubits)
        qubits = cirq.LineQubit.range(3)
        ghz_circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.CNOT(qubits[1], qubits[2]),
            cirq.measure(*qubits, key="result"),
        )
        circuits["ghz_state"] = ghz_circuit

        # Quantum Fourier Transform (4 qubits)
        qubits = cirq.LineQubit.range(4)
        qft_circuit = cirq.Circuit()
        for i in range(4):
            qft_circuit.append(cirq.H(qubits[i]))
            for j in range(i + 1, 4):
                qft_circuit.append(
                    cirq.CZ(qubits[i], qubits[j]) ** (1 / (2 ** (j - i)))
                )
        qft_circuit.append(cirq.measure(*qubits, key="result"))
        circuits["qft"] = qft_circuit

        return circuits

    def run_benchmark(
        self,
        hardware_interfaces: List[QuantumHardwareInterface],
        circuits: Optional[Dict[str, Any]] = None,
        shots: int = 1000,
        compile_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Run benchmark circuits on multiple hardware interfaces.

        Args:
            hardware_interfaces: List of hardware interfaces to benchmark
            circuits: Custom benchmark circuits (if None, use default benchmarks)
            shots: Number of shots for each circuit execution
            compile_only: Only compile circuits without execution (for transpiler benchmarking)

        Returns:
            Dictionary of benchmark results
        """
        benchmark_circuits = circuits or self.benchmark_circuits
        results = {}

        for hw_interface in hardware_interfaces:
            hw_name = hw_interface.name
            results[hw_name] = {}

            logger.info(f"Benchmarking {hw_name}...")

            # Create a transpiler for this hardware
            transpiler = CircuitTranspiler(hw_interface)

            for circuit_name, circuit in benchmark_circuits.items():
                logger.info(f"  Running {circuit_name} benchmark...")

                # Measure transpilation time
                transpile_start = time.time()
                transpiled_circuit = transpiler.transpile(
                    circuit,
                    optimization_level=OptimizationLevel.ADVANCED,
                    preserve_measurements=True,
                )
                transpile_time = time.time() - transpile_start

                # Record metrics
                circuit_metrics = {
                    "original_depth": len(circuit),
                    "transpiled_depth": len(transpiled_circuit),
                    "original_gate_count": self._count_gates(circuit),
                    "transpiled_gate_count": self._count_gates(transpiled_circuit),
                    "transpile_time": transpile_time,
                }

                # Execute the circuit if not compile_only
                if not compile_only:
                    try:
                        execution_start = time.time()
                        execution_result = hw_interface.run_circuit(
                            transpiled_circuit, repetitions=shots
                        )
                        execution_time = time.time() - execution_start

                        # Add execution metrics
                        circuit_metrics["execution_time"] = execution_time
                        circuit_metrics["results"] = execution_result

                        # Calculate fidelity if it's a known benchmark circuit
                        if circuit_name == "bell_state":
                            circuit_metrics["fidelity"] = (
                                self._calculate_bell_state_fidelity(execution_result)
                            )

                    except Exception as e:
                        logger.error(f"Error executing circuit on {hw_name}: {str(e)}")
                        circuit_metrics["execution_error"] = str(e)

                results[hw_name][circuit_name] = circuit_metrics

        self.results = results
        return results

    def _count_gates(self, circuit: cirq.Circuit) -> int:
        """Count the number of gates in a circuit."""
        count = 0
        for moment in circuit:
            count += len(moment.operations)
        return count

    def _calculate_bell_state_fidelity(self, results: Dict[str, int]) -> float:
        """
        Calculate the fidelity of Bell state results.

        For a perfect Bell state, we expect only '00' and '11' with equal probability.

        Args:
            results: Counts dictionary from executing a Bell state circuit

        Returns:
            Estimated fidelity
        """
        total_shots = sum(results.values())
        # For Bell state, only '00' and '11' should appear with equal probability
        correct_counts = results.get("00", 0) + results.get("11", 0)
        return correct_counts / total_shots if total_shots > 0 else 0.0

    def compare_results(
        self, metric: str = "transpiled_depth"
    ) -> Dict[str, List[float]]:
        """
        Compare a specific metric across all benchmarked hardware.

        Args:
            metric: The metric to compare (e.g., 'transpiled_depth', 'execution_time')

        Returns:
            Dictionary mapping hardware names to lists of metric values
        """
        if not self.results:
            logger.warning("No benchmark results available. Run benchmarks first.")
            return {}

        comparison = {}

        for hw_name, hw_results in self.results.items():
            comparison[hw_name] = []

            for circuit_name, metrics in hw_results.items():
                if metric in metrics:
                    comparison[hw_name].append(metrics[metric])
                else:
                    comparison[hw_name].append(None)

        return comparison

    def plot_comparison(
        self,
        metric: str = "transpiled_depth",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Plot comparison of a specific metric across all benchmarked hardware.

        Args:
            metric: The metric to compare
            title: Custom title for the plot
            save_path: Path to save the plot image (if None, display instead)
        """
        comparison = self.compare_results(metric)

        if not comparison:
            logger.warning("No data to plot. Run benchmarks first.")
            return

        circuit_names = list(self.benchmark_circuits.keys())
        hw_names = list(comparison.keys())

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Set width of bars
        bar_width = 0.8 / len(hw_names)

        # Set positions of bars on x-axis
        indices = np.arange(len(circuit_names))

        # Create bars
        for i, (hw_name, values) in enumerate(comparison.items()):
            positions = indices + i * bar_width
            ax.bar(positions, values, bar_width, label=hw_name)

        # Add labels and title
        ax.set_xlabel("Benchmark Circuit")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(
            title or f"Comparison of {metric.replace('_', ' ').title()} Across Hardware"
        )
        ax.set_xticks(indices + bar_width * (len(hw_names) - 1) / 2)
        ax.set_xticklabels(circuit_names)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


# ===================== Visualization Tools =====================


class HardwareVisualizer:
    """
    Tools for visualizing quantum hardware characteristics.

    This class provides methods to visualize device connectivity,
    noise properties, and other hardware characteristics.
    """

    def __init__(self, hardware_interface: Optional[QuantumHardwareInterface] = None):
        """
        Initialize the hardware visualizer.

        Args:
            hardware_interface: The quantum hardware interface to visualize
        """
        self.hardware_interface = hardware_interface
        self.device_properties = {}

        if hardware_interface:
            self.device_properties = hardware_interface.get_device_properties()

    def set_hardware_interface(self, hardware_interface: QuantumHardwareInterface):
        """
        Set or change the hardware interface to visualize.

        Args:
            hardware_interface: The quantum hardware interface to visualize
        """
        self.hardware_interface = hardware_interface
        self.device_properties = hardware_interface.get_device_properties()

    def visualize_connectivity(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        layout: str = "spring",
    ):
        """
        Visualize the qubit connectivity graph of the device.

        Args:
            title: Custom title for the plot
            save_path: Path to save the plot image (if None, display instead)
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        """
        if not self.device_properties:
            logger.warning(
                "No device properties available. Set a hardware interface first."
            )
            return

        # Get connectivity information
        connectivity = self.device_properties.get("connectivity", {})

        if not connectivity:
            logger.warning("No connectivity information available for this device.")
            return

        # Create a graph
        G = nx.Graph()

        # Add nodes (qubits)
        num_qubits = self.device_properties.get("num_qubits", 0)
        if num_qubits > 0:
            G.add_nodes_from(range(num_qubits))
        else:
            # Try to infer from connectivity
            all_qubits = set()
            for source, targets in connectivity.items():
                all_qubits.add(source)
                all_qubits.update(targets)
            G.add_nodes_from(all_qubits)

        # Add edges (connections)
        for source, targets in connectivity.items():
            for target in targets:
                G.add_edge(source, target)

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Get error rates for node colors if available
        error_rates = self.device_properties.get("error_rates", {})
        single_qubit_errors = error_rates.get("single_qubit", {})

        # Generate node colors based on error rates
        node_colors = []
        if single_qubit_errors:
            vmin = min(single_qubit_errors.values()) if single_qubit_errors else 0
            vmax = max(single_qubit_errors.values()) if single_qubit_errors else 1

            for node in G.nodes():
                error = single_qubit_errors.get(node, vmin)
                # Higher error = redder color
                # Lower error = greener color
                node_colors.append(error)
        else:
            node_colors = "skyblue"

        # Draw the graph
        nodes = nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, cmap=plt.cm.RdYlGn_r, node_size=500
        )
        edges = nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
        labels = nx.draw_networkx_labels(G, pos, font_size=10)

        # Add a colorbar if we have error data
        if isinstance(node_colors, list) and len(node_colors) > 0:
            plt.colorbar(nodes, label="Single-qubit error rate")

        plt.title(title or f"Qubit Connectivity for {self.hardware_interface.name}")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Connectivity visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_noise_heatmap(
        self,
        error_type: str = "single_qubit",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize noise characteristics as a heatmap.

        Args:
            error_type: Type of error to visualize ('single_qubit', 'two_qubit', 'readout')
            title: Custom title for the plot
            save_path: Path to save the plot image (if None, display instead)
        """
        if not self.device_properties:
            logger.warning(
                "No device properties available. Set a hardware interface first."
            )
            return

        # Get error rates
        error_rates = self.device_properties.get("error_rates", {})

        if not error_rates:
            logger.warning("No error rate information available for this device.")
            return

        # Get the specific error type
        if error_type == "single_qubit":
            errors = error_rates.get("single_qubit", {})
            if not errors:
                logger.warning("No single-qubit error rates available.")
                return

            # Create a simple heatmap for single-qubit errors
            num_qubits = self.device_properties.get(
                "num_qubits", max(errors.keys()) + 1
            )
            error_matrix = np.zeros(num_qubits)

            for qubit, error in errors.items():
                if isinstance(qubit, int) and qubit < num_qubits:
                    error_matrix[qubit] = error

            plt.figure(figsize=(12, 6))
            plt.bar(range(num_qubits), error_matrix, color="skyblue")
            plt.xlabel("Qubit Index")
            plt.ylabel("Error Rate")
            plt.title(
                title or f"Single-Qubit Error Rates for {self.hardware_interface.name}"
            )
            plt.xticks(range(num_qubits))
            plt.grid(axis="y", alpha=0.3)

        elif error_type == "two_qubit":
            errors = error_rates.get("two_qubit", {})
            if not errors:
                logger.warning("No two-qubit error rates available.")
                return

            # Create a heatmap for two-qubit errors
            num_qubits = self.device_properties.get("num_qubits", 0)
            if num_qubits == 0:
                # Try to infer from error data
                all_qubits = set()
                for pair in errors.keys():
                    if isinstance(pair, tuple) and len(pair) == 2:
                        all_qubits.add(pair[0])
                        all_qubits.add(pair[1])
                num_qubits = max(all_qubits) + 1 if all_qubits else 0

            if num_qubits == 0:
                logger.warning("Could not determine number of qubits.")
                return

            # Initialize error matrix
            error_matrix = np.zeros((num_qubits, num_qubits))

            # Fill in known error rates
            for pair, error in errors.items():
                if isinstance(pair, tuple) and len(pair) == 2:
                    q1, q2 = pair
                    if q1 < num_qubits and q2 < num_qubits:
                        error_matrix[q1, q2] = error
                        error_matrix[q2, q1] = error  # Symmetric

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(error_matrix, cmap="RdYlGn_r", interpolation="nearest")
            plt.colorbar(label="Error Rate")
            plt.xlabel("Qubit Index")
            plt.ylabel("Qubit Index")
            plt.title(
                title or f"Two-Qubit Error Rates for {self.hardware_interface.name}"
            )
            plt.xticks(range(num_qubits))
            plt.yticks(range(num_qubits))

        elif error_type == "readout":
            errors = error_rates.get("readout", {})
            if not errors:
                logger.warning("No readout error rates available.")
                return

            # Create a simple heatmap for readout errors
            num_qubits = self.device_properties.get(
                "num_qubits", max(errors.keys()) + 1
            )
            error_matrix = np.zeros(num_qubits)

            for qubit, error in errors.items():
                if isinstance(qubit, int) and qubit < num_qubits:
                    error_matrix[qubit] = error

            plt.figure(figsize=(12, 6))
            plt.bar(range(num_qubits), error_matrix, color="salmon")
            plt.xlabel("Qubit Index")
            plt.ylabel("Readout Error Rate")
            plt.title(
                title or f"Readout Error Rates for {self.hardware_interface.name}"
            )
            plt.xticks(range(num_qubits))
            plt.grid(axis="y", alpha=0.3)

        else:
            logger.warning(f"Unsupported error type: {error_type}")
            return

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Noise heatmap saved to {save_path}")
        else:
            plt.show()

        plt.close()


# ===================== Device Selection Tools =====================


class DeviceSelector:
    """
    Tool for selecting optimal quantum hardware based on circuit requirements.

    This class provides methods to analyze circuits and recommend the most
    suitable hardware backend based on various criteria.
    """

    def __init__(self):
        """Initialize the device selector."""
        self.available_hardware = []

    def set_available_hardware(
        self, hardware_interfaces: List[QuantumHardwareInterface]
    ):
        """
        Set the list of available hardware interfaces.

        Args:
            hardware_interfaces: List of available hardware interfaces
        """
        self.available_hardware = hardware_interfaces

    def analyze_circuit(self, circuit: cirq.Circuit) -> Dict[str, Any]:
        """
        Analyze a circuit to determine its requirements.

        Args:
            circuit: The quantum circuit to analyze

        Returns:
            Dictionary of circuit requirements
        """
        requirements = {
            "num_qubits": 0,
            "gate_types": set(),
            "two_qubit_gates": 0,
            "depth": len(circuit),
            "total_gates": 0,
            "connectivity_requirements": set(),
        }

        # Analyze qubits and gates
        all_qubits = set()
        connectivity_pairs = set()

        for moment in circuit:
            for op in moment.operations:
                # Count qubits
                qubits = op.qubits
                all_qubits.update(qubits)

                # Count gate types
                gate_name = type(op.gate).__name__ if op.gate else "Measurement"
                requirements["gate_types"].add(gate_name)

                # Count two-qubit gates and track connectivity
                if len(qubits) == 2:
                    requirements["two_qubit_gates"] += 1
                    q1, q2 = qubits
                    # For cirq LineQubit objects, extract the x value
                    if hasattr(q1, "x") and hasattr(q2, "x"):
                        connectivity_pairs.add((q1.x, q2.x))
                        connectivity_pairs.add((q2.x, q1.x))  # Add both directions

                requirements["total_gates"] += 1

        requirements["num_qubits"] = len(all_qubits)
        requirements["connectivity_requirements"] = connectivity_pairs

        return requirements

    def score_hardware(self, circuit_requirements: Dict[str, Any]) -> Dict[str, float]:
        """
        Score available hardware based on circuit requirements.

        Args:
            circuit_requirements: Dictionary of circuit requirements from analyze_circuit

        Returns:
            Dictionary mapping hardware names to scores (higher is better)
        """
        if not self.available_hardware:
            logger.warning(
                "No hardware interfaces available. Set available hardware first."
            )
            return {}

        scores = {}

        for hw_interface in self.available_hardware:
            hw_name = hw_interface.name

            # Get device properties
            device_props = hw_interface.get_device_properties()

            # Initialize score
            scores[hw_name] = 100.0  # Start with perfect score

            # Check qubit count
            num_qubits = device_props.get("num_qubits", 0)
            required_qubits = circuit_requirements["num_qubits"]

            if num_qubits < required_qubits:
                # Not enough qubits - disqualify
                scores[hw_name] = 0.0
                continue

            # Slight penalty for having many more qubits than needed (might indicate a larger, noisier system)
            if num_qubits > 2 * required_qubits:
                scores[hw_name] -= 5.0

            # Check gate set
            basis_gates = set(device_props.get("basis_gates", []))
            required_gates = circuit_requirements["gate_types"]

            # Simplified check - in a real implementation, would map circuit gates to device basis gates
            if len(required_gates - basis_gates) > 0:
                # Some gates not directly supported - penalty
                scores[hw_name] -= 10.0 * len(required_gates - basis_gates)

            # Check connectivity
            connectivity = device_props.get("connectivity", {})
            required_connectivity = circuit_requirements["connectivity_requirements"]

            if required_connectivity:
                # Count how many required connections are missing
                missing_connections = 0
                for q1, q2 in required_connectivity:
                    if q1 not in connectivity or q2 not in connectivity.get(q1, []):
                        missing_connections += 1

                # Penalty for missing connections (each requires SWAP gates)
                scores[hw_name] -= 5.0 * missing_connections

            # Check error rates
            error_rates = device_props.get("error_rates", {})

            # Penalty based on average single-qubit errors
            single_qubit_errors = error_rates.get("single_qubit", {})
            if single_qubit_errors:
                avg_error = sum(single_qubit_errors.values()) / len(single_qubit_errors)
                scores[hw_name] -= 50.0 * avg_error  # Assuming errors are in [0,1]

            # Penalty based on two-qubit errors (weighted more heavily)
            two_qubit_errors = error_rates.get("two_qubit", {})
            if two_qubit_errors and circuit_requirements["two_qubit_gates"] > 0:
                avg_error = sum(two_qubit_errors.values()) / len(two_qubit_errors)
                scores[hw_name] -= (
                    100.0 * avg_error * circuit_requirements["two_qubit_gates"]
                )

            # Ensure score is non-negative
            scores[hw_name] = max(0.0, scores[hw_name])

        return scores

    def select_device(self, circuit: cirq.Circuit) -> Dict[str, Any]:
        """
        Select the best hardware device for a given circuit.

        Args:
            circuit: The quantum circuit to run

        Returns:
            Dictionary with selected hardware and reasoning
        """
        # Analyze circuit
        requirements = self.analyze_circuit(circuit)

        # Score hardware
        scores = self.score_hardware(requirements)

        if not scores:
            return {
                "selected": None,
                "scores": {},
                "requirements": requirements,
                "reason": "No hardware available to evaluate.",
            }

        # Find best hardware
        best_hardware = max(scores.items(), key=lambda x: x[1])
        best_name, best_score = best_hardware

        # Get the actual hardware interface
        selected_interface = next(
            (hw for hw in self.available_hardware if hw.name == best_name), None
        )

        return {
            "selected": selected_interface,
            "scores": scores,
            "requirements": requirements,
            "reason": f"Selected {best_name} with score {best_score:.2f}",
        }

    def visualize_scores(
        self,
        scores: Dict[str, float],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize hardware scores.

        Args:
            scores: Dictionary mapping hardware names to scores
            title: Custom title for the plot
            save_path: Path to save the plot image (if None, display instead)
        """
        if not scores:
            logger.warning("No scores to visualize.")
            return

        # Sort hardware by score
        sorted_hardware = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        hw_names = [hw[0] for hw in sorted_hardware]
        hw_scores = [hw[1] for hw in sorted_hardware]

        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(hw_names, hw_scores, color="skyblue")

        # Add score values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.xlabel("Hardware Provider")
        plt.ylabel("Score (higher is better)")
        plt.title(title or "Hardware Selection Scores")
        plt.xticks(rotation=45)
        plt.ylim(0, max(hw_scores) * 1.1 if hw_scores else 100)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Score visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()  # Simple function to get all available hardware interfaces


def get_all_hardware_interfaces() -> List[QuantumHardwareInterface]:
    """
    Get all available hardware interfaces.

    Returns:
        List of available hardware interfaces
    """
    if not HARDWARE_INTERFACE_AVAILABLE:
        logger.warning("Hardware interface module not available.")
        return []

    # This is a placeholder - in a real implementation would get all available interfaces
    interfaces = []

    # Get default interface
    default_interface = hardware_manager.get_interface()
    if default_interface:
        interfaces.append(default_interface)

    # In a real implementation, would get other available interfaces
    # Example:
    # for name in hardware_manager.get_available_interface_names():
    #     interfaces.append(hardware_manager.get_interface(name))

    return interfaces


# Example usage functions


def example_benchmark():
    """Run an example hardware benchmark."""
    interfaces = get_all_hardware_interfaces()

    if not interfaces:
        logger.warning("No hardware interfaces available for benchmarking.")
        return

    benchmark = HardwareBenchmark()
    results = benchmark.run_benchmark(interfaces, compile_only=True)

    # Plot comparison of transpiled circuit depth
    benchmark.plot_comparison(
        "transpiled_depth",
        title="Comparison of Transpiled Circuit Depth",
        save_path="benchmark_depth.png",
    )

    return results


def example_visualization():
    """Run an example hardware visualization."""
    interfaces = get_all_hardware_interfaces()

    if not interfaces:
        logger.warning("No hardware interfaces available for visualization.")
        return

    # Visualize the first available interface
    visualizer = HardwareVisualizer(interfaces[0])

    # Visualize connectivity
    visualizer.visualize_connectivity(save_path="connectivity.png")

    # Visualize noise
    visualizer.visualize_noise_heatmap(
        error_type="single_qubit", save_path="single_qubit_errors.png"
    )

    return "Visualization examples saved as PNG files."


def example_device_selection():
    """Run an example device selection."""
    interfaces = get_all_hardware_interfaces()

    if not interfaces:
        logger.warning("No hardware interfaces available for selection.")
        return

    # Create a test circuit
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.CNOT(qubits[2], qubits[3]),
        cirq.measure(*qubits, key="result"),
    )

    # Select the best device
    selector = DeviceSelector()
    selector.set_available_hardware(interfaces)
    result = selector.select_device(circuit)

    # Visualize scores
    if result["scores"]:
        selector.visualize_scores(
            result["scores"], save_path="device_selection_scores.png"
        )

    return result


# Main function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run example functions if this file is executed directly
    print("Running hardware analytics examples...")

    try:
        benchmark_results = example_benchmark()
        print("Benchmark examples completed.")
    except Exception as e:
        print(f"Error in benchmark examples: {str(e)}")

    try:
        viz_result = example_visualization()
        print("Visualization examples completed.")
    except Exception as e:
        print(f"Error in visualization examples: {str(e)}")

    try:
        selection_result = example_device_selection()
        print("Device selection examples completed.")
    except Exception as e:
        print(f"Error in device selection examples: {str(e)}")

    print("All examples completed.")
