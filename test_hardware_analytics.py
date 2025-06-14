#!/usr/bin/env python3
"""
Test script for Quantum Hardware Analytics

This script demonstrates the use of the quantum hardware analytics module
with simplified mock hardware interfaces for testing purposes.
"""

import logging
import sys

import cirq
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Import the hardware analytics module
try:
    from quantum_hardware_analytics import (
        DeviceSelector,
        HardwareBenchmark,
        HardwareVisualizer,
    )
except ImportError:
    print(
        "Error importing quantum_hardware_analytics. Make sure the module is in your Python path."
    )
    sys.exit(1)

# Import the consolidated hardware backend interface
from quantum_hardware_interface import QuantumBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HardwareAnalytics:
    """Combined hardware analytics class for testing."""
    
    def __init__(self, hardware_interfaces):
        self.hardware_interfaces = hardware_interfaces
        self.benchmark = HardwareBenchmark()
        self.visualizer = HardwareVisualizer()
        self.selector = DeviceSelector()
        self.selector.set_available_hardware(hardware_interfaces)
    
    def run_benchmarks(self, custom_circuits=None, repetitions=1000):
        """Run benchmarks on all hardware interfaces."""
        return self.benchmark.run_benchmark(
            self.hardware_interfaces, 
            circuits=custom_circuits, 
            shots=repetitions
        )
    
    def analyze_circuit(self, circuit):
        """Analyze circuit requirements."""
        return self.selector.analyze_circuit(circuit)
    
    def score_hardware(self, requirements):
        """Score hardware based on requirements."""
        return self.selector.score_hardware(requirements)
    
    def select_best_hardware(self, circuit):
        """Select the best hardware for a circuit."""
        return self.selector.select_device(circuit)
    
    def visualize_scores(self, scores, show=True):
        """Visualize hardware scores."""
        return self.selector.visualize_scores(scores)
    
    def visualize_connectivity_all(self, show=True):
        """Visualize connectivity for all hardware."""
        # For simplicity, just visualize the first hardware interface
        if self.hardware_interfaces:
            self.visualizer.set_hardware_interface(self.hardware_interfaces[0])
            # The visualize_connectivity method doesn't take show parameter, so we'll just call it
            self.visualizer.visualize_connectivity()
        return None


# Mock Hardware Backends for testing (inheriting from QuantumBackend)
class MockQuantumBackendImpl(QuantumBackend):
    """A mock hardware backend for testing the analytics module."""

    def __init__(
        self, name, num_qubits=5, error_rates=None, connectivity=None, is_simulator=True
    ):
        super().__init__(name=name)
        self.num_qubits = num_qubits
        self.is_simulator = is_simulator

        # Default error rates if none provided
        if error_rates is None:
            self.error_rates = {
                "single_qubit": {i: 0.001 + 0.0005 * i for i in range(num_qubits)},
                "two_qubit": {
                    (i, i + 1): 0.01 + 0.002 * i for i in range(num_qubits - 1)
                },
                "readout": {i: 0.02 + 0.003 * i for i in range(num_qubits)},
            }
        else:
            self.error_rates = error_rates

        # Default connectivity if none provided (linear chain)
        if connectivity is None:
            self.connectivity_graph = {i: [i + 1] for i in range(num_qubits - 1)}
            # Add backward connections
            for i in range(1, num_qubits):
                if i - 1 not in self.connectivity_graph.get(i, []):
                    self.connectivity_graph.setdefault(i, []).append(i - 1)
        else:
            self.connectivity_graph = connectivity

    def get_device_properties(self):
        """Return mock device properties."""
        return {
            "name": self.name,
            "is_simulator": self.is_simulator,
            "num_qubits": self.num_qubits,
            "max_qubits": self.num_qubits,  # Add max_qubits
            "connectivity": self.connectivity_graph,
            "error_rates": self.error_rates,
            "basis_gates": ["x", "y", "z", "h", "cx", "cz", "rx", "ry", "rz"],
            "supports_pulse": False,
            "available_gates": [
                "x",
                "y",
                "z",
                "h",
                "cx",
                "cz",
                "rx",
                "ry",
                "rz",
            ],  # Add available_gates
        }

    def run_circuit(self, circuit, repetitions=1000, **kwargs):
        """Execute a circuit and return mock results."""
        # Simple mock execution - just simulate a Bell state result
        if repetitions <= 0:
            return {}

        # Count the number of qubits being measured
        num_measured_qubits = 0
        measured_qubits = []
        measurement_key = "result"  # Default key
        for op in circuit.all_operations():
            if cirq.is_measurement(op):
                measured_qubits = op.qubits
                num_measured_qubits = len(measured_qubits)
                measurement_key = op.key  # Get the actual key
                break

        # Generate mock results
        if num_measured_qubits == 0:
            # No measurements, maybe return state vector if supported?
            # For simplicity, return empty results dictionary
            return {
                "metadata": {
                    "execution_time": 0.01,
                    "backend_name": self.name,
                    "repetitions": repetitions,
                }
            }

        # For simplicity, let's just create a distribution biased toward 0s and 1s
        # with some noise based on the device's error rates
        avg_error = np.mean(list(self.error_rates["single_qubit"].values()))
        noise_factor = avg_error * 10  # Scale the noise effect

        # Generate random bitstrings based on noise
        results_array = np.random.rand(repetitions, num_measured_qubits) < (
            0.5
            + np.random.uniform(
                -noise_factor, noise_factor, size=(repetitions, num_measured_qubits)
            )
        )
        results_array = results_array.astype(np.int8)

        return {
            measurement_key: results_array,
            "metadata": {
                "execution_time": 0.01 + 0.0001 * repetitions,
                "backend_name": self.name,
                "repetitions": repetitions,
            },
        }

    # Add dummy statevector methods
    def supports_statevector(self) -> bool:
        return self.is_simulator  # Only simulators support statevector here

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        if not self.supports_statevector():
            raise NotImplementedError("Hardware mock does not support statevector.")
        # Return a dummy statevector
        num_qubits = len(list(circuit.all_qubits()))
        size = 2**num_qubits
        sv = np.random.rand(size) + 1j * np.random.rand(size)
        return sv / np.linalg.norm(sv)


def create_mock_backends():
    """Create a set of mock hardware backends for testing."""
    simple_device = MockQuantumBackendImpl(name="SimpleMockDevice", num_qubits=5)
    low_error_device = MockQuantumBackendImpl(
        name="LowErrorMockDevice",
        num_qubits=5,
        error_rates={
            "single_qubit": {i: 0.0005 + 0.0002 * i for i in range(5)},
            "two_qubit": {(i, i + 1): 0.005 + 0.001 * i for i in range(4)},
            "readout": {i: 0.01 + 0.001 * i for i in range(5)},
        },
    )
    large_device = MockQuantumBackendImpl(
        name="LargeMockDevice",
        num_qubits=20,
        error_rates={
            "single_qubit": {i: 0.002 + 0.0008 * i for i in range(20)},
            "two_qubit": {(i, i + 1): 0.02 + 0.003 * i for i in range(19)},
            "readout": {i: 0.03 + 0.004 * i for i in range(20)},
        },
    )
    grid_device = MockQuantumBackendImpl(
        name="GridMockDevice",
        num_qubits=9,
        connectivity={
            0: [1, 3],
            1: [0, 2, 4],
            2: [1, 5],
            3: [0, 4, 6],
            4: [1, 3, 5, 7],
            5: [2, 4, 8],
            6: [3, 7],
            7: [4, 6, 8],
            8: [5, 7],
        },
    )
    return [simple_device, low_error_device, large_device, grid_device]


def get_mock_devices():
    """Get mock devices for testing (alias for create_mock_backends)."""
    return create_mock_backends()


def create_test_circuit_for_analytics():
    """Create a test circuit for analytics testing."""
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.measure(*qubits, key="result")
    )
    return circuit


def test_benchmarking():
    """Test the benchmarking functionality of the Hardware Analytics module."""
    print("\n--- Testing Hardware Benchmarking ---")
    
    # Create an analytics instance with mock devices
    analytics = HardwareAnalytics(get_mock_devices())
    
    # Define a custom benchmark circuit
    ghz_qubits = cirq.LineQubit.range(4)
    ghz_circuit = cirq.Circuit(
        cirq.H(ghz_qubits[0]),
        cirq.CNOT(ghz_qubits[0], ghz_qubits[1]),
        cirq.CNOT(ghz_qubits[1], ghz_qubits[2]),
        cirq.CNOT(ghz_qubits[2], ghz_qubits[3])
    )
    
    # Run the benchmark
    results = analytics.run_benchmarks(
        custom_circuits={"custom_ghz": ghz_circuit},
        repetitions=100
    )
    
    # Assert that results are generated for all devices
    assert len(results) == len(get_mock_devices())
    assert "SimpleMockDevice" in results
    assert "custom_ghz" in results["SimpleMockDevice"]
    assert "transpiled_depth" in results["SimpleMockDevice"]["custom_ghz"]


def test_visualization():
    """Test the visualization capabilities of the Hardware Analytics module."""
    print("\n--- Testing Hardware Visualization ---")
    
    # Create an analytics instance and score devices
    analytics = HardwareAnalytics(get_mock_devices())
    circuit = create_test_circuit_for_analytics()
    requirements = analytics.analyze_circuit(circuit)
    scores = analytics.score_hardware(requirements)
    
    # Test plotting scores
    try:
        analytics.visualize_scores(scores, show=False)
        # The method doesn't return a figure, but should not raise an exception
        print("Score visualization completed successfully")
    except Exception as e:
        pytest.fail(f"Score visualization failed: {e}")
        
    # Test plotting connectivity
    try:
        analytics.visualize_connectivity_all(show=False)
        # The method doesn't return a figure, but should not raise an exception
        print("Connectivity visualization completed successfully")
    except Exception as e:
        pytest.fail(f"Connectivity visualization failed: {e}")


def test_device_selection():
    """Test the device selection functionality."""
    print("\n--- Testing Device Selection ---")
    
    analytics = HardwareAnalytics(get_mock_devices())
    circuit = create_test_circuit_for_analytics()
    
    # Select the best device
    selection = analytics.select_best_hardware(circuit)
    
    # Assert that a device was selected and has a reason
    assert "selected" in selection
    assert "reason" in selection
    assert selection["selected"] is not None


def main():
    """Run all hardware analytics tests."""
    print("=== Quantum Hardware Analytics Tests ===")

    try:
        # Test benchmarking
        test_benchmarking()

        # Test visualization
        test_visualization()

        # Test device selection
        test_device_selection()

        print("\nAll hardware analytics tests completed successfully!")

    except Exception as e:
        logger.error(f"Error in hardware analytics tests: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
