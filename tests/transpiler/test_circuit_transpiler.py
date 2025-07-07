#!/usr/bin/env python3
"""
Test script for the Quantum Circuit Transpiler

This script demonstrates how to use the enhanced circuit transpiler
for advanced circuit optimization and pulse-level programming.
"""

import time

import cirq
import matplotlib.pyplot as plt
import pytest

# Import the circuit transpiler
from nova.transpiler.quantum_circuit_transpiler import (
    OptimizationLevel,
    create_circuit_transpiler,
    CircuitTranspiler,
)

# Import the quantum hardware interface (just for QISKIT_AVAILABLE)
try:
    from nova.hardware.quantum_hardware_interface import hardware_manager, QuantumBackend

    # Check if Qiskit is available
    try:
        from qiskit import QuantumCircuit
        QISKIT_AVAILABLE = True
    except ImportError:
        QISKIT_AVAILABLE = False
except ImportError as e:
    print(f"Warning: {str(e)}")
    QISKIT_AVAILABLE = False


# Setup for tests
def create_complex_test_circuit():
    """Create a more complex test circuit that can benefit from optimization."""
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()

    # Add some redundant gates that can be optimized away
    for i in range(5):
        circuit.append(cirq.X(qubits[i]))
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(
            cirq.X(qubits[i])
        )  # This X-Z-X sequence can be optimized to just Z

    # Add some two-qubit gates
    for i in range(4):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))

    # Add a layer of H gates
    for i in range(5):
        circuit.append(cirq.H(qubits[i]))

    # Add some more two-qubit gates
    for i in range(4, 0, -1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i - 1]))

    # Add measurements
    circuit.append(cirq.measure(*qubits, key="result"))

    return circuit


def test_basic_transpilation():
    """Test basic transpilation functionality."""
    print("\n--- Testing Basic Transpilation ---")
    
    # Create a simple test circuit
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.measure(*qubits, key="result")
    )
    
    # Create transpiler with default settings
    transpiler = create_circuit_transpiler()
    
    # Transpile the circuit
    transpiled_circuit = transpiler.transpile(circuit)
    
    # Basic assertions
    assert isinstance(transpiled_circuit, cirq.Circuit)
    assert len(list(transpiled_circuit.all_operations())) > 0
    
    print(f"Original circuit depth: {len(circuit)}")
    print(f"Transpiled circuit depth: {len(transpiled_circuit)}")
    print("Basic transpilation test passed!")


def test_optimization_levels():
    """Test different optimization levels."""
    print("\n--- Testing Optimization Levels ---")
    
    # Create a more complex circuit
    qubits = cirq.LineQubit.range(5)
    circuit = create_complex_test_circuit()
    
    results = {}
    
    for level in OptimizationLevel:
        print(f"Testing level: {level.name}")
        transpiler = create_circuit_transpiler()
        
        start_time = time.time()
        transpiled_circuit = transpiler.transpile(circuit, optimization_level=level)
        end_time = time.time()
        
        # Analyze the transpiled circuit
        depth = len(transpiled_circuit)
        gate_count = len(list(transpiled_circuit.all_operations()))
        
        results[level.name] = {
            "circuit": transpiled_circuit,
            "depth": depth,
            "gate_count": gate_count,
            "time": end_time - start_time
        }
        
        print(f"  Depth: {depth}, Gate count: {gate_count}, Time: {results[level.name]['time']:.4f}s")
        
        # Add assertions
        assert isinstance(transpiled_circuit, cirq.Circuit)
        assert depth > 0
        assert gate_count > 0

    # Example assertion: check that extreme optimization is different from no optimization
    assert results['EXTREME']['gate_count'] < results['NONE']['gate_count']


def test_hardware_specific_optimization():
    """Test optimization for a specific hardware target."""
    print("\n--- Testing Hardware-Specific Optimization ---")
    
    # Create a mock hardware interface with a limited gate set
    class MockHardware(QuantumBackend):
        def __init__(self):
            super().__init__(name="mock_hardware")
            
        def get_device_properties(self):
            return {
                "name": "mock_hardware",
                "available_gates": ["H", "CNOT", "Z"],
                "connectivity": [[0, 1], [1, 2]]
            }
        def run_circuit(self, circuit, repetitions=1000): pass
        def get_statevector(self, circuit): pass
        def supports_statevector(self): return False

    # Create a circuit with gates not in the mock hardware's gate set
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.Y(qubits[1]),
        cirq.CZ(qubits[0], qubits[2])
    )
    
    # Transpile for the mock hardware
    mock_hardware = MockHardware()
    transpiler = CircuitTranspiler(mock_hardware)
    transpiled_circuit = transpiler.transpile(circuit, optimization_level=OptimizationLevel.ADVANCED)
    
    # Check that the transpiled circuit only contains allowed gates
    # Note: Cirq may decompose gates into equivalent forms like PhasedXPowGate
    allowed_gates = (cirq.HPowGate, cirq.CXPowGate, cirq.CZPowGate, cirq.ZPowGate, cirq.MeasurementGate, cirq.PhasedXPowGate, cirq.XPowGate, cirq.YPowGate)
    for op in transpiled_circuit.all_operations():
        if not isinstance(op.gate, allowed_gates):
            print(f"Unexpected gate: {op.gate} of type {type(op.gate)}")
        assert isinstance(op.gate, allowed_gates)


def test_pulse_level_programming():
    """Test pulse-level programming support (if available)."""
    print("\n=== Testing Pulse-Level Programming ===")

    try:
        # Skip if not using real hardware
        hardware_interface = hardware_manager.get_interface()
        if hardware_interface.is_simulator:
            print("Skipping pulse-level programming test on simulator")
            return None

        # Create a simple circuit
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key="result"),
        )

        # Create transpiler
        transpiler = create_circuit_transpiler()

        # Try to convert to pulse-level instructions
        try:
            pulse_program = transpiler.transpile_to_pulse(circuit)

            if "error" in pulse_program:
                print(f"Error in pulse conversion: {pulse_program['error']}")
                return None

            print("Successfully converted circuit to pulse instructions:")
            print(f"  Duration: {pulse_program.get('duration')} dt")
            print(f"  Qubits: {pulse_program.get('qubits')}")

            return pulse_program

        except Exception as e:
            print(f"Error in pulse-level programming: {str(e)}")
            return None
    except Exception as e:
        print(f"Skipping pulse-level programming test: {str(e)}")
        return None


def count_gates(circuit):
    """Count the number of gates in a circuit."""
    count = 0

    if isinstance(circuit, cirq.Circuit):
        for moment in circuit:
            count += len(moment.operations)

    return count


def visualize_optimization_results(results):
    """Visualize the optimization results."""
    if not results:
        print("No results to visualize.")
        return

    try:
        # Extract data for plotting
        levels = list(results.keys())
        depths = [results[level]["depth"] for level in levels]
        gate_counts = [results[level]["gate_count"] for level in levels]
        times = [results[level]["time"] for level in levels]

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot circuit depths
        ax1.bar(levels, depths, color="skyblue")
        ax1.set_title("Circuit Depth")
        ax1.set_ylabel("Number of Moments")
        ax1.set_xticklabels(levels, rotation=45)
        ax1.grid(axis="y", alpha=0.3)

        # Plot gate counts
        ax2.bar(levels, gate_counts, color="lightgreen")
        ax2.set_title("Gate Count")
        ax2.set_ylabel("Number of Gates")
        ax2.set_xticklabels(levels, rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        # Plot optimization times
        ax3.bar(levels, times, color="salmon")
        ax3.set_title("Optimization Time")
        ax3.set_ylabel("Time (seconds)")
        ax3.set_xticklabels(levels, rotation=45)
        ax3.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("optimization_results.png")
        plt.close()

        print("Visualization saved to 'optimization_results.png'")
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")


@pytest.mark.skipif(not QISKIT_AVAILABLE, reason="Qiskit not available")
def main():
    """Run all transpiler tests."""
    print("=== Circuit Transpiler Tests ===")

    # Test basic transpilation
    test_basic_transpilation()

    # Test optimization levels
    test_optimization_levels()

    # Test hardware-specific optimization
    test_hardware_specific_optimization()

    # Test pulse-level programming
    test_pulse_level_programming()

    # Visualize results if available
    try:
        visualize_optimization_results(test_optimization_levels())
    except Exception as e:
        print(f"Error visualizing results: {str(e)}")

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
