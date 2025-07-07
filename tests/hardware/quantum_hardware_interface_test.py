#!/usr/bin/env python3
"""
Quantum Hardware Interface Test

This script demonstrates the usage of the standardized quantum hardware interface
to execute circuits on different quantum computing platforms.
"""

import time

import cirq
import matplotlib.pyplot as plt

from nova.hardware.quantum_hardware_enhanced import default_provider

# Import the hardware backend module and enhanced provider
from nova.hardware.quantum_hardware_interface import (
    CirqSimulatorBackend,  # Specific implementation
)

# Try to import Qiskit-related components if available
try:
    import qiskit

    # Need a Qiskit backend implementation if it exists in quantum_hardware_interface
    # Assuming QiskitSimulatorBackend exists for this example
    from nova.hardware.quantum_hardware_interface import QiskitSimulatorBackend

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def test_bell_state_cirq():
    """Test executing a Bell state circuit using the Cirq backend."""
    print("\n=== Testing Bell State with Cirq Backend ===")

    # Get the default Cirq interface via the enhanced provider
    try:
        cirq_backend = default_provider.get_backend("cirq_simulator")
    except Exception as e:
        print(f"Could not get cirq_simulator backend: {e}")
        return

    # Create a Bell state circuit
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key="result"),
    )

    print(f"Circuit:\n{circuit}")

    # Execute the circuit using the enhanced backend's method
    shots = 1000
    try:
        results_dict = cirq_backend.run_circuit(circuit, repetitions=shots)
        from nova.hardware.quantum_hardware_enhanced import EnhancedExecutionResult

        results = EnhancedExecutionResult(
            raw_results=results_dict, backend_name=cirq_backend.name
        )

        # Print results
        print(f"Execution time: {results.execution_time:.3f} seconds")
        print(f"Counts: {results.get_counts()}")

        # Plot results
        fig = results.plot_histogram()
        if fig:
            plt.savefig("bell_state_cirq.png")
            plt.close(fig)
            print("Plot saved to bell_state_cirq.png")

    except Exception as e:
        print(f"Error during Cirq execution: {e}")


def test_bell_state_qiskit():
    """Test executing a Bell state circuit using the Qiskit backend."""
    if not QISKIT_AVAILABLE:
        print("\n=== Qiskit is not available, skipping Qiskit test ===")
        return

    print("\n=== Testing Bell State with Qiskit Backend ===")

    # Register Qiskit backend if not already registered
    if "qiskit_simulator" not in default_provider.list_backends():
        try:
            # Assuming QiskitSimulatorBackend is defined in quantum_hardware_interface
            default_provider.register_backend(
                "qiskit_simulator", QiskitSimulatorBackend
            )
        except NameError:
            print("QiskitSimulatorBackend not defined, cannot register.")
            return
        except Exception as e:
            print(f"Failed to register Qiskit backend: {e}")
            return

    # Get the Qiskit backend
    try:
        qiskit_backend = default_provider.get_backend("qiskit_simulator")
    except Exception as e:
        print(f"Could not get qiskit_simulator backend: {e}")
        return

    # Create a Bell state circuit in Qiskit
    qc = qiskit.QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    print(f"Circuit:\n{qc}")

    # Execute the circuit - Need conversion or direct support
    shots = 1000
    try:
        # Currently, EnhancedQuantumBackend primarily handles Cirq circuits.
        # Qiskit execution would require circuit conversion or a Qiskit-native EnhancedBackend.
        # For now, simulate by calling the base backend directly if possible.
        if hasattr(qiskit_backend, "backend") and hasattr(
            qiskit_backend.backend, "run"
        ):  # Check if it has the underlying Qiskit backend object
            qiskit_base_backend = qiskit_backend.backend
            job = qiskit_base_backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()
            exec_time = result.time_taken
            print(f"Execution time: {exec_time:.3f} seconds")
            print(f"Counts: {counts}")

            # Plot results
            from qiskit.visualization import plot_histogram

            fig = plot_histogram(
                counts, title=f"Bell State Results - Qiskit ({shots} shots)"
            )
            fig.savefig("bell_state_qiskit.png")
            plt.close(fig)
            print("Plot saved to bell_state_qiskit.png")
        else:
            print("Cannot execute Qiskit circuit directly with current setup.")

    except Exception as e:
        print(f"Error during Qiskit execution: {e}")


def test_circuit_conversion():
    """Test converting circuits between Cirq and Qiskit formats (basic)."""
    if not QISKIT_AVAILABLE:
        print("\n=== Qiskit is not available, skipping conversion test ===")
        return

    print("\n=== Testing Circuit Conversion (Basic) ===")

    # Conversion logic is complex and often requires specific libraries (like qiskit-terra's converters)
    # This test will be a placeholder or use very basic manual conversion if available.

    print(
        "Note: Robust circuit conversion requires dedicated libraries and is not fully implemented here."
    )

    # Example: Basic Cirq to Qiskit (if Qiskit backend implements basic conversion)
    try:
        default_provider.get_backend("qiskit_simulator")
        default_provider.get_backend("cirq_simulator")

        # Create Cirq circuit
        qubits = cirq.LineQubit.range(2)
        cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))

        # Create Qiskit circuit
        if QISKIT_AVAILABLE:
            from qiskit import QuantumCircuit

        # Attempt conversion (this depends heavily on backend implementation)
        print("Attempting Cirq -> Qiskit conversion (likely placeholder)")
        # converted_qiskit = qiskit_backend_enhanced.convert_circuit(cirq_circuit, "cirq")
        # print(f"Converted to Qiskit:\n{converted_qiskit}")

        print("\nAttempting Qiskit -> Cirq conversion (likely placeholder)")
        # converted_cirq = cirq_backend_enhanced.convert_circuit(qiskit_circuit, "qiskit")
        # print(f"Converted to Cirq:\n{converted_cirq}")

    except Exception as e:
        print(f"Error during conversion test: {e}")


def test_create_noisy_simulator():
    """Test creating a noisy simulator using the enhanced provider framework."""
    print("\n=== Testing Noisy Simulation ===")

    # Create a depolarizing noise model for Cirq
    noise_strength = 0.01  # 1% noise
    cirq_noise_model = cirq.depolarize(p=noise_strength)

    # Define a constructor for the noisy backend
    def noisy_cirq_constructor(**kwargs):
        return CirqSimulatorBackend(
            name="noisy_cirq_sim", noise_model=cirq_noise_model, **kwargs
        )

    # Register the noisy backend
    noisy_backend_name = "noisy_cirq_simulator"
    default_provider.register_backend(noisy_backend_name, noisy_cirq_constructor)

    try:
        # Get the noisy and ideal backends
        noisy_backend = default_provider.get_backend(noisy_backend_name)
        ideal_backend = default_provider.get_backend("cirq_simulator")

        # Create a simple circuit
        qubits = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.measure(*qubits, key="result"),
        )

        # Execute on both simulators
        ideal_results_dict = ideal_backend.run_circuit(circuit, repetitions=1000)
        noisy_results_dict = noisy_backend.run_circuit(circuit, repetitions=1000)

        from nova.hardware.quantum_hardware_enhanced import EnhancedExecutionResult

        ideal_results = EnhancedExecutionResult(ideal_results_dict)
        noisy_results = EnhancedExecutionResult(noisy_results_dict)

        # Compare results
        print("Ideal simulator counts:", ideal_results.get_counts())
        print("Noisy simulator counts:", noisy_results.get_counts())

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ideal_counts = ideal_results.get_counts()
        noisy_counts = noisy_results.get_counts()
        all_keys = sorted(set(ideal_counts.keys()) | set(noisy_counts.keys()))

        ax1.bar(all_keys, [ideal_counts.get(k, 0) for k in all_keys])
        ax1.set_title("Ideal Counts")
        ax1.set_xlabel("Measurement")
        ax1.set_ylabel("Counts")

        ax2.bar(all_keys, [noisy_counts.get(k, 0) for k in all_keys])
        ax2.set_title(f"Noisy Simulator ({noise_strength * 100}% depolarizing)")
        ax2.set_xlabel("Measurement")

        plt.tight_layout()
        plt.savefig("noise_comparison.png")
        plt.close(fig)
        print("Noise comparison plot saved to noise_comparison.png")

    except Exception as e:
        print(f"Error during noisy simulation: {str(e)}")


def test_optimization_levels():
    """Test different transpilation optimization levels (via CircuitTranspiler)."""
    print("\n=== Testing Optimization Levels (using CircuitTranspiler) ===")

    try:
        from nova.transpiler.quantum_circuit_transpiler import (
            OptimizationLevel,
            create_circuit_transpiler,
        )
    except ImportError:
        print("CircuitTranspiler module not found. Skipping optimization test.")
        return

    # Create a more complex circuit
    num_qubits = 5
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    for i in range(num_qubits):
        circuit.append(cirq.X(qubits[i]))
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.X(qubits[i]))
    for i in range(num_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append(cirq.measure(*qubits, key="result"))

    print(f"Original circuit depth: {len(circuit)}")

    # Create transpiler for the default simulator
    try:
        transpiler = create_circuit_transpiler()
    except Exception as e:
        print(f"Could not create transpiler: {e}. Skipping test.")
        return

    # Try different optimization levels
    for level in OptimizationLevel:
        try:
            start_time = time.time()
            optimized = transpiler.transpile(circuit, optimization_level=level)
            end_time = time.time()

            print(f"\nOptimization level {level.name}:")
            print(f"  Optimized circuit depth: {len(optimized)}")
            print(f"  Optimization time: {(end_time - start_time) * 1000:.2f} ms")
        except Exception as e:
            print(f"\nError during {level.name} optimization: {e}")


def test_device_properties():
    """Test getting device properties from the enhanced provider."""
    print("\n=== Testing Device Properties ===")

    # List all available backends
    try:
        backend_names = default_provider.list_backends()
        print(f"Available backends: {backend_names}")
    except Exception as e:
        print(f"Could not list backends: {e}")
        return

    # Get properties for each backend
    for name in backend_names:
        try:
            backend = default_provider.get_backend(name)
            properties = backend.get_device_properties()

            print(f"\nProperties for {name}:")
            for key, value in properties.items():
                # Skip potentially large/complex fields for brevity
                if key not in [
                    "diagnostics",
                    "device_specification",
                    "error_rates",
                    "connectivity",
                ]:
                    print(f"  {key}: {value}")
                elif key == "connectivity":
                    print(f"  {key}: Present (details omitted)")
                elif key == "error_rates":
                    print(f"  {key}: Present (details omitted)")

        except Exception as e:
            print(f"\nCould not get properties for {name}: {e}")


def main():
    """Run all tests."""
    print("=== Quantum Hardware Interface & Enhanced Provider Tests ===")

    # List available backends
    try:
        print(f"Available backends: {default_provider.list_backends()}")
    except Exception as e:
        print(f"Could not list backends: {e}")
        return

    # Run tests
    test_bell_state_cirq()
    test_bell_state_qiskit()
    test_circuit_conversion()
    test_create_noisy_simulator()
    test_optimization_levels()
    test_device_properties()

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
