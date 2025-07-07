#!/usr/bin/env python3
"""
Minimal script to test just the basic QNN functionality.
"""

import re

import cirq
import numpy as np

from nova.core.qnn_molecular_energy import MolecularQNN

# Create a basic QNN
print("Creating a QNN for H2 molecule...")
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    depth=1,  # Minimize depth to avoid complexity
    ansatz_type="hardware_efficient",
)

print(f"QNN created with {qnn.n_qubits} qubits")
print(f"Molecule: {qnn.molecule}, Bond length: {qnn.bond_length} Å")

# Get the quantum circuit
try:
    print("\nTrying to get the quantum circuit...")
    circuit = qnn.get_circuit(resolved=False)
    print("Circuit obtained successfully:")
    print(circuit)

    # Count parameters
    circuit_str = str(circuit)
    params = re.findall(r"θ_(\d+)", circuit_str)
    unique_params = set(params)
    max_param_idx = max(map(int, unique_params)) if unique_params else -1
    param_count = max_param_idx + 1
    print(f"Circuit has approximately {param_count} parameters")
except Exception as e:
    print(f"Error getting circuit: {e}")
    param_count = 36  # Default based on previous error message

# Try to get the current energy (without training)
try:
    print("\nTrying to get the current energy...")
    energy = qnn.get_energy()
    print(f"Current energy estimate: {energy:.6f} Hartree")
except Exception as e:
    print(f"Error getting energy: {e}")

# Try with appropriate number of random parameters
try:
    print(f"\nTrying with {param_count} random parameters...")
    # Create random parameters based on detected count
    params = np.random.random(param_count) * 2 * np.pi
    print(f"Random parameters shape: {params.shape}")

    # Manually compute energy expectation if available
    if hasattr(qnn, "_energy_expectation"):
        try:
            energy = qnn._energy_expectation(params)
            print(f"Energy with random parameters: {energy:.6f} Hartree")
        except Exception as e:
            print(f"Error computing energy expectation: {e}")

            # If we get a specific parameter count error, try with that count
            if "Expected" in str(e) and "parameter values" in str(e):
                match = re.search(r"Expected (\d+) parameter values", str(e))
                if match:
                    expected_count = int(match.group(1))
                    print(f"Trying again with {expected_count} parameters...")
                    params = np.random.random(expected_count) * 2 * np.pi
                    try:
                        energy = qnn._energy_expectation(params)
                        print(
                            f"Energy with {expected_count} parameters: {energy:.6f} Hartree"
                        )
                    except Exception as e2:
                        print(f"Error with adjusted parameters: {e2}")
except Exception as e:
    print(f"Error with random parameters: {e}")

# Let's try a very specific approach - direct circuit simulation
try:
    print("\nTrying direct circuit simulation...")
    # Get a resolved circuit with random parameters
    resolved_circuit = qnn.get_circuit(resolved=True)
    print("Resolved circuit:")
    print(resolved_circuit)

    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(resolved_circuit)
    print("Circuit simulation successful")
    print(f"Final state vector has shape: {result.final_state_vector.shape}")
except Exception as e:
    print(f"Error in direct simulation: {e}")

print("\nMinimal test completed")
