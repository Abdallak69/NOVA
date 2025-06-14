#!/usr/bin/env python3
"""
Basic script to test QNN functionality with a simplified approach.
"""

import cirq
import numpy as np

from ansatz_circuits import create_ansatz

print("=== Basic QNN Test ===")

# Create qubits
qubits = cirq.LineQubit.range(4)
print(f"Created {len(qubits)} qubits")

# Create a simple ansatz circuit
ansatz = create_ansatz(
    "hardware_efficient",
    qubits,
    depth=1,
    rotation_gates="XYZ",
    entangle_pattern="linear",
)
print("Created hardware-efficient ansatz")

# Build the circuit
circuit = ansatz.build_circuit()
print("Circuit:")
print(circuit)

# Create random parameters
param_count = ansatz.param_count()
print(f"Ansatz has {param_count} parameters")
params = np.random.random(param_count) * 2 * np.pi

# Get symbols from the circuit
symbols = sorted(
    [s for s in cirq.parameter_names(circuit)], key=lambda x: int(x.replace("θ_", ""))
)
print(f"Found {len(symbols)} symbols in circuit")

# Create parameter mapping
param_mapping = {symbols[i]: params[i] for i in range(min(len(symbols), len(params)))}
print("Parameter mapping created")

# Assign parameters to the circuit using cirq's resolver
resolver = cirq.ParamResolver(param_mapping)
resolved_circuit = cirq.resolve_parameters(circuit, resolver)
print("\nResolved circuit:")
print(resolved_circuit)

# Simulate the circuit
simulator = cirq.Simulator()
result = simulator.simulate(resolved_circuit)
print("\nCircuit simulation successful")
print(f"Final state vector has shape: {result.final_state_vector.shape}")

# Create a simple Hamiltonian (Z_0 * Z_1)
hamiltonian = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
print(f"\nSimple Hamiltonian: {hamiltonian}")

# Calculate expectation value using the state vector
try:
    # Use the current Cirq API for expectation values
    expectation = cirq.expectation_from_state_vector(
        state_vector=result.final_state_vector,
        observable=hamiltonian,
        qubit_map={q: i for i, q in enumerate(qubits)},
    )
    print(f"Expectation value: {expectation.real}")
except Exception as e:
    print(f"Error with expectation calculation: {e}")

    # As a fallback, calculate it manually for Z tensors
    if isinstance(hamiltonian, cirq.PauliString):
        # For Z tensors, we can compute expectation by summing probabilities with appropriate signs
        state_vector = result.final_state_vector
        expectation = 0.0

        # Get the Pauli operations
        pauli_ops = list(hamiltonian.items())

        # Iterate through all computational basis states
        for i in range(len(state_vector)):
            # Convert index to binary representation
            binary = format(i, f"0{len(qubits)}b")

            # Calculate sign based on parity of relevant qubits
            sign = 1
            for qubit, pauli in pauli_ops:
                qubit_index = qubits.index(qubit)
                if str(pauli) == "Z" and binary[qubit_index] == "1":
                    sign *= -1

            # Add contribution to expectation value
            prob = abs(state_vector[i]) ** 2
            expectation += sign * prob

        print(f"Expectation value (manual calc): {expectation}")

# Try a simpler approach for individual Z expectation values
print("\nTrying individual Z expectation calculations:")
z0 = cirq.Z(qubits[0])
z1 = cirq.Z(qubits[1])

try:
    exp_z0 = cirq.expectation_from_state_vector(
        state_vector=result.final_state_vector,
        observable=z0,
        qubit_map={q: i for i, q in enumerate(qubits)},
    )
    exp_z1 = cirq.expectation_from_state_vector(
        state_vector=result.final_state_vector,
        observable=z1,
        qubit_map={q: i for i, q in enumerate(qubits)},
    )
    print(f"<Z0>: {exp_z0.real}")
    print(f"<Z1>: {exp_z1.real}")
    print("Note: <Z0*Z1> is not simply <Z0>*<Z1> unless states are uncorrelated")
except Exception as e:
    print(f"Error calculating individual expectations: {e}")

print("\nBasic test completed successfully")
