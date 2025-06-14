#!/usr/bin/env python3
"""
Very simple quantum circuit test using Cirq directly.
"""

import cirq
import matplotlib.pyplot as plt
import numpy as np

print("=== Simple Cirq Circuit Test ===")

# Create two qubits
q0, q1 = cirq.LineQubit.range(2)
print(f"Created qubits: {q0}, {q1}")

# Create a simple circuit
circuit = cirq.Circuit(
    cirq.H(q0),  # Hadamard on qubit 0
    cirq.CNOT(q0, q1),  # CNOT gate with control q0 and target q1
    cirq.measure(q0, q1, key="result"),  # Measure both qubits
)

print("\nCircuit:")
print(circuit)

# Simulate the circuit
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=100)
print("\nSimulation result:")
print(result)

# Get the measurement results
measurements = result.measurements["result"]
print("\nMeasurements:")
print(measurements)

# Count the occurrences of each outcome
unique, counts = np.unique(measurements, axis=0, return_counts=True)
print("\nOutcome counts:")
for outcome, count in zip(unique, counts):
    print(f"|{outcome[0]}{outcome[1]}⟩: {count} times")

# Plot results
plt.figure(figsize=(8, 4))
labels = [f"|{outcome[0]}{outcome[1]}⟩" for outcome in unique]
plt.bar(labels, counts)
plt.title("Bell State Measurement Results")
plt.ylabel("Count")
plt.savefig("bell_state_results.png")
print("\nPlot saved to bell_state_results.png")

print("\nSimple test completed successfully")
