# Quantum Circuit Transpiler Guide

This guide explains how to use the enhanced quantum circuit transpiler, which provides advanced optimization techniques, hardware-specific optimizations, and pulse-level programming support.

## Overview

The `quantum_circuit_transpiler.py` module extends the basic circuit transpilation capabilities provided by quantum hardware vendors with more sophisticated optimization techniques. The transpiler is designed to:

1. Optimize circuits with different levels of aggressiveness
2. Apply hardware-specific optimizations based on device connectivity and noise profiles
3. Support pulse-level programming on compatible hardware

## Getting Started

### Basic Usage

```python
from quantum_circuit_transpiler import create_circuit_transpiler, OptimizationLevel
import cirq

# Create a circuit
qubits = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(qubits[0]),
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.measure(*qubits, key='result')
)

# Create a transpiler with the default hardware interface
transpiler = create_circuit_transpiler()

# Transpile the circuit with basic optimization
optimized_circuit = transpiler.transpile(
    circuit, 
    optimization_level=OptimizationLevel.BASIC
)

# Print the original and optimized circuits
print("Original circuit:")
print(circuit)
print("\nOptimized circuit:")
print(optimized_circuit)
```

### Optimization Levels

The transpiler supports different optimization levels, from no optimization to experimental techniques:

- `OptimizationLevel.NONE`: No optimization
- `OptimizationLevel.BASIC`: Basic gate cancellation and merging
- `OptimizationLevel.INTERMEDIATE`: Layout optimization and gate decomposition
- `OptimizationLevel.ADVANCED`: Full optimization including noise-aware optimizations
- `OptimizationLevel.EXTREME`: Experimental optimization techniques (may be slower)

Example:
```python
# Transpile with advanced optimization
advanced_circuit = transpiler.transpile(
    circuit, 
    optimization_level=OptimizationLevel.ADVANCED,
    noise_aware=True
)
```

### Hardware-Specific Optimization

The transpiler can optimize circuits for specific quantum hardware by considering device connectivity and noise characteristics:

```python
# Get a transpiler for a specific hardware interface
qiskit_transpiler = create_circuit_transpiler('qiskit')

# Transpile with hardware-specific optimization
hardware_optimized = qiskit_transpiler.transpile(
    circuit,
    optimization_level=OptimizationLevel.ADVANCED,
    noise_aware=True,  # Enable noise-aware optimization
    target_gates=['x', 'sx', 'rz', 'cx']  # Specify target gate set
)
```

### Pulse-Level Programming

For hardware that supports it, the transpiler can convert circuits to pulse-level instructions:

```python
# Convert a circuit to pulse-level instructions
pulse_program = transpiler.transpile_to_pulse(circuit)

# Check if conversion was successful
if "error" not in pulse_program:
    print(f"Pulse program duration: {pulse_program.get('duration')} dt")
    print(f"Qubits: {pulse_program.get('qubits')}")
else:
    print(f"Error: {pulse_program['error']}")
```

## Advanced Features

### Custom Optimization Plugins

You can register custom optimization plugins to extend the transpiler:

```python
def my_custom_optimization(circuit, optimization_level):
    # Apply custom optimization
    # ...
    return optimized_circuit

# Register the plugin
transpiler.register_optimization_plugin(my_custom_optimization)
```

### Caching

The transpiler automatically caches the results of transpilation to avoid redundant work:

```python
# Clear the cache if needed
transpiler.clear_cache()
```

## Comparing Optimization Techniques

The test script (`test_circuit_transpiler.py`) demonstrates how to compare different optimization levels and techniques:

```bash
python test_circuit_transpiler.py
```

This will run a series of tests that compare different optimization levels and generate visualizations of the results.

## Future Improvements

Planned improvements for the circuit transpiler include:

1. Support for more advanced circuit synthesis techniques
2. Integration with machine learning for adaptive optimization
3. Enhanced pulse-level control for better hardware performance
4. More comprehensive noise modeling and mitigation
5. Support for additional quantum hardware platforms

## Troubleshooting

### Common Issues

1. **Unsupported circuit type**: The transpiler supports Cirq and Qiskit circuits. Make sure you're using one of these formats.
2. **Missing hardware support**: Not all operations are supported on all hardware. Check your target device's capabilities.
3. **Pulse programming errors**: Pulse-level programming is an advanced feature and not all hardware supports it.

### Getting Help

If you encounter issues with the transpiler, check the logs for detailed error messages. The transpiler uses Python's logging module to provide information about the transpilation process.

## API Reference

For a complete API reference, see the docstrings in the `quantum_circuit_transpiler.py` file. 