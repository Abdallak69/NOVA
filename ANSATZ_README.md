# Advanced Ansatz Circuits and Hardware Integration for Quantum Neural Networks

This document explains the sophisticated ansatz circuits and hardware integration capabilities available in this Quantum Neural Network (QNN) project for molecular energy estimation.

## Table of Contents
1. [Available Ansatz Types](#available-ansatz-types)
2. [Hardware Integration](#hardware-integration)
3. [Advanced Optimization Strategies](#advanced-optimization-strategies)
4. [Usage Examples](#usage-examples)
5. [Performance Considerations](#performance-considerations)

## Available Ansatz Types

### Hardware-Efficient Ansatz
The hardware-efficient ansatz is designed for near-term quantum devices with limited coherence times and gate fidelities.

**Key features:**
- Configurable rotation gates (X, Y, Z or combinations)
- Flexible entanglement patterns (linear, full, custom)
- Minimal circuit depth while maintaining expressibility
- Well-suited for NISQ (Noisy Intermediate-Scale Quantum) devices

### Unitary Coupled Cluster (UCC) Ansatz
A chemically inspired ansatz that preserves important symmetries of the electronic structure problem.

**Key features:**
- Includes single and double excitation operators
- Naturally preserves particle number and spin symmetries
- Based on the unitary coupled cluster formalism from quantum chemistry
- More accurate for molecular systems, but requires deeper circuits

### Custom Hardware-Efficient Ansatz (CHEA)
A customizable variant of the hardware-efficient ansatz with problem-specific improvements.

**Key features:**
- Combines hardware efficiency with chemical intuition
- Allows for custom gate patterns that respect the problem structure
- Maintains low circuit depth for NISQ compatibility
- Better performance than generic hardware-efficient ansatz in many cases

### Symmetry-Preserving Ansatz
Focuses on preserving important symmetries of the target Hamiltonian.

**Key features:**
- Enforces particle number conservation
- Optionally preserves spin symmetries
- Reduces the search space by focusing on physically relevant states
- Improves convergence for systems with strong symmetry constraints

### Hamiltonian Variational Ansatz (HVA)
Based on the structure of the problem Hamiltonian itself, applying layers of evolution under Hamiltonian terms.

**Key features:**
- Guided by the structure of the problem Hamiltonian
- Requires fewer parameters for a given accuracy
- Better captures the physics of the system
- Often achieves better results with shallower circuits

## Hardware Integration

This QNN implementation now supports running on actual quantum hardware through various providers. 

### Supported Hardware Backends

The system supports the following types of quantum hardware:

- **Cirq Simulators**: Local high-performance simulators
- **Google Quantum Hardware**: Connect to Google's quantum processors via Cirq
- **IBM Quantum Hardware**: Connect to IBM's quantum devices via Qiskit
- **Custom Hardware**: Interface with other quantum hardware via custom adapters

### Error Mitigation Strategies

When running on real quantum hardware, various error mitigation techniques are available:

1. **Readout Error Mitigation**: Corrects for measurement errors using calibration data
2. **Circuit Knitting**: Breaks large circuits into smaller ones that can be run more accurately
3. **Zero-Noise Extrapolation**: Runs circuits at different noise levels and extrapolates to zero noise
4. **Measurement Error Mitigation**: Mitigates errors in the measurement process

### Hardware Selection in CLI/GUI

Both the CLI and GUI interfaces allow users to:
- Select from available quantum hardware backends
- Configure error mitigation strategies
- Set the number of measurement shots
- Compare results between simulator and hardware runs

## Advanced Optimization Strategies

The QNN now includes sophisticated optimization strategies to improve training, especially on noisy hardware.

### Available Optimizers

1. **Standard BFGS**: Default optimizer for noiseless simulations
2. **Noise-Aware BFGS**: Adapted for noisy function evaluations by averaging multiple samples
3. **Gradient-Free Optimizers**: 
   - Nelder-Mead simplex method
   - Differential Evolution
   - Powell's method
4. **Adaptive Optimizer**: Dynamically switches between methods based on progress
5. **Parallel Tempering**: Runs multiple optimizations at different "temperatures" to avoid local minima

### Optimizer Selection

Users can select optimization strategies through:
- The CLI interface in the "Advanced optimizer testing" menu
- When running training on quantum hardware
- Programmatically when creating a QNN instance

### Comparing Optimizers

The system includes tools to compare the performance of different optimizers on:
- Standard test functions (Rosenbrock, quadratic)
- Actual QNN energy estimation for molecular systems
- With or without simulated noise

## Usage Examples

### Basic Usage with Hardware Integration

```python
from qnn_molecular_energy import MolecularQNN
from quantum_hardware import QuantumHardwareProvider

# Create a hardware provider
provider = QuantumHardwareProvider()

# List available backends
backends = provider.list_backends()
print("Available backends:", backends)

# Create a QNN with hardware backend
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    depth=2,
    ansatz_type="hardware_efficient",
    hardware_backend="cirq_simulator"  # or a real hardware backend
)

# Run energy estimation with shots
result = qnn.estimate_energy(shots=1000)
print(f"Energy: {result['energy']:.6f} Hartree")
print(f"Uncertainty: {result['uncertainty']:.6f}")
```

### Using Advanced Optimizers

```python
from qnn_molecular_energy import MolecularQNN
from advanced_optimizers import create_optimizer

# Create a noise-aware optimizer
optimizer = create_optimizer("noise_aware", averaging_samples=5)

# Create a QNN
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    depth=2,
    ansatz_type="ucc",
    optimizer=optimizer
)

# Train with the advanced optimizer
results = qnn.train(iterations=100, verbose=True)
print(f"Final energy: {results['energy']:.6f} Hartree")
```

### Comparing Multiple Ansatz Types on Hardware

```python
from qnn_molecular_energy import MolecularQNN
from advanced_optimizers import create_optimizer

# Create a gradient-free optimizer for hardware
optimizer = create_optimizer("gradient_free", method="nelder-mead")

# Create QNN
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    depth=2,
    hardware_backend="ibmq_qasm_simulator",
    optimizer=optimizer
)

# Compare different ansatz types
results, fig = qnn.compare_ansatz_types(
    methods=["hardware_efficient", "ucc", "hva"],
    iterations=50,
    shots=1000
)

# Display and save results
import matplotlib.pyplot as plt
plt.savefig("hardware_ansatz_comparison.png")
plt.show()
```

## Performance Considerations

### Hardware vs. Simulator Performance

When running on real quantum hardware:
- Expect higher energy variances due to hardware noise
- Consider using more measurement shots (1000+) to reduce statistical error
- Enable appropriate error mitigation techniques
- Use noise-aware or gradient-free optimizers

### Optimization Strategy Selection

Guidelines for choosing optimizers:
- **Noiseless simulation**: Standard BFGS or L-BFGS-B
- **Noisy simulation**: Noise-aware BFGS or Gradient-free methods
- **Real hardware**: Noise-aware BFGS for simple circuits, Gradient-free for complex circuits
- **Difficult landscapes**: Parallel Tempering or Adaptive optimizers

### Ansatz Selection for Hardware

Best practices for ansatz selection on real hardware:
- Use Hardware-Efficient or CHEA for NISQ devices with limited connectivity
- Use shallower circuits with fewer parameters
- Consider the native gate set of the target hardware
- Test multiple ansatz types and compare for your specific problem

---

By combining sophisticated ansatz circuits, hardware integration, and advanced optimization strategies, this QNN framework provides powerful tools for molecular energy estimation on both simulators and real quantum hardware. 