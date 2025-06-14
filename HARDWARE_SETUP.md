# Quantum Hardware Integration Guide

This guide explains how to configure and use the quantum hardware integration features of the QNN project for molecular energy estimation.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Supported Hardware Providers](#supported-hardware-providers)
4. [Configuration Setup](#configuration-setup)
5. [Using Hardware in Code](#using-hardware-in-code)
6. [Error Mitigation Techniques](#error-mitigation-techniques)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)

## Overview

The Quantum Neural Network (QNN) project now includes support for executing quantum circuits on real quantum hardware, enabling more realistic simulations and experimentation with actual quantum processors. This allows for:

- Running molecular energy estimations on quantum processors
- Benchmarking the performance of different ansatz types on real hardware
- Studying the effects of noise and error mitigation techniques
- Exploring the practicality of quantum neural networks for chemistry applications

## Prerequisites

To use quantum hardware integration, you need the following:

1. **Core Dependencies**:
   - All standard QNN dependencies: Cirq, NumPy, SciPy, etc.
   - Additional hardware-specific libraries (depending on provider)

2. **Provider-Specific Requirements**:
   - For Google Quantum Hardware: Valid Google Cloud credentials
   - For IBM Quantum Hardware: IBM Quantum account and authentication token
   - For other providers: Respective API keys or authentication credentials

## Supported Hardware Providers

### Google Quantum AI

Access to Google's quantum processors through the Cirq framework.

**Setup Requirements**:
- Google Cloud account with Quantum API access enabled
- Valid authentication credentials
- Cirq version 1.0 or higher

**Available Devices**:
- Weber processor (Sycamore-like)
- Sycamore processors
- Various simulator backends

### IBM Quantum

Access to IBM's quantum devices through the Qiskit interface.

**Setup Requirements**:
- IBM Quantum account (https://quantum-computing.ibm.com/)
- Generated API token
- Qiskit version 0.30 or higher

**Available Devices**:
- Various processors from 5 to 127 qubits
- QASM Simulator
- Statevector Simulator

### Local Simulators

High-performance local simulators that can mimic quantum hardware.

**Available Simulators**:
- Cirq simulator (default)
- Density matrix simulator (for noise modeling)
- Clifford simulator (for stabilizer states)

## Configuration Setup

### Configuration File

Create a file named `quantum_config.json` in the project root directory with the following structure:

```json
{
  "default_provider": "cirq_simulator",
  "providers": {
    "google": {
      "project_id": "your-google-project-id",
      "processor_id": "weber",
      "credentials_path": "/path/to/credentials.json"
    },
    "ibmq": {
      "token": "your-ibm-token",
      "hub": "ibm-q",
      "group": "open",
      "project": "main",
      "device": "ibmq_bogota"
    }
  },
  "error_mitigation": {
    "default_strategy": "readout_correction",
    "zero_noise_extrapolation": {
      "scale_factors": [1.0, 1.5, 2.0, 2.5]
    }
  }
}
```

### Environment Variables

Alternatively, you can use environment variables:

```bash
# Google Quantum
export GOOGLE_QUANTUM_PROJECT_ID="your-project-id"
export GOOGLE_QUANTUM_PROCESSOR_ID="weber"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# IBM Quantum
export IBMQ_TOKEN="your-ibm-token"
export IBMQ_DEVICE="ibmq_bogota"
```

## Using Hardware in Code

### Programmatic Usage

```python
from qnn_molecular_energy import MolecularQNN
from quantum_hardware import QuantumHardwareProvider

# Create a hardware provider
provider = QuantumHardwareProvider()

# List available backends
backends = provider.list_backends()
print("Available backends:", backends)

# Create a QNN with specific hardware backend
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    depth=2,
    ansatz_type="hardware_efficient",
    hardware_backend="ibmq_bogota",
    error_mitigation="readout_correction"
)

# Run with 1000 measurement shots
result = qnn.estimate_energy(shots=1000)
print(f"Energy: {result['energy']:.6f} Hartree")
print(f"Uncertainty: {result['uncertainty']:.6f}")

# Train the model on quantum hardware
results = qnn.train(
    iterations=50,
    shots=1000,
    optimizer="noise_aware"  # Use a noise-aware optimizer
)
```

### Through the CLI Interface

1. Launch the CLI interface:
   ```bash
   python cli_interface.py
   ```

2. Select option 4 "Run on quantum hardware"

3. Follow the interactive prompts to select:
   - QNN configuration (molecule, bond length, ansatz)
   - Hardware backend
   - Error mitigation strategy
   - Number of measurement shots
   - Optimization method

## Error Mitigation Techniques

### Available Techniques

1. **Readout Error Mitigation**
   - Corrects for measurement errors using calibration data
   - Most effective for errors that occur during measurement
   - Requires calibration data from the device

2. **Circuit Knitting**
   - Breaks large circuits into smaller ones that can be run more accurately
   - Reconnects results to get the full solution
   - Good for circuits that exceed the coherence time of the device

3. **Zero-Noise Extrapolation**
   - Runs the circuit at different noise levels
   - Extrapolates results to estimate zero-noise result
   - Works well for gate errors and decoherence

4. **Symmetry Verification**
   - Uses known symmetries of the problem to detect and correct errors
   - Particularly useful for molecular energy estimation with known symmetries
   - Can detect errors that break physical conservation laws

### Configuring Error Mitigation

In code:
```python
# Set up hardware with specific error mitigation
qnn.set_hardware_backend(
    "ibmq_bogota",
    error_mitigation="zero_noise_extrapolation",
    mitigation_kwargs={"scale_factors": [1.0, 1.5, 2.0]}
)
```

In CLI: Follow the prompts after selecting "Use error mitigation".

## Performance Considerations

### Shots vs. Precision

- More shots = better statistical precision but longer run time
- Rule of thumb: Energy precision scales as 1/√(shots)
- Recommended starting point: 1000 shots
- For high precision: 5000+ shots

### Circuit Depth Limitations

- Real hardware has coherence time limitations
- Keep circuit depth < 20-30 gates for best results on current NISQ devices
- Consider using shallower ansatz or fewer parameters on noisy hardware
- Hardware-efficient and symmetry-preserving ansatz typically perform better on real hardware

### Queuing and Execution Time

- Hardware access often involves queuing in a job system
- Jobs may wait minutes to hours depending on provider and load
- Consider using the `async_estimate_energy()` and `async_train()` methods 
- CLI/GUI operations will wait for results, but display progress

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify API keys and credentials are correctly set up
   - Check for expired tokens (IBM tokens expire every 30 days)
   - Ensure proper permissions are set for credential files

2. **Job Failures**
   - Circuits too large for the selected device
   - Device availability changed during submission
   - Check provider's status page for hardware maintenance

3. **Inconsistent Results**
   - High variability is expected on real hardware due to noise
   - Use more shots and error mitigation to improve consistency
   - Consider using a noise-aware optimizer

### Diagnostics

The system includes diagnostic tools to help identify issues:

```python
# Run hardware diagnostics
from quantum_hardware import hardware_diagnostics

# Check connectivity to providers
hardware_diagnostics.check_providers()

# Test a simple circuit on specific hardware
hardware_diagnostics.test_circuit("ibmq_bogota", num_qubits=2)

# Measure current noise levels
noise_report = hardware_diagnostics.measure_noise("ibmq_bogota")
print(noise_report)
```

For more detailed help, refer to the provider-specific documentation:
- [Google Quantum AI](https://quantumai.google/cirq)
- [IBM Quantum](https://quantum-computing.ibm.com/docs/)

---

By following this guide, you should be able to configure and use quantum hardware with the QNN project, opening up new possibilities for quantum chemistry simulations on actual quantum processors. 