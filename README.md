# NOVA - Quantum Neural Network Framework

[![CI](https://github.com/nova-team/nova-qnn/workflows/CI/badge.svg)](https://github.com/nova-team/nova-qnn/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modern, professional Python package for **Quantum Neural Networks** focused on molecular energy estimation using variational quantum algorithms (VQAs).

## ✨ Features

- **🏗️ Modern Package Structure**: Professional src-layout with proper module organization
- **🎯 Multiple Ansatz Types**: Hardware-efficient, UCC, CHEA, HVA, symmetry-preserving circuits
- **⚡ Flexible Training**: Support for various optimizers (BFGS, L-BFGS-B, Nelder-Mead, Powell)
- **🔧 Hardware Abstraction**: Pluggable quantum backends (Cirq, Qiskit) with error mitigation
- **🖥️ Modern CLI**: Click-based command-line interface with comprehensive options
- **📊 GUI Applications**: PyQt5-based graphical interfaces for interactive exploration
- **🧪 Comprehensive Testing**: Full test suite with CI/CD pipeline
- **📦 Easy Installation**: pip-installable with optional extras for different features

## 🚀 Quick Start

### Installation

```bash
# Install the base package
pip install nova-qnn

# Or install with all optional dependencies
pip install "nova-qnn[all]"

# For development
git clone https://github.com/nova-team/nova-qnn.git
cd nova-qnn
pip install -e ".[dev]"
```

### Basic Usage

```python
from nova.core.qnn_molecular_energy import MolecularQNN

# Create a quantum neural network for H2 molecule
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    ansatz_type="hardware_efficient",
    depth=2
)

# Train the model
results = qnn.train(iterations=100, method="BFGS")

# Get the final energy estimate
energy = qnn.get_energy()
print(f"Final energy: {energy:.6f} Hartree")
```

### Command Line Interface

```bash
# Get help
nova --help

# Train a model
nova train --molecule H2 --depth 2 --iterations 100 --save my_model.pkl

# Load and inspect a saved model
nova load my_model.pkl

# Compare different ansatz types
nova compare --molecule H2 --iterations 50

# Launch GUI applications
nova-gui launcher
nova-gui compare
```

## 📖 Getting Started Guide

### 1. Understanding the Package Structure

NOVA is organized into focused modules:

```
src/nova/
├── core/           # Core QNN functionality and optimizers
├── ansatz/         # Quantum circuit ansatz implementations  
├── hardware/       # Hardware abstraction and quantum backends
├── transpiler/     # Circuit optimization and transpilation
├── mitigation/     # Quantum error mitigation strategies
├── cli/            # Command-line interface
└── gui/            # Graphical user interfaces
```

### 2. Training Your First Model

```python
from nova.core.qnn_molecular_energy import MolecularQNN

# Step 1: Create a QNN instance
qnn = MolecularQNN(
    molecule="H2",           # Molecule: H2, LiH, or H2O
    bond_length=0.74,        # Bond length in Angstroms
    ansatz_type="ucc",       # Ansatz: hardware_efficient, ucc, chea, hva, symmetry_preserving
    depth=3                  # Circuit depth
)

# Step 2: Train the model
print("Training quantum neural network...")
results = qnn.train(
    iterations=200,          # Number of optimization iterations
    method="BFGS",          # Optimization method
    verbose=True            # Show progress
)

# Step 3: Analyze results
print(f"\nTraining Results:")
print(f"Final Energy: {results['energy']:.6f} Hartree")
print(f"Training Time: {results['training_time']:.2f} seconds")
print(f"Converged: {results['success']}")

# Step 4: Save the trained model
qnn.save_model("h2_model.pkl")
```

### 3. Comparing Different Ansatz Types

```python
# Compare multiple ansatz types for the same molecule
results, fig = qnn.compare_ansatz_types(
    methods=["hardware_efficient", "ucc", "chea"],
    iterations=100,
    save_path="ansatz_comparison.png"
)

# Analyze which ansatz performed best
for ansatz, data in results.items():
    if ansatz != "exact_energy":
        print(f"{ansatz}: {data['final_energy']:.6f} Hartree")
```

### 4. Using Hardware Backends

```python
from nova.hardware.quantum_hardware_interface import CirqSimulatorBackend

# Create a quantum hardware backend
backend = CirqSimulatorBackend(name="cirq_simulator")

# Use it in your QNN
qnn = MolecularQNN(
    molecule="H2",
    bond_length=0.74,
    hardware_backend=backend
)

# Training will now use the specified backend
results = qnn.train(iterations=50)
```

### 5. Command Line Workflows

```bash
# Quick training session
nova train --molecule H2 --bond-length 0.74 --ansatz ucc --depth 3 --iterations 100 --plot

# Compare multiple ansatz types
nova compare --molecule LiH --iterations 75 --save comparison_results.png

# Get system information
nova info

# List available quantum backends
nova backends

# Run the test suite
nova test
```

### 6. GUI Applications

```bash
# Launch the main GUI selector
nova-gui launcher

# Launch ansatz comparison GUI directly
nova-gui compare

# Launch the main GUI interface
nova-gui main-gui
```

## 🛠️ Advanced Usage

### Custom Optimizers

```python
from nova.core.advanced_optimizers import create_optimizer

# Use advanced optimizers
optimizer = create_optimizer("gradient_free", method="nelder-mead")
qnn = MolecularQNN(molecule="H2", optimizer=optimizer)
```

### Error Mitigation

```python
from nova.mitigation.quantum_error_mitigation import ZeroNoiseExtrapolation

# Apply error mitigation
mitigation = ZeroNoiseExtrapolation()
qnn = MolecularQNN(
    molecule="H2", 
    error_mitigation=mitigation
)
```

### Circuit Transpilation

```python
from nova.transpiler.quantum_circuit_transpiler import CircuitTranspiler

# Optimize circuits for specific hardware
transpiler = CircuitTranspiler(optimization_level=3)
optimized_circuit = transpiler.transpile(qnn.get_circuit())
```

## 📦 Installation Options

### Basic Installation
```bash
pip install nova-qnn
```

### With Optional Dependencies
```bash
# GUI support
pip install "nova-qnn[gui]"

# Qiskit support  
pip install "nova-qnn[qiskit]"

# Enhanced Cirq features
pip install "nova-qnn[cirq]"

# Quantum chemistry (PySCF)
pip install "nova-qnn[pyscf]"

# Everything
pip install "nova-qnn[all]"
```

### Development Setup
```bash
git clone https://github.com/nova-team/nova-qnn.git
cd nova-qnn
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check src --fix
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"           # Skip slow tests
pytest -m integration          # Only integration tests
pytest tests/core/             # Test specific module

# With coverage
pytest --cov=nova --cov-report=html
```

## 📊 Available Molecules and Ansatz Types

### Supported Molecules
- **H2**: Hydrogen molecule (customizable bond length)
- **LiH**: Lithium hydride  
- **H2O**: Water molecule

### Ansatz Types
- **hardware_efficient**: Hardware-efficient ansatz with parameterized rotations
- **ucc**: Unitary Coupled Cluster ansatz
- **chea**: Chemistry-inspired ansatz
- **hva**: Hamiltonian Variational Ansatz
- **symmetry_preserving**: Particle-number conserving ansatz

### Optimization Methods
- **BFGS**: Broyden-Fletcher-Goldfarb-Shanno
- **L-BFGS-B**: Limited-memory BFGS with bounds
- **Nelder-Mead**: Simplex method
- **Powell**: Powell's conjugate direction method

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Make your changes and add tests
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [Cirq](https://quantumai.google/cirq) and [OpenFermion](https://quantumai.google/openfermion)
- Inspired by quantum machine learning research
- Thanks to all contributors and the quantum computing community

## 📚 Documentation and Support

- **Documentation**: [Coming soon]
- **Issues**: [GitHub Issues](https://github.com/nova-team/nova-qnn/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nova-team/nova-qnn/discussions)

---

*NOVA - Advancing quantum neural networks for molecular simulation* 🚀 