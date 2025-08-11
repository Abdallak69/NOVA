# NOVA - Quantum Neural Network Framework

[![CI](https://github.com/nova-team/nova-qnn/workflows/CI/badge.svg)](https://github.com/nova-team/nova-qnn/actions)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modern, professional Python package for **Quantum Neural Networks** focused on molecular energy estimation using variational quantum algorithms (VQAs).

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Quick Start](#quick-start)
- [Getting Started Guide](#getting-started-guide)
- [Advanced Usage](#advanced-usage)
- [Installation Options](#installation-options)
- [Testing](#testing)
- [Available Molecules and Ansatz Types](#available-molecules-and-ansatz-types)
- [Community Discussions](#community-discussions)
- [Challenges](#challenges)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Documentation and Support](#documentation-and-support)

## 🧠 Motivation

Modern quantum chemistry quickly becomes intractable on classical hardware as molecular size grows. NOVA exists to make hybrid quantum‑classical workflows practical for ground‑state energy estimation today, on both simulators and near‑term (noisy) quantum devices. It unifies three pillars under one consistent API:

- Expressive, chemistry‑aware ansatz families for VQE‑style training
- Robust classical optimizers designed for noisy, non‑convex landscapes
- Portable hardware integration with transpilation and error‑mitigation

The goal is to help researchers and practitioners prototype, compare, and deploy QNN/VQE approaches for molecular systems with reproducible results and hardware realism.

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

## 💬 Community Discussions

Join the conversation on [GitHub Discussions](https://github.com/nova-team/nova-qnn/discussions) to ask questions, propose ideas, and share results. Before posting, search existing topics to avoid duplicates. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚧 Challenges

Implementing practical quantum neural networks for chemistry involves several real‑world challenges:

- Noise and decoherence: Real hardware introduces gate and readout errors that bias energy estimates.
- Barren plateaus: Gradients can vanish in deeper circuits, slowing or stalling training.
- Limited shot budgets: Objective evaluations are expensive; sample efficiency is critical.
- Hardware constraints: Device connectivity and native gate sets require careful transpilation and mapping.
- Error mitigation overhead: Techniques like ZNE and readout mitigation improve accuracy but increase runtime.
- Initialization sensitivity: Poor parameter initializations can trap optimization in bad regions.
- Reproducibility: Results may vary across providers, backends, and noise models; consistent logging and seeds matter.
- Scalability: Moving beyond small molecules requires smarter ansatz design and resource‑aware compilation.

See `OPTIMIZER_GUIDE.md`, `CIRCUIT_TRANSPILER_GUIDE.md`, and `QUANTUM_HARDWARE_INTERFACE.md` for deeper discussion and mitigations.

## 🗺️ Roadmap

Planned enhancements aligned with the current architecture:

- **Quantum hardware back‑ends**: Extend beyond Cirq/Qiskit to include IonQ, Rigetti and cloud providers via AWS Braket/Azure Quantum. Provide a unified hardware‑abstraction layer with backend selection via configuration file and environment variables. Improve device discovery and capability introspection.
- **Automatic differentiation & hybrid models**: Integrate with PyTorch, TensorFlow and JAX to offer differentiable NOVA layers and parameter‑shift gradients for hybrid classical–quantum training. Ship examples and unit tests for end‑to‑end training.
- **More molecular systems & Hamiltonians**: Add small organic molecules (e.g., BeH₂, CH₄) and expose a clean API to load custom Hamiltonians (OpenFermion/PySCF pathways), including active‑space helpers and geometry import.
- **Additional ansatz families**: Implement periodic/plane‑wave or momentum‑space ansatz, ADAPT‑VQE, and machine‑learned ansätze compatible with the existing `create_ansatz` factory and transpiler.
- **Noise‑aware optimisation**: Add Bayesian optimisation and SPSA/shot‑frugal gradient methods; first‑class utilities for shot‑averaging and batched evaluations; noise‑aware gradient descent building on existing noise‑aware optimizers.
- **Visualization and analysis tools**: Provide modules and notebooks to visualise convergence metrics, energy landscapes, and circuit depth vs. accuracy, plus utilities to plot and compare optimisation trajectories across backends.
- **Pre‑trained models and benchmarks**: Publish pre‑trained parameter sets for benchmark molecules/ansatz types and maintain reproducible benchmark suites; make results downloadable alongside `benchmark_results/` artifacts.
- **Cross‑platform packaging**: Distribute conda packages (conda‑forge feedstock) in addition to PyPI; build wheels for common architectures (x86‑64, Apple Silicon).
- **Containerisation**: Provide Docker images (e.g., `ghcr.io/nova-team/nova-qnn`) with optional extras for tutorials, CI and cloud runs.
- **User interface enhancements**: Expand PyQt5 GUI with interactive circuit visualisation, parameter sliders and real‑time convergence plots; offer an optional web‑based GUI (Streamlit) for zero‑install usage.
- **Datasets and ML pipelines**: Introduce datasets and training pipelines for classification/regression tasks using hybrid models, beyond molecular energy estimation.

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

- Me 
- Built on top of [Cirq](https://quantumai.google/cirq) and [OpenFermion](https://quantumai.google/openfermion)
- Inspired by my quantum machine learning research and general interest
- Thanks to all contributors and the quantum computing community

## 📚 Documentation and Support

- **Documentation**: [NOVA Documentation on Read the Docs](https://nova-qnn.readthedocs.io)
- **FAQ**: See the [FAQ](https://nova-qnn.readthedocs.io/en/latest/faq.html)
- **Issues**: [GitHub Issues](https://github.com/nova-team/nova-qnn/issues) [Coming Soon]
- **Discussions**: [GitHub Discussions](https://github.com/nova-team/nova-qnn/discussions) [Coming Soon]
- **Releases**: [GitHub Releases](https://github.com/nova-team/nova-qnn/releases)
- **Code of Conduct**: See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- **Support**: See [SUPPORT.md](SUPPORT.md)
- **Citation**: See [CITATION.cff](CITATION.cff)

---

*NOVA - Advancing quantum neural networks for molecular simulation* 🚀 