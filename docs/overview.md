# Architecture Overview

NOVA is organized as a modular, src-layout Python package with clear separation of concerns.

## Package Structure

- `nova.core`: Core QNN logic (energy estimation, training loops, optimizers)
- `nova.ansatz`: Parameterized circuit families (hardware-efficient, UCC, CHEA, HVA, symmetry-preserving)
- `nova.hardware`: Backend abstraction and enhanced hardware utilities
- `nova.transpiler`: Circuit optimization and device-aware transpilation
- `nova.mitigation`: Error mitigation strategies (e.g., readout mitigation, ZNE)
- `nova.cli`: Command-line interface entry points
- `nova.gui`: PyQt-based interfaces for interactive exploration

## High-Level Flow

1. Build the molecular Hamiltonian and choose an ansatz.
2. Construct a parameterized circuit and a cost function (expectation value).
3. Optimize parameters with classical optimizers.
4. Execute on simulator or hardware via the hardware interface.
5. Optionally apply transpilation and error mitigation.
6. Analyze results and persist trained models.

## Key Design Principles

- Hardware portability via interfaces and transpilation
- Extensibility for new ansatz families and optimizers
- Reproducibility and benchmarking across backends
- Clean APIs for CLI, GUI, and programmatic use
