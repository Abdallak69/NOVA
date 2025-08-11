# Frequently Asked Questions (FAQ)

## What is NOVA and what problem does it solve?
NOVA is a Python framework for variational quantum algorithms (VQAs) that estimates molecular ground‑state energies using quantum neural networks. It provides expressive ansatz circuits, robust classical optimizers, and hardware integration to run on simulators or real devices.

## Which molecules are supported?
Currently: H₂, LiH, and H₂O. You can add more by constructing or importing a molecular Hamiltonian and using it with the `MolecularQNN`. Active‑space and larger systems are planned (see Roadmap in the README).

## What differentiates NOVA from other frameworks?
- Specialized, chemistry‑aware ansatz families (hardware‑efficient, UCC, CHEA, HVA, symmetry‑preserving)
- Flexible hardware abstraction (Cirq/Qiskit) and transpilation
- Built‑in error‑mitigation strategies
- Modern CLI and GUI applications for interactive exploration

## Which quantum hardware or simulators can I use?
- Cirq simulators (local)
- Interfaces to Google/IBM hardware via Cirq/Qiskit backends
- Additional providers (IonQ, Rigetti, AWS Braket, Azure Quantum) are planned, together with unified configuration via environment variables and config files (see `QUANTUM_HARDWARE_INTERFACE.md`).

## How do I choose an ansatz or optimization method?
- Start on simulators with a hardware‑efficient ansatz
- For higher accuracy, try symmetry‑preserving or HVA
- Optimizers available: BFGS, L‑BFGS‑B, Nelder‑Mead, Powell, and gradient‑free/adaptive methods (see `OPTIMIZER_GUIDE.md`)

## How do I install NOVA?
- Base: `pip install nova-qnn`
- With extras: `pip install "nova-qnn[gui]"`, `"nova-qnn[qiskit]"`, `"nova-qnn[cirq]"`, `"nova-qnn[pyscf]"`, or `"nova-qnn[all]"`
- Requires Python ≥ 3.8 (see `pyproject.toml`)

## How do I run the CLI and GUIs?
- CLI examples: `nova --help`, `nova train ...`, `nova compare ...`, `nova backends`, `nova info`
- GUIs: `nova-gui launcher`, `nova-gui compare`, `nova-gui main-gui`

## Can I use NOVA with automatic differentiation libraries?
NOVA currently focuses on classical optimizers and finite‑difference/gradient‑free approaches. Deep integration with PyTorch/TensorFlow/JAX and parameter‑shift gradients are on the roadmap to enable differentiable hybrid models.

## Does NOVA perform error mitigation?
Yes. Techniques include zero‑noise extrapolation and readout error mitigation. See `HARDWARE_INTEGRATION_README.md` and `ANSATZ_README.md` for details.

## Where can I find tutorials and examples?
- Read the Docs: `https://nova-qnn.readthedocs.io` (Tutorials → Quickstart, Hardware Execution)
- Examples folder: `examples/` (notebooks for quickstart and hardware execution). Additional notebooks for hybrid autodiff training and visualisation/benchmarking are planned.

## How can I contribute or report a bug?
Read `CONTRIBUTING.md`. Use GitHub Issues for bugs/feature requests and GitHub Discussions for Q&A. See `SUPPORT.md` for support routes and security reporting.

## How should I cite NOVA?
See `CITATION.cff` (also visible via GitHub’s “Cite this repository” when published).

## What license governs the project?
Apache‑2.0. See `LICENSE`.
