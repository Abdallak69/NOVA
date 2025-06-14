# Quantum Neural Network Framework

A research-oriented Python toolkit for building, training, and benchmarking **variational quantum algorithms** (VQAs) for molecular-energy estimation.

It provides:

* Ready-made ansatz circuits (hardware-efficient, UCC, CHEA, HVA, symmetry-preserving …).
* A flexible `MolecularQNN` class that wraps training, energy evaluation, hardware back-ends and error-mitigation in one object.
* A hardware abstraction layer (`quantum_hardware_interface.py`) with pluggable back-ends (Cirq simulator by default, Qiskit and others optional).
* An enhanced provider (`quantum_hardware_enhanced.py`) that adds logging, retry logic, circuit validation and a registry of custom back-ends / error-mitigation strategies.
* A circuit-transpiler with multiple optimisation levels and hardware-aware passes.
* CLI and PyQt5 GUI launchers for interactive experimentation.
* A comprehensive pytest suite plus pre-commit / ruff for code-quality enforcement.

---

## Quick-start (TL;DR)

```bash
# 1. Clone & enter the project
$ git clone https://github.com/your-org/quantum-qnn.git
$ cd quantum-qnn

# 2. Create Python 3.11 virtual-env & install deps
$ python -m venv qnn_env
$ source qnn_env/bin/activate
$ pip install -r requirements.txt

# 3. Run the test-suite (optional but recommended)
$ pytest -q

# 4. Launch the GUI
$ python ansatz_comparison_gui.py

# 5. Or play with the CLI
$ python cli_interface.py
```

---

## Repository layout

```
.
├── ansatz_circuits.py          # All ansatz circuit classes & helpers
├── qnn_molecular_energy.py     # MolecularQNN class: training, energy eval …
├── quantum_hardware_interface.py   # Minimal hardware back-end ABC + Cirq impl.
├── quantum_hardware_enhanced.py    # Provider with logging, retries, mitigation
├── quantum_circuit_transpiler.py   # Optimisation / hardware-aware transpiler
├── cli_interface.py             # Interactive command-line launcher
├── ansatz_comparison_gui.py     # PyQt5 GUI for side-by-side ansatz runs
├── tests/                       # Pytest suite (unit + integration)
├── docs/ / *.md                 # Design notes & deep-dives
└── requirements.txt             # Pinned run-time dependencies
```

### Key entry points

| File | What it does |
|------|--------------|
| `qnn_molecular_energy.py` | Core API. Instantiate `MolecularQNN` → call `train()` / `get_energy()` |
| `cli_interface.py` | Menu-driven flow for quick experiments without coding |
| `ansatz_comparison_gui.py` | Desktop GUI with live plots for comparing ansätze |
| `quantum_hardware_interface.py` | Minimal abstraction of *any* quantum device / simulator |
| `quantum_hardware_enhanced.py` | Adds provider registry, retries, circuit validation, error-mitigation |
| `quantum_circuit_transpiler.py` | Gate-set & connectivity aware transpilation with 5 optimisation levels |

---

## Installation details

1. **Python** 3.10+ (tested with 3.11).
2. `pip install -r requirements.txt` installs pinned versions of:
   * Cirq 1.2, OpenFermion 1.6, NumPy/SciPy, Matplotlib, PyQt5 …
   * Dev-tools: `pytest`, `ruff`, `pre-commit`.
3. Optional extras (commented in `requirements.txt`): Qiskit, PennyLane, Braket …

> ✨ Tip: activate pre-commit hooks with `pre-commit install` to auto-run ruff & pytest on each commit.

---

## Typical workflows

### 1. Energy estimation from Python

```python
from qnn_molecular_energy import MolecularQNN

qnn = MolecularQNN(
    molecule="H2",           # or "LiH", "H2O"
    bond_length=0.74,
    ansatz_type="ucc",      # see list at top
    depth=2,
)
qnn.train(iterations=100, method="BFGS")
print("Energy:", qnn.get_energy())
```

### 2. CLI exploration
Run `python cli_interface.py` and follow the prompts to:
* create a model
* train with different optimisers
* run quick comparisons or hardware simulations

### 3. GUI visual comparison
Run `python ansatz_comparison_gui.py` and select molecules / ansätze; live charts will show convergence per ansatz.

### 4. Add a new hardware back-end
```python
from quantum_hardware_interface import QuantumBackend
from quantum_hardware_enhanced import default_provider

class MyBackend(QuantumBackend):
    def run_circuit(self, circuit, repetitions=1000): ...
    def get_device_properties(self): ...

def backend_factory(**kwargs):
    return MyBackend("my_backend", **kwargs)

default_provider.register_backend("my_backend", backend_factory)
```

---

## Testing & quality gates

* **Run all tests**: `pytest -q`
* **Lint / format**: `ruff check --fix . && ruff format .`
* **Pre-commit**: installs hooks that run the above automatically.

---

## Contributing

1. Fork / branch off `master`.
2. Activate pre-commit: `pre-commit install`.
3. Add tests for any new feature or bug-fix.
4. Open a PR – GitHub Actions will run lint + tests.

---

## License

This project is released under the MIT License. See `LICENSE` for details. 