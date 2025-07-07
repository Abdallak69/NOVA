# Contributing to NOVA Quantum Neural Network

We welcome contributions to the NOVA project! This document provides guidelines for contributing code, reporting issues, and submitting improvements.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Optional: PyQt5 for GUI features
- Optional: PySCF for advanced molecular calculations

### Setting up the development environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nova-team/nova-qnn.git
   cd nova-qnn
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .[dev]
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Package Structure

The project follows the src-layout structure:

```
NOVA/
├── src/nova/           # Main package code
│   ├── core/          # Core QNN implementation
│   ├── ansatz/        # Quantum circuit ansätze
│   ├── hardware/      # Hardware interfaces
│   ├── transpiler/    # Circuit optimization
│   ├── mitigation/    # Error mitigation
│   ├── cli/           # Command-line interface
│   └── gui/           # Graphical interfaces
├── tests/             # Test suite
├── examples/          # Usage examples
├── docs/              # Documentation
└── scripts/           # Helper scripts
```

## Code Style

We use several tools to maintain code quality:

### Formatting
- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting and additional formatting

### Type Checking
- **MyPy**: Static type checking (optional but encouraged)

### Running code quality checks

All formatting and linting is handled by pre-commit hooks, but you can run them manually:

```bash
# Format code
black src tests examples

# Sort imports
isort src tests examples

# Lint code
ruff check src tests examples

# Type checking
mypy src
```

### Code Style Guidelines

1. **Line length**: Maximum 88 characters
2. **Docstrings**: Use Google-style docstrings
3. **Type hints**: Add type hints for public APIs
4. **Naming**: Use descriptive names, snake_case for functions/variables
5. **Comments**: Write clear, concise comments for complex logic

Example function:

```python
def calculate_energy(
    circuit: cirq.Circuit,
    hamiltonian: openfermion.QubitOperator,
    backend: str = "cirq_simulator"
) -> float:
    """Calculate the expectation value of a Hamiltonian.
    
    Args:
        circuit: The quantum circuit to execute
        hamiltonian: The Hamiltonian operator
        backend: Name of the quantum backend to use
        
    Returns:
        The calculated energy expectation value
        
    Raises:
        ValueError: If the circuit or Hamiltonian is invalid
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nova

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "not hardware"  # Skip hardware tests
pytest tests/core/  # Run core tests only
```

### Test Categories

Tests are organized with markers:
- `slow`: Long-running tests
- `integration`: Integration tests
- `hardware`: Tests requiring quantum hardware
- `gui`: Tests requiring GUI dependencies

### Writing Tests

1. **Test file naming**: `test_*.py` or `*_test.py`
2. **Test function naming**: `test_*`
3. **Assertions**: Use pytest assertions
4. **Fixtures**: Use pytest fixtures for common setup
5. **Mocking**: Mock external dependencies when appropriate

Example test:

```python
import pytest
from nova.core.qnn_molecular_energy import MolecularQNN

def test_qnn_creation():
    """Test basic QNN creation."""
    qnn = MolecularQNN(molecule="H2", depth=2)
    assert qnn.molecule == "h2"
    assert qnn.depth == 2
    assert qnn.n_qubits > 0

@pytest.mark.slow
def test_qnn_training():
    """Test QNN training (slow test)."""
    qnn = MolecularQNN(molecule="H2", depth=1)
    qnn.train(iterations=10)
    assert len(qnn.energy_history) > 0
```

## Submitting Changes

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding guidelines
3. **Add tests** for any new functionality
4. **Update documentation** if needed
5. **Ensure all tests pass** and code quality checks succeed
6. **Submit a pull request** with a clear description

### Branch Naming

Use descriptive branch names:
- `feature/add-new-ansatz`
- `fix/hardware-connection-bug`
- `docs/update-installation-guide`
- `refactor/improve-error-handling`

### Commit Messages

Follow conventional commits format:

```
type(scope): brief description

Longer description if needed

- List any breaking changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

Example:
```
feat(ansatz): add symmetry-preserving ansatz

Implement symmetry-preserving ansatz circuit that conserves
particle number and spin symmetries.

- Add SymmetryPreservingAnsatz class
- Include tests for symmetry conservation
- Update documentation

Fixes #45
```

### Pull Request Template

When submitting a PR, include:

- **Description**: What changes were made and why
- **Type of change**: Bug fix, new feature, breaking change, etc.
- **Testing**: How the changes were tested
- **Checklist**: Ensure all requirements are met

## Issue Reporting

### Bug Reports

Include the following information:
- NOVA version
- Python version and OS
- Complete error traceback
- Minimal code example to reproduce
- Expected vs actual behavior

### Feature Requests

Describe:
- The problem or use case
- Proposed solution
- Alternative solutions considered
- Implementation details if known

### Questions and Support

For questions about usage:
1. Check existing documentation
2. Search existing issues
3. Create a new issue with the "question" label

## Documentation

### Code Documentation

- Use Google-style docstrings
- Document all public APIs
- Include examples in docstrings when helpful
- Keep documentation up-to-date with code changes

### User Documentation

Documentation is built with Sphinx and hosted on Read the Docs:

```bash
# Build documentation locally
cd docs
make html
```

### Examples

Add examples to the `examples/` directory:
- Include clear comments
- Use realistic but simple examples
- Test examples to ensure they work

## Development Workflow Summary

1. **Setup**: Fork, clone, install dependencies
2. **Develop**: Create feature branch, write code and tests
3. **Quality**: Run pre-commit hooks, ensure tests pass
4. **Submit**: Create pull request with clear description
5. **Review**: Address feedback, iterate as needed
6. **Merge**: Squash and merge when approved

## Getting Help

- **Documentation**: Read the docs at [link]
- **Issues**: Search and create GitHub issues
- **Discussions**: Use GitHub Discussions for questions
- **Chat**: Join our development chat [if available]

Thank you for contributing to NOVA! 🚀 