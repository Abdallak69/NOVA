# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Modern package structure with src-layout
- Command-line interface using Click
- GUI launcher for different interfaces
- Comprehensive test suite
- CI/CD pipeline with GitHub Actions
- Pre-commit hooks for code quality
- Type checking with MyPy
- Code formatting with Black and Ruff

### Changed
- Migrated to src-layout package structure
- Updated all imports to use new package paths
- Refactored CLI from interactive scripts to modern Click commands
- Enhanced project configuration in pyproject.toml
- Improved test organization and structure

### Fixed
- Import paths for all modules
- Package discovery for setuptools
- CLI entry points and command structure

## [0.1.0] - 2024-01-XX

### Added
- Initial release of NOVA Quantum Neural Network
- Molecular energy estimation using quantum circuits
- Multiple ansatz implementations (Hardware-Efficient, UCC, CHEA, HVA, Symmetry-Preserving)
- Quantum hardware integration with Cirq and Qiskit
- Error mitigation strategies
- Circuit transpilation and optimization
- Hardware analytics and benchmarking tools
- GUI interfaces for interactive use
- Support for H2, LiH, and H2O molecules
- Advanced optimizers for quantum parameter optimization
- Visualization tools for results and circuits
- Comprehensive logging and error handling

### Documentation
- README with installation and usage instructions
- Code documentation and examples
- Hardware integration guides
- Ansatz comparison documentation

### Dependencies
- Core: NumPy, SciPy, Matplotlib, Cirq, OpenFermion
- GUI: PyQt5
- Optional: Qiskit, PySCF for advanced features
- Development: pytest, ruff, mypy, pre-commit 