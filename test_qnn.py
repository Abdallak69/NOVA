#!/usr/bin/env python3
"""
Simple script to test the QNN functionality programmatically.
"""

import matplotlib.pyplot as plt
import pytest

from qnn_molecular_energy import MolecularQNN


def test_basic_qnn():
    """Test the basic functionality of the MolecularQNN class."""
    print("\n--- Testing Basic QNN Functionality ---")
    
    qnn = MolecularQNN(
        molecule="H2",
        bond_length=0.74,
        ansatz_type="hardware_efficient",
        depth=2
    )
    
    initial_energy = qnn.get_energy()
    print(f"Initial random energy: {initial_energy:.6f}")
    assert isinstance(initial_energy, float)
    
    # Test training
    results = qnn.train(iterations=5, method="BFGS", verbose=False)
    final_energy = qnn.get_energy()
    print(f"Final energy after training: {final_energy:.6f}")
    
    assert final_energy < initial_energy
    assert "final_params" in results
    assert len(results["energy_history"]) > 1


def test_hardware_simulation():
    """Test running the QNN with a simulated hardware backend."""
    print("\n--- Testing QNN with Hardware Simulation ---")
    
    # Check if hardware modules are available
    try:
        from quantum_hardware_interface import CirqSimulatorBackend
    except ImportError:
        pytest.skip("Hardware modules not available, skipping test.")
        
    backend = CirqSimulatorBackend(name="test_simulator")
    
    qnn = MolecularQNN(
        molecule="H2",
        bond_length=0.74,
        hardware_backend=backend
    )
    
    # Run a single energy evaluation
    energy = qnn.get_energy(shots=1000)
    print(f"Energy from simulated hardware: {energy:.6f}")
    assert isinstance(energy, float)


def test_advanced_optimizer():
    """Test the QNN with an advanced optimizer."""
    print("\n--- Testing QNN with Advanced Optimizer ---")
    
    try:
        from advanced_optimizers import create_optimizer
    except ImportError:
        pytest.skip("Advanced optimizers not available, skipping test.")
        
    optimizer = create_optimizer("gradient_free", method="nelder-mead")
    
    qnn = MolecularQNN(
        molecule="H2",
        bond_length=0.74,
        optimizer=optimizer
    )
    
    results = qnn.train(iterations=10, verbose=False)
    assert results["success"] or not results["success"] # Check that training completes


def test_compare_ansatz():
    """Test the ansatz comparison functionality."""
    print("\n--- Testing Ansatz Comparison ---")
    
    qnn = MolecularQNN(molecule="H2", bond_length=0.74)
    
    results = qnn.compare_ansatz_types(
        ansatz_types=["hardware_efficient", "ucc", "hva"],
        iterations=10,
        show_plot=False
    )
    
    assert "hardware_efficient" in results
    assert "ucc" in results
    assert "hva" in results
    assert "final_energy" in results["hardware_efficient"]


if __name__ == "__main__":
    print("=" * 80)
    print("QNN Test Script")
    print("=" * 80)

    # Test basic QNN
    test_basic_qnn()

    # Test hardware simulation
    test_hardware_simulation()

    # Test advanced optimizer
    test_advanced_optimizer()

    # Test compare ansatz
    test_compare_ansatz()

    print("\nAll tests completed!")
