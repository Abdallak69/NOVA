#!/usr/bin/env python3
"""
Unit tests for the ansatz_circuits.py module.

This module contains tests to verify the functionality of different ansatz circuits
implemented in the QNN project.
"""

import os
import sys
import unittest

import cirq
import numpy as np
import openfermion

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module to test
from ansatz_circuits import (
    CHEAnsatz,
    HamiltonianVariationalAnsatz,
    HardwareEfficientAnsatz,
    SymmetryPreservingAnsatz,
    UCCAnsatz,
    create_ansatz,
)


class TestAnsatzCircuits(unittest.TestCase):
    """Tests for the ansatz_circuits module."""

    def test_hardware_efficient_ansatz(self):
        """Test the HardwareEfficientAnsatz circuit."""
        qubits = cirq.LineQubit.range(4)
        ansatz = HardwareEfficientAnsatz(
            qubits=qubits, depth=2, rotation_gates="XYZ", entangle_pattern="linear"
        )
        circuit = ansatz.build_circuit()

        # Check that the circuit is a valid Cirq Circuit
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the number of parameters matches expectations
        # For XYZ rotations: 3 rotations per qubit per layer + 1 final layer
        expected_params = (3 * 4 * 2) + (3 * 4)  # depth=2 + final layer
        self.assertEqual(ansatz.param_count(), expected_params)

        # Check that the circuit acts on the expected qubits
        circuit_qubits = {q for op in circuit.all_operations() for q in op.qubits}
        self.assertEqual(circuit_qubits, set(qubits))

    def test_ucc_ansatz(self):
        """Test the UCCAnsatz circuit."""
        qubits = cirq.LineQubit.range(4)
        ansatz = UCCAnsatz(
            qubits=qubits, depth=1, include_singles=True, include_doubles=True
        )
        circuit = ansatz.build_circuit()

        # Check that the circuit is a valid Cirq Circuit
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the circuit acts on the expected qubits
        circuit_qubits = {q for op in circuit.all_operations() for q in op.qubits}
        self.assertEqual(circuit_qubits, set(qubits))

    def test_chea_ansatz(self):
        """Test the CHEA ansatz circuit."""
        qubits = cirq.LineQubit.range(4)
        ansatz = CHEAnsatz(qubits=qubits, depth=2, skip_final_rotation_layer=False)
        circuit = ansatz.build_circuit()

        # Check that the circuit is a valid Cirq Circuit
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the circuit acts on the expected qubits
        circuit_qubits = {q for op in circuit.all_operations() for q in op.qubits}
        self.assertEqual(circuit_qubits, set(qubits))

    def test_symmetry_preserving_ansatz(self):
        """Test the SymmetryPreservingAnsatz circuit."""
        qubits = cirq.LineQubit.range(4)
        ansatz = SymmetryPreservingAnsatz(
            qubits=qubits, depth=1, conserve_particle_number=True
        )
        circuit = ansatz.build_circuit()

        # Check that the circuit is a valid Cirq Circuit
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the circuit acts on the expected qubits
        circuit_qubits = {q for op in circuit.all_operations() for q in op.qubits}
        self.assertEqual(circuit_qubits, set(qubits))

    def test_hamiltonian_variational_ansatz(self):
        """Test the HamiltonianVariationalAnsatz class."""
        # Create a 4-qubit HVA
        qubits = cirq.LineQubit.range(4)

        # Create a simple test Hamiltonian
        test_ham_terms = []
        for i in range(len(qubits) - 1):
            test_ham_terms.append(cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]))

        ansatz = HamiltonianVariationalAnsatz(
            qubits=qubits, hamiltonian_terms=test_ham_terms, depth=2
        )

        # Build the circuit
        circuit = ansatz.build_circuit()

        # Check that the circuit is a valid Cirq Circuit
        self.assertIsInstance(circuit, cirq.Circuit)

        # Check that the circuit acts on the expected qubits
        circuit_qubits = {q for op in circuit.all_operations() for q in op.qubits}
        self.assertEqual(circuit_qubits, set(qubits))

    def test_create_ansatz_factory(self):
        """Test the create_ansatz factory function."""
        qubits = cirq.LineQubit.range(2)

        # Test creation of different ansatz types
        he_ansatz = create_ansatz("hardware_efficient", qubits)
        self.assertIsInstance(he_ansatz, HardwareEfficientAnsatz)

        ucc_ansatz = create_ansatz("ucc", qubits)
        self.assertIsInstance(ucc_ansatz, UCCAnsatz)

        chea_ansatz = create_ansatz("chea", qubits)
        self.assertIsInstance(chea_ansatz, CHEAnsatz)

        sp_ansatz = create_ansatz("symmetry_preserving", qubits)
        self.assertIsInstance(sp_ansatz, SymmetryPreservingAnsatz)

        # For HVA, we need hamiltonian terms
        ham_terms = [cirq.Z(qubits[0]) * cirq.Z(qubits[1])]
        hva_ansatz = create_ansatz("hva", qubits, hamiltonian_terms=ham_terms)
        self.assertIsInstance(hva_ansatz, HamiltonianVariationalAnsatz)

        # Test with invalid ansatz type
        with self.assertRaises(ValueError):
            create_ansatz("invalid_type", qubits)

    def test_parameter_values(self):
        """Test setting and getting parameter values."""
        qubits = cirq.LineQubit.range(2)
        ansatz = HardwareEfficientAnsatz(qubits, depth=1)

        # Build the circuit first
        circuit = ansatz.build_circuit()

        # Get the parameters
        params = ansatz.get_parameters()

        # Create random values
        values = np.random.random(len(params))

        # Test parameter assignment
        param_dict = ansatz.assign_parameters(values)

        # Check that all parameters have been assigned values
        self.assertEqual(len(param_dict), len(params))

        # Check that the circuit resolves correctly with these parameters
        resolved_circuit = cirq.resolve_parameters(circuit, param_dict)
        self.assertFalse(cirq.is_parameterized(resolved_circuit))


if __name__ == "__main__":
    unittest.main()
