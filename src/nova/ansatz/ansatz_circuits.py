#!/usr/bin/env python3

"""
Advanced Ansatz Circuits for Quantum Neural Networks

This module provides a collection of sophisticated ansatz circuits for
quantum neural networks to improve molecular energy estimation accuracy.
"""

from typing import Dict, List, Tuple, Union

import cirq
import sympy


class AnsatzCircuit:
    """Base class for different ansatz circuit implementations."""

    def __init__(self, qubits: List[cirq.Qid], name: str = "base"):
        """
        Initialize the base ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            name: Name of the ansatz circuit
        """
        self.qubits = qubits
        self.name = name
        self.parameters = []

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the quantum circuit for this ansatz.

        Returns:
            The constructed quantum circuit
        """
        raise NotImplementedError("Subclasses must implement build_circuit method")

    def param_count(self) -> int:
        """
        Get the number of parameters in the circuit.

        Returns:
            Number of parameters
        """
        return len(self.parameters)

    def get_parameters(self) -> List[sympy.Symbol]:
        """
        Get the list of circuit parameters.

        Returns:
            List of parameter symbols
        """
        return self.parameters

    def assign_parameters(self, values: List[float]) -> Dict[sympy.Symbol, float]:
        """
        Assign values to the parameters.

        Args:
            values: List of parameter values

        Returns:
            Dictionary mapping parameter symbols to values
        """
        if len(values) != len(self.parameters):
            raise ValueError(
                f"Expected {len(self.parameters)} parameter values, got {len(values)}"
            )
        return dict(zip(self.parameters, values))

    def __str__(self) -> str:
        """String representation of the ansatz circuit."""
        return f"{self.name} Ansatz with {self.param_count()} parameters on {len(self.qubits)} qubits"


class HardwareEfficientAnsatz(AnsatzCircuit):
    """
    Hardware-Efficient Ansatz using native gates in a layered structure.

    This ansatz consists of single-qubit rotation gates followed by entangling gates
    in a repeating pattern. It's well-suited for NISQ devices as it uses gates that
    are natively available on the hardware.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        depth: int = 2,
        rotation_gates: str = "XYZ",
        entangle_pattern: str = "linear",
        skip_final_rotation_layer: bool = False,
    ):
        """
        Initialize a Hardware-Efficient ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            depth: Number of repetitions of rotation-entanglement layers
            rotation_gates: String of rotation gates to use (any combination of 'X', 'Y', 'Z')
            entangle_pattern: Entanglement pattern ('linear', 'full', or custom list of pairs)
            skip_final_rotation_layer: Whether to skip the final rotation layer
        """
        super().__init__(qubits, name="Hardware-Efficient")
        self.depth = depth
        self.rotation_gates = rotation_gates
        self.entangle_pattern = entangle_pattern
        self.skip_final_rotation_layer = skip_final_rotation_layer

        # Validate rotation gates
        for gate in self.rotation_gates:
            if gate not in "XYZ":
                raise ValueError(
                    f"Invalid rotation gate: {gate}. Must be one of 'X', 'Y', 'Z'"
                )

        # Create entangling pairs based on the pattern
        self.entangling_pairs = []
        if isinstance(entangle_pattern, list):
            # Custom pairs provided
            self.entangling_pairs = entangle_pattern
        elif entangle_pattern == "linear":
            # Linear nearest-neighbor
            for i in range(len(qubits) - 1):
                self.entangling_pairs.append((i, i + 1))
        elif entangle_pattern == "full":
            # All-to-all
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    self.entangling_pairs.append((i, j))
        else:
            raise ValueError(f"Invalid entanglement pattern: {entangle_pattern}")

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the hardware-efficient ansatz circuit.

        Returns:
            The quantum circuit
        """
        circuit = cirq.Circuit()
        self.parameters = []

        # Create parameters and rotations
        param_idx = 0

        # Create layers of rotations and entanglement
        for _layer in range(self.depth):
            # Rotation layer
            for _q_idx, qubit in enumerate(self.qubits):
                for gate in self.rotation_gates:
                    param = sympy.Symbol(f"θ_{param_idx}")
                    self.parameters.append(param)
                    param_idx += 1

                    if gate == "X":
                        circuit.append(cirq.rx(param)(qubit))
                    elif gate == "Y":
                        circuit.append(cirq.ry(param)(qubit))
                    elif gate == "Z":
                        circuit.append(cirq.rz(param)(qubit))

            # Entanglement layer
            for i, j in self.entangling_pairs:
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[j]))

        # Final rotation layer (optional)
        if not self.skip_final_rotation_layer:
            for _q_idx, qubit in enumerate(self.qubits):
                for gate in self.rotation_gates:
                    param = sympy.Symbol(f"θ_{param_idx}")
                    self.parameters.append(param)
                    param_idx += 1

                    if gate == "X":
                        circuit.append(cirq.rx(param)(qubit))
                    elif gate == "Y":
                        circuit.append(cirq.ry(param)(qubit))
                    elif gate == "Z":
                        circuit.append(cirq.rz(param)(qubit))

        return circuit


class UCCAnsatz(AnsatzCircuit):
    """
    Unitary Coupled Cluster Ansatz for quantum chemistry.

    This ansatz is inspired by the classical coupled cluster method in quantum chemistry.
    It preserves particle number and often provides a more accurate representation
    for molecular systems.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        depth: int = 1,
        include_singles: bool = True,
        include_doubles: bool = True,
    ):
        """
        Initialize a UCC ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            depth: Number of repetitions of the UCC operator
            include_singles: Whether to include single excitation operators
            include_doubles: Whether to include double excitation operators
        """
        super().__init__(qubits, name="UCC")
        self.depth = depth
        self.include_singles = include_singles
        self.include_doubles = include_doubles

        # Prepare excitation operators
        self._prepare_excitation_ops()

    def _prepare_excitation_ops(self):
        """Prepare single and double excitation operators."""
        n_qubits = len(self.qubits)
        n_orbitals = n_qubits // 2  # Assuming spin orbitals

        self.singles = []
        self.doubles = []

        # Singles: excite from occupied to unoccupied orbital
        if self.include_singles:
            for i in range(n_orbitals // 2):
                for a in range(n_orbitals // 2, n_orbitals):
                    # Alpha-alpha excitation
                    self.singles.append((2 * i, 2 * a))
                    # Beta-beta excitation
                    self.singles.append((2 * i + 1, 2 * a + 1))

        # Doubles: excite two electrons from occupied to unoccupied orbitals
        if self.include_doubles:
            for i in range(n_orbitals // 2):
                for j in range(i, n_orbitals // 2):
                    for a in range(n_orbitals // 2, n_orbitals):
                        for b in range(a, n_orbitals):
                            # Different excitation types
                            if i != j and a != b:
                                # Double excitation: (i,j) -> (a,b)
                                self.doubles.append((2 * i, 2 * j, 2 * a, 2 * b))
                                self.doubles.append(
                                    (2 * i + 1, 2 * j + 1, 2 * a + 1, 2 * b + 1)
                                )
                                self.doubles.append(
                                    (2 * i, 2 * j + 1, 2 * a, 2 * b + 1)
                                )
                                self.doubles.append(
                                    (2 * i + 1, 2 * j, 2 * a + 1, 2 * b)
                                )

    def _apply_single_excitation(self, circuit, param, i, a):
        """Apply a single excitation operator to the circuit."""
        # Ensure indices are within bounds
        if i >= len(self.qubits) or a >= len(self.qubits):
            return

        # Fermionic single excitation: a†_a a_i - a†_i a_a
        # This is approximated with CNOT + Rotations
        q_i = self.qubits[i]
        q_a = self.qubits[a]

        circuit.append(cirq.CNOT(q_i, q_a))
        circuit.append(cirq.rx(param)(q_a))
        circuit.append(cirq.CNOT(q_i, q_a))

    def _apply_double_excitation(self, circuit, param, i, j, a, b):
        """Apply a double excitation operator to the circuit."""
        # Ensure indices are within bounds
        if max(i, j, a, b) >= len(self.qubits):
            return

        # Fermionic double excitation: a†_a a†_b a_j a_i - a†_i a†_j a_b a_a
        # This is approximated with a sequence of CNOTs and rotations
        q_i = self.qubits[i]
        q_j = self.qubits[j]
        q_a = self.qubits[a]
        q_b = self.qubits[b]

        # Simplified double excitation circuit
        circuit.append(cirq.CNOT(q_i, q_j))
        circuit.append(cirq.CNOT(q_a, q_b))
        circuit.append(cirq.CNOT(q_j, q_a))

        circuit.append(cirq.rz(param)(q_a))

        circuit.append(cirq.CNOT(q_j, q_a))
        circuit.append(cirq.CNOT(q_a, q_b))
        circuit.append(cirq.CNOT(q_i, q_j))

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the UCC ansatz circuit.

        Returns:
            The quantum circuit
        """
        circuit = cirq.Circuit()
        self.parameters = []

        # Initial Hartree-Fock state preparation
        # Set the first half of qubits to |1⟩ (occupied)
        for i in range(len(self.qubits) // 2):
            circuit.append(cirq.X(self.qubits[i]))

        # Apply UCC operator layers
        param_idx = 0
        for _ in range(self.depth):
            # Apply single excitations
            for i, a in self.singles:
                param = sympy.Symbol(f"θ_{param_idx}")
                self.parameters.append(param)
                self._apply_single_excitation(circuit, param, i, a)
                param_idx += 1

            # Apply double excitations
            for i, j, a, b in self.doubles:
                param = sympy.Symbol(f"θ_{param_idx}")
                self.parameters.append(param)
                self._apply_double_excitation(circuit, param, i, j, a, b)
                param_idx += 1

        return circuit


class CHEAnsatz(AnsatzCircuit):
    """
    Custom Hardware-Efficient Ansatz with enhanced flexibility.

    This is an enhanced version of the hardware-efficient ansatz that provides
    more flexibility and expressivity through additional rotation freedom and
    customizable entanglement.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        depth: int = 2,
        entangle_pattern: Union[str, List[Tuple[int, int]]] = "linear",
        skip_final_rotation_layer: bool = False,
    ):
        """
        Initialize a Custom Hardware-Efficient Ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            depth: Number of repetitions of rotation-entanglement layers
            entangle_pattern: Entanglement pattern ('linear', 'full', or custom list of pairs)
            skip_final_rotation_layer: Whether to skip the final rotation layer
        """
        super().__init__(qubits, name="Custom Hardware-Efficient")
        self.depth = depth
        self.skip_final_rotation_layer = skip_final_rotation_layer

        # Create entangling pairs based on the pattern
        self.entangling_pairs = []
        if isinstance(entangle_pattern, list):
            # Custom pairs provided
            self.entangling_pairs = entangle_pattern
        elif entangle_pattern == "linear":
            # Linear nearest-neighbor
            for i in range(len(qubits) - 1):
                self.entangling_pairs.append((i, i + 1))
        elif entangle_pattern == "full":
            # All-to-all
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    self.entangling_pairs.append((i, j))
        else:
            raise ValueError(f"Invalid entanglement pattern: {entangle_pattern}")

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the custom hardware-efficient ansatz circuit.

        Returns:
            The quantum circuit
        """
        circuit = cirq.Circuit()
        self.parameters = []

        # Create parameters and rotations
        param_idx = 0

        # Create layers of rotations and entanglement
        for _layer in range(self.depth):
            # Full rotation layer (Rx, Ry, Rz on each qubit)
            for _q_idx, qubit in enumerate(self.qubits):
                # Apply all three rotation gates for maximum flexibility
                for gate_type in ["X", "Y", "Z"]:
                    param = sympy.Symbol(f"θ_{param_idx}")
                    self.parameters.append(param)
                    param_idx += 1

                    if gate_type == "X":
                        circuit.append(cirq.rx(param)(qubit))
                    elif gate_type == "Y":
                        circuit.append(cirq.ry(param)(qubit))
                    elif gate_type == "Z":
                        circuit.append(cirq.rz(param)(qubit))

            # Entanglement layer with parameterized two-qubit gates
            for i, j in self.entangling_pairs:
                # CZ gate (fixed, not parameterized)
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[j]))

                # Additional Z rotations after entanglement
                param1 = sympy.Symbol(f"θ_{param_idx}")
                param2 = sympy.Symbol(f"θ_{param_idx + 1}")
                self.parameters.extend([param1, param2])
                param_idx += 2

                circuit.append(cirq.rz(param1)(self.qubits[i]))
                circuit.append(cirq.rz(param2)(self.qubits[j]))

        # Final rotation layer (optional)
        if not self.skip_final_rotation_layer:
            for _q_idx, qubit in enumerate(self.qubits):
                for gate_type in ["X", "Y", "Z"]:
                    param = sympy.Symbol(f"θ_{param_idx}")
                    self.parameters.append(param)
                    param_idx += 1

                    if gate_type == "X":
                        circuit.append(cirq.rx(param)(qubit))
                    elif gate_type == "Y":
                        circuit.append(cirq.ry(param)(qubit))
                    elif gate_type == "Z":
                        circuit.append(cirq.rz(param)(qubit))

        return circuit


class SymmetryPreservingAnsatz(AnsatzCircuit):
    """
    Symmetry-Preserving Ansatz for quantum systems with conservation laws.

    This ansatz is designed to preserve important symmetries of the problem Hamiltonian,
    such as particle number, spin, or geometric symmetries.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        depth: int = 1,
        conserve_particle_number: bool = True,
    ):
        """
        Initialize a Symmetry-Preserving ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            depth: Number of repetitions of the symmetry-preserving block
            conserve_particle_number: Whether to conserve particle number
        """
        super().__init__(qubits, name="Symmetry-Preserving")
        self.depth = depth
        self.conserve_particle_number = conserve_particle_number

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the symmetry-preserving ansatz circuit.

        Returns:
            The quantum circuit
        """
        circuit = cirq.Circuit()
        self.parameters = []
        param_idx = 0

        # Prepare initial state with the correct number of particles
        n_qubits = len(self.qubits)
        n_particles = n_qubits // 2  # Half-filling by default

        if self.conserve_particle_number:
            # Set the first n_particles qubits to |1⟩ (occupied)
            for i in range(n_particles):
                circuit.append(cirq.X(self.qubits[i]))
        else:
            # Use Hadamard gates for superposition
            for i in range(n_qubits):
                circuit.append(cirq.H(self.qubits[i]))

        # Apply symmetry-preserving blocks
        for _layer in range(self.depth):
            # Apply particle-number-conserving operations
            if self.conserve_particle_number:
                # These operations exchange particles but maintain the total count
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        # Particle-swap operation
                        param = sympy.Symbol(f"θ_{param_idx}")
                        self.parameters.append(param)
                        param_idx += 1

                        # SWAP-like operation that preserves particle number
                        circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                        circuit.append(cirq.ry(param)(self.qubits[j]))
                        circuit.append(cirq.CNOT(self.qubits[j], self.qubits[i]))
                        circuit.append(cirq.ry(-param)(self.qubits[i]))
                        circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
            else:
                # Without particle number conservation, we can use general rotations
                for _i, qubit in enumerate(self.qubits):
                    for axis in ["X", "Y", "Z"]:
                        param = sympy.Symbol(f"θ_{param_idx}")
                        self.parameters.append(param)
                        param_idx += 1

                        if axis == "X":
                            circuit.append(cirq.rx(param)(qubit))
                        elif axis == "Y":
                            circuit.append(cirq.ry(param)(qubit))
                        elif axis == "Z":
                            circuit.append(cirq.rz(param)(qubit))

                # Add entanglement
                for i in range(n_qubits - 1):
                    circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))

        return circuit


class HamiltonianVariationalAnsatz(AnsatzCircuit):
    """
    Hamiltonian Variational Ansatz based on problem structure.

    This ansatz is based on the structure of the problem Hamiltonian itself,
    using alternating layers of evolution under different terms of the Hamiltonian.
    """

    def __init__(
        self,
        qubits: List[cirq.Qid],
        hamiltonian_terms: List[cirq.PauliSum],
        depth: int = 2,
    ):
        """
        Initialize a Hamiltonian Variational Ansatz circuit.

        Args:
            qubits: List of qubits to use in the circuit
            hamiltonian_terms: List of Hamiltonian terms to use in the ansatz
            depth: Number of repetitions of the Hamiltonian evolution block
        """
        super().__init__(qubits, name="Hamiltonian Variational")
        self.hamiltonian_terms = hamiltonian_terms
        self.depth = depth

    def _apply_pauli_string_evolution(self, circuit, pauli_string, param):
        """Apply time evolution under a Pauli string operator."""
        if isinstance(pauli_string, str) and pauli_string == "I":
            # Identity term, just add global phase
            return

        # Extract rotation axis and qubits
        if isinstance(pauli_string, cirq.PauliString):
            # Apply entangling gates first
            for i in range(len(self.qubits) - 1):
                circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))

            # Apply specific rotations for this Pauli operator
            for qubit, gate in pauli_string.items():
                if gate == cirq.X:
                    circuit.append(cirq.rx(param)(qubit))
                elif gate == cirq.Y:
                    circuit.append(cirq.ry(param)(qubit))
                elif gate == cirq.Z:
                    circuit.append(cirq.rz(param)(qubit))

    def build_circuit(self) -> cirq.Circuit:
        """
        Build the Hamiltonian variational ansatz circuit.

        Returns:
            The quantum circuit
        """
        circuit = cirq.Circuit()
        self.parameters = []

        # Initial state preparation (Hadamard on all qubits)
        for qubit in self.qubits:
            circuit.append(cirq.H(qubit))

        # Apply layers of Hamiltonian evolution
        param_idx = 0
        for _layer in range(self.depth):
            # Apply evolution under each Hamiltonian term
            for term in self.hamiltonian_terms:
                param = sympy.Symbol(f"θ_{param_idx}")
                self.parameters.append(param)
                param_idx += 1

                self._apply_pauli_string_evolution(circuit, term, param)

        return circuit


def create_ansatz(name: str, qubits: List[cirq.Qid], **kwargs) -> AnsatzCircuit:
    """
    Factory function to create an ansatz circuit of the specified type.

    Args:
        name: Name of the ansatz type
        qubits: List of qubits to use in the circuit
        **kwargs: Additional arguments for the specific ansatz type

    Returns:
        An instance of the specified ansatz circuit

    Raises:
        ValueError: If the ansatz type is not recognized
    """
    name = name.lower()

    if name == "hardware_efficient":
        return HardwareEfficientAnsatz(qubits, **kwargs)
    elif name == "ucc":
        return UCCAnsatz(qubits, **kwargs)
    elif name == "chea":
        return CHEAnsatz(qubits, **kwargs)
    elif name == "symmetry_preserving":
        return SymmetryPreservingAnsatz(qubits, **kwargs)
    elif name == "hva":
        # Ensure Hamiltonian terms are provided
        if "hamiltonian_terms" not in kwargs:
            raise ValueError("Hamiltonian terms must be provided for HVA")
        return HamiltonianVariationalAnsatz(qubits, **kwargs)
    else:
        raise ValueError(f"Unknown ansatz type: {name}")


# Example usage
if __name__ == "__main__":
    # Create qubits
    qubits = cirq.LineQubit.range(4)

    # Hardware-efficient ansatz
    he_ansatz = HardwareEfficientAnsatz(
        qubits, depth=2, rotation_gates="XYZ", entangle_pattern="linear"
    )
    he_circuit = he_ansatz.build_circuit()
    print(f"Hardware-Efficient Ansatz: {he_ansatz.param_count()} parameters")
    print(he_circuit)

    # UCC ansatz
    ucc_ansatz = UCCAnsatz(qubits, depth=1, include_singles=True, include_doubles=True)
    ucc_circuit = ucc_ansatz.build_circuit()
    print(f"UCC Ansatz: {ucc_ansatz.param_count()} parameters")
    print(ucc_circuit)

    # Custom hardware-efficient ansatz
    chea_ansatz = CHEAnsatz(qubits, depth=2, entangle_pattern="linear")
    chea_circuit = chea_ansatz.build_circuit()
    print(f"CHEA Ansatz: {chea_ansatz.param_count()} parameters")
    print(chea_circuit)

    # Symmetry-preserving ansatz
    sp_ansatz = SymmetryPreservingAnsatz(qubits, depth=1, conserve_particle_number=True)
    sp_circuit = sp_ansatz.build_circuit()
    print(f"Symmetry-Preserving Ansatz: {sp_ansatz.param_count()} parameters")
    print(sp_circuit)

    # Hamiltonian variational ansatz
    ham_terms = []
    for i in range(len(qubits) - 1):
        ham_terms.append(cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1]))

    hva_ansatz = HamiltonianVariationalAnsatz(
        qubits, hamiltonian_terms=ham_terms, depth=2
    )
    hva_circuit = hva_ansatz.build_circuit()
    print(f"Hamiltonian Variational Ansatz: {hva_ansatz.param_count()} parameters")
    print(hva_circuit)
