#!/usr/bin/env python3

"""
Quantum Neural Network for Molecular Energy Estimation

This module implements a Quantum Neural Network (QNN) using Google's Cirq
framework to estimate ground state energies of small molecules.
"""

import logging
import os
import pickle
import time
from typing import List, Optional, Tuple, Union

import cirq

# Removing TensorFlow and TensorFlow Quantum imports
# import tensorflow as tf
# import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import numpy as np
import openfermion
from scipy.optimize import minimize

# Set up logging
logger = logging.getLogger(__name__)

# Import the new ansatz circuits (kept near top-level imports)
from nova.ansatz.ansatz_circuits import create_ansatz  # noqa: E402

# Import the consolidated hardware and optimization modules
try:
    from nova.core.advanced_optimizers import QuantumOptimizer, create_optimizer
    from nova.hardware.quantum_hardware_enhanced import (
        EnhancedQuantumBackend,
        EnhancedQuantumHardwareProvider,
        default_provider as enhanced_default_provider,
        enhanced_execute_with_hardware,
        enhanced_expectation_with_hardware,
    )
    from nova.hardware.quantum_hardware_interface import QuantumBackend  # Base class
    from nova.mitigation.quantum_error_mitigation import (
        ErrorMitigationStrategy,  # Base class
    )

    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(
        f"Warning: Hardware or optimizer modules not found: {e}. Hardware execution will be disabled."
    )
    HARDWARE_AVAILABLE = False

    # Define dummy classes if import fails
    class QuantumBackend:
        pass

    class EnhancedQuantumHardwareProvider:
        pass

    class EnhancedQuantumBackend:
        pass

    class ErrorMitigationStrategy:
        pass

    class QuantumOptimizer:
        pass

    def enhanced_execute_with_hardware(*args, **kwargs):
        raise NotImplementedError("Hardware execution not available.")

    def enhanced_expectation_with_hardware(*args, **kwargs):
        raise NotImplementedError("Hardware execution not available.")

    def create_optimizer(name, **kwargs):
        raise NotImplementedError(f"Optimizer '{name}' not available.")

    enhanced_default_provider = None

from nova.hardware.quantum_error_handling import QuantumHardwareError  # noqa: E402

# Import optional advanced components (handle ImportErrors)
PYSCF_AVAILABLE = False
try:
    import pyscf

    PYSCF_AVAILABLE = True
    logger.info("PySCF found, enabling advanced Hamiltonian generation.")
except ImportError:
    logger.warning(
        "PySCF not found. Hamiltonian generation will be limited to predefined H2, LiH, H2O."
    )

ADVANCED_COMPONENTS_AVAILABLE = False


class MolecularQNN:
    """
    A Quantum Neural Network implementation for estimating molecular ground state energies.

    This class builds and trains a parameterized quantum circuit to find the ground state
    energy of small molecules like H₂, LiH, and H₂O using the Variational Quantum Eigensolver (VQE)
    approach. It leverages Cirq for quantum circuit simulation and classical optimization methods
    for parameter tuning. It uses the enhanced hardware interface for hardware execution.
    """

    def __init__(
        self,
        molecule="H2",
        bond_length=0.74,
        n_qubits=None,
        depth=2,
        seed=42,
        ansatz_type="hardware_efficient",
        ansatz_kwargs=None,
        hardware_backend: Optional[
            Union[str, QuantumBackend, EnhancedQuantumBackend]
        ] = None,
        error_mitigation: Optional[Union[str, ErrorMitigationStrategy]] = None,
        optimizer: Optional[Union[str, QuantumOptimizer]] = None,
        basis: str = "sto-3g",  # Added basis set argument
        geometry: Optional[
            Union[str, List[Tuple[str, Tuple[float, float, float]]]]
        ] = None,  # Added geometry argument
    ):
        """
        Initialize the Molecular QNN.

        Args:
            molecule: Name of the molecule ('H2', 'LiH', 'H2O')
            bond_length: Bond length in Angstroms
            n_qubits: Number of qubits, if None, determined automatically from molecule
            depth: Circuit depth for the ansatz
            seed: Random seed for initialization
            ansatz_type: Type of ansatz circuit ('hardware_efficient', 'ucc', 'chea', 'hva', 'symmetry_preserving')
            ansatz_kwargs: Additional arguments for the ansatz circuit
            hardware_backend: Name or instance of quantum hardware backend (uses enhanced interface)
            error_mitigation: Name or instance of error mitigation strategy
            optimizer: Name or instance of custom optimizer
            basis: Basis set for Hamiltonian generation (e.g., 'sto-3g', '6-31g').
                   Defaults to 'sto-3g'. Requires PySCF.
            geometry: Molecular geometry. Can be a string (e.g., from PubChem) or a list of
                      tuples like [('H', (0, 0, 0)), ('H', (0, 0, 0.74))].
                      If None, uses default bond length for H2/LiH or standard H2O geometry.
                      Requires PySCF for arbitrary geometries.

        Raises:
            ValueError: If the molecule or ansatz type is invalid.
            ImportError: If required libraries (like PySCF) are missing for the requested feature.
        """
        # Set random seed
        np.random.seed(seed)

        self.logger = logging.getLogger(f"{__name__}.MolecularQNN")
        self.molecule = molecule.lower()
        self.bond_length = (
            bond_length  # Still used for simple H2/LiH if geometry not given
        )
        self.depth = depth
        self.seed = seed
        self.ansatz_type = ansatz_type
        self.ansatz_kwargs = ansatz_kwargs if ansatz_kwargs is not None else {}
        self.basis = basis
        self._geometry = geometry  # Store explicitly provided geometry
        self.hamiltonian = None
        self.n_qubits = n_qubits
        self.ansatz_circuit = None
        self.params = None

        # Add a description attribute for the GUI
        self.description = f"Molecular QNN for {molecule} (bond length: {bond_length} Å) using {ansatz_type} ansatz"

        # Hardware integration settings using enhanced interface
        self._backend_input = hardware_backend  # Store the user input
        self._error_mitigation_input = error_mitigation
        self._optimizer_input = optimizer
        self.use_hardware = hardware_backend is not None
        self.hardware_provider = enhanced_default_provider  # Use the default provider
        self._backend: Optional[Union[QuantumBackend, EnhancedQuantumBackend]] = None
        self._error_mitigation: Optional[ErrorMitigationStrategy] = None
        self._optimizer: Optional[QuantumOptimizer] = None
        self.measurements_per_circuit = 1000  # Default shots count

        # Set up hardware/optimizer if specified
        if HARDWARE_AVAILABLE:
            if self.use_hardware:
                self.set_hardware_backend(
                    self._backend_input, self._error_mitigation_input
                )
            if self._optimizer_input:
                self.set_optimizer(self._optimizer_input)
        elif self.use_hardware or self._optimizer_input:
            print(
                "Warning: Hardware/Optimizer specified but modules are not available. Disabling related features."
            )
            self.use_hardware = False
            self._optimizer_input = None

        # Initialize Hamiltonian and Qubit count
        self._setup_molecule_hamiltonian()
        if self.n_qubits is None:
            self.n_qubits = self._get_n_qubits_from_molecule()
            self.logger.info(
                f"Determined qubit count based on molecule and basis: {self.n_qubits}"
            )

        # Initialize ansatz circuit
        self._initialize_ansatz()

        # Initialize energy history and parameters
        self.energy_history = []
        self.training_time = 0
        self.params = self.random_params()  # Initialize with random parameters
        self.final_params = None
        self.final_energy = None
        self.current_energy = None
        self._update_param_resolver()

    def _get_n_qubits_from_molecule(self) -> int:
        """Determine the number of qubits required based on the molecule and basis.
        Uses PySCF if available for accurate calculation.
        """
        if PYSCF_AVAILABLE and self.hamiltonian:
            # If Hamiltonian was generated via PySCF, use its qubit count
            return self.hamiltonian.n_qubits

        # Fallback for predefined molecules if PySCF is not available or failed
        if self.molecule == "h2":
            return 4
        elif self.molecule == "lih":
            return 12  # Common active space size
        elif self.molecule == "h2o":
            return 14  # Common active space size
        elif self.molecule == "beh2":
            return 14  # Example, adjust based on desired active space/basis
        elif self.molecule == "ch4":
            return 18  # Example, adjust based on desired active space/basis
        else:
            self.logger.warning(
                f"Cannot determine qubit count for {self.molecule} without PySCF. Defaulting to 4."
            )
            return 4

    def _setup_molecule_hamiltonian(self):
        """Set up the molecular Hamiltonian using PySCF if available, or fallback methods."""
        self.logger.info(
            f"Setting up Hamiltonian for {self.molecule} with basis {self.basis}"
        )
        start_time = time.time()

        if PYSCF_AVAILABLE:
            try:
                # --- PySCF Integration ---
                mol_geometry = self._get_molecule_geometry()
                if not mol_geometry:
                    raise ValueError(
                        f"Could not determine geometry for molecule {self.molecule}"
                    )

                self.logger.debug(f"Using geometry: {mol_geometry}")

                # Create PySCF molecule object
                mol = pyscf.gto.M(
                    atom=mol_geometry, basis=self.basis, verbose=0
                )  # verbose=0 to silence PySCF output

                # Run Hartree-Fock
                mf = pyscf.scf.RHF(mol)
                mf.kernel()

                # Calculate FCI energy for reference (can be slow for larger molecules)
                # Consider making this optional or using CCSD(T) as a cheaper reference
                try:
                    fci = pyscf.fci.FCI(mf)
                    self.exact_energy = fci.kernel()[0]
                    self.logger.info(
                        f"Calculated FCI energy (Reference): {self.exact_energy:.8f} Hartree"
                    )
                except Exception as fci_err:
                    self.logger.warning(
                        f"FCI calculation failed ({fci_err}). Trying CCSD(T)..."
                    )
                    try:
                        cc = pyscf.cc.CCSD(mf)
                        et = cc.kernel()[1]  # Get T2 diagnostic correlation energy
                        self.exact_energy = mf.e_tot + cc.e_corr + et  # Approx CCSD(T)
                        self.logger.info(
                            f"Using CCSD(T) energy (Reference): {self.exact_energy:.8f} Hartree"
                        )
                    except Exception as cc_err:
                        self.logger.warning(
                            f"CCSD(T) calculation also failed ({cc_err}). No exact energy reference available."
                        )
                        self.exact_energy = None

                # Create OpenFermion MolecularData object
                mol_data = openfermion.chem.MolecularData(
                    geometry=mol_geometry, basis=self.basis, multiplicity=1
                )
                mol_data.hf_energy = mf.e_tot
                # Add integrals (note: PySCF uses chemist notation, OpenFermion physics notation - need conversion)
                # For simplicity, let OpenFermion handle generation from PySCF object directly if possible
                # OR use PySCF tools to get integrals and populate mol_data manually
                # Simplified approach: Let OpenFermion run PySCF (can be redundant if HF already run)
                mol_data.pyscf_method = "RHF"  # Indicate which SCF method to use
                # mol_data.run_pyscf() # This runs pyscf internally

                # --- Get Hamiltonian from PySCF results ---
                # Generate Molecular Hamiltonian (Fermionic)
                # Need 1- and 2-body integrals from PySCF mf object
                h1 = mf.get_hcore()
                h2 = mf.mol.intor("int2e", aosym="s1")  # Get 2-electron integrals
                from openfermion.chem.integrals import general_basis_change

                mo_coeff = mf.mo_coeff
                n_orbitals = mo_coeff.shape[1]

                h1_mo = general_basis_change(h1, mo_coeff, (1, 0))
                h2_mo = np.einsum(
                    "pi,qj,pqrs,rk,sl->ijkl",
                    mo_coeff,
                    mo_coeff,
                    h2,
                    mo_coeff,
                    mo_coeff,
                    optimize=True,
                )

                nuclear_repulsion = mol.energy_nuc()
                self.logger.debug(f"Nuclear Repulsion Energy: {nuclear_repulsion}")

                # Create OpenFermion MolecularData object and populate it
                mol_data = openfermion.chem.MolecularData(
                    geometry=mol_geometry,
                    basis=self.basis,
                    multiplicity=1,
                    charge=mol.charge,
                    filename=f"molecule_{self.molecule}_{self.basis}.hdf5",  # Optional filename
                )
                mol_data.hf_energy = mf.e_tot
                mol_data.nuclear_repulsion = nuclear_repulsion
                mol_data.n_orbitals = n_orbitals
                mol_data.n_qubits = 2 * n_orbitals  # For standard mappings
                mol_data.one_body_integrals = h1_mo
                mol_data.two_body_integrals = h2_mo
                mol_data.fci_energy = self.exact_energy  # Store reference energy
                # mol_data.save() # Optionally save molecule data

                # Get the Fermionic Hamiltonian
                fermion_hamiltonian = mol_data.get_molecular_hamiltonian()

                # Map to Qubit Hamiltonian (Jordan-Wigner)
                qubit_hamiltonian = openfermion.transforms.jordan_wigner(
                    fermion_hamiltonian
                )
                qubit_hamiltonian = openfermion.transforms.simplify(qubit_hamiltonian)

                self.hamiltonian = qubit_hamiltonian
                self.n_qubits = (
                    qubit_hamiltonian.n_qubits
                )  # Update n_qubits based on Hamiltonian
                self.logger.info(
                    f"Successfully generated qubit Hamiltonian with {self.n_qubits} qubits."
                )
                self.logger.debug(f"Hamiltonian terms: {len(qubit_hamiltonian.terms)}")

            except Exception as e:
                self.logger.error(
                    f"Hamiltonian generation using PySCF failed: {e}. Falling back to predefined Hamiltonians.",
                    exc_info=True,
                )
                self.hamiltonian = None  # Ensure fallback is triggered
                self.exact_energy = None

        # Fallback to predefined Hamiltonians if PySCF failed or is unavailable
        if self.hamiltonian is None:
            self.logger.warning("Using fallback Hamiltonian generation.")
            if self.molecule == "h2":
                self.hamiltonian = self._h2_hamiltonian(self.bond_length)
                self.n_qubits = 4
                self.exact_energy = (
                    -1.137
                )  # Approximate FCI energy for STO-3G at 0.74 Å
            elif self.molecule == "lih":
                self.hamiltonian = self._lih_hamiltonian(self.bond_length)
                self.n_qubits = 12
                self.exact_energy = -8.06  # Approximate
            elif self.molecule == "h2o":
                self.hamiltonian = self._h2o_hamiltonian()
                self.n_qubits = 14
                self.exact_energy = -76.3  # Approximate
            # Add fallbacks for new molecules if needed, otherwise they won't work without PySCF
            elif self.molecule in ["beh2", "ch4"]:
                raise ValueError(
                    f"Hamiltonian for {self.molecule} requires PySCF installation."
                )
            else:
                raise ValueError(
                    f"Unknown molecule: {self.molecule}. Add definition or install PySCF."
                )
            self.logger.info(
                f"Using predefined Hamiltonian for {self.molecule} with {self.n_qubits} qubits."
            )

        # Ensure we have a Hamiltonian
        if self.hamiltonian is None:
            raise RuntimeError("Failed to set up molecular Hamiltonian.")

        elapsed_time = time.time() - start_time
        self.logger.info(f"Hamiltonian setup completed in {elapsed_time:.3f} seconds.")

    def _get_molecule_geometry(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        """Determine the molecular geometry based on inputs."""
        if self._geometry:  # Use explicitly provided geometry
            if isinstance(self._geometry, str):
                # Attempt to parse standard formats if needed, e.g., XYZ format
                # For now, assume it's pre-parsed or handle specific formats
                self.logger.warning(
                    "Parsing geometry from string is not implemented. Expecting list format."
                )
                return None  # Or raise error
            return self._geometry
        else:  # Generate default geometries
            if self.molecule == "h2":
                d = self.bond_length / 2.0
                return [("H", (0.0, 0.0, -d)), ("H", (0.0, 0.0, d))]
            elif self.molecule == "lih":
                return [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, self.bond_length))]
            elif self.molecule == "h2o":
                # Standard H2O geometry (Angstroms)
                return [
                    ("O", (0.000000, 0.000000, 0.117790)),
                    ("H", (0.000000, 0.755453, -0.471160)),
                    ("H", (0.000000, -0.755453, -0.471160)),
                ]
            elif self.molecule == "beh2":
                # Linear BeH2 geometry (Angstroms), example bond length
                d = 1.34 / 2.0
                return [
                    ("Be", (0.0, 0.0, 0.0)),
                    ("H", (0.0, 0.0, -d)),
                    ("H", (0.0, 0.0, d)),
                ]
            elif self.molecule == "ch4":
                # Tetrahedral CH4 geometry (Angstroms), example bond length
                bond_len = 1.087
                r = bond_len * np.sqrt(2.0 / 9.0)
                phi = 2.0 * np.pi / 3.0
                return [
                    ("C", (0, 0, 0)),
                    ("H", (0, 0, bond_len)),
                    ("H", (r * np.cos(0 * phi), r * np.sin(0 * phi), -bond_len / 3)),
                    ("H", (r * np.cos(1 * phi), r * np.sin(1 * phi), -bond_len / 3)),
                    ("H", (r * np.cos(2 * phi), r * np.sin(2 * phi), -bond_len / 3)),
                ]
            else:
                return None

    # --- Fallback Hamiltonian methods for predefined molecules ---
    def _h2_hamiltonian(self, distance):
        """Create a simplified H2 Hamiltonian."""
        # Create a simple 4-qubit H2 Hamiltonian
        # This is a simplified version for testing
        h2_op = openfermion.QubitOperator()

        # Add main terms (simplified)
        h2_op += openfermion.QubitOperator("", -1.0523732)  # Identity
        h2_op += openfermion.QubitOperator("Z0", -0.39793742)
        h2_op += openfermion.QubitOperator("Z1", -0.39793742)
        h2_op += openfermion.QubitOperator("Z2", -0.01128010)
        h2_op += openfermion.QubitOperator("Z3", -0.01128010)
        h2_op += openfermion.QubitOperator("Z0 Z1", -0.01128010)
        h2_op += openfermion.QubitOperator("Z0 Z2", 0.18093120)
        h2_op += openfermion.QubitOperator("Z0 Z3", 0.18093120)
        h2_op += openfermion.QubitOperator("Z1 Z2", 0.18093120)
        h2_op += openfermion.QubitOperator("Z1 Z3", 0.18093120)
        h2_op += openfermion.QubitOperator("Z2 Z3", -0.01128010)
        h2_op += openfermion.QubitOperator("X0 X1 Y2 Y3", -0.04544288)
        h2_op += openfermion.QubitOperator("X0 Y1 Y2 X3", 0.04544288)
        h2_op += openfermion.QubitOperator("Y0 X1 X2 Y3", 0.04544288)
        h2_op += openfermion.QubitOperator("Y0 Y1 X2 X3", -0.04544288)

        return h2_op

    def _lih_hamiltonian(self, distance):
        """Create a simplified LiH Hamiltonian."""
        # Simplified LiH Hamiltonian for 12 qubits
        lih_op = openfermion.QubitOperator()

        # Add main diagonal terms
        lih_op += openfermion.QubitOperator("", -7.8629)  # Identity
        for i in range(12):
            lih_op += openfermion.QubitOperator(f"Z{i}", 0.1 * (-1) ** (i % 2))

        # Add some coupling terms
        for i in range(11):
            lih_op += openfermion.QubitOperator(f"Z{i} Z{i + 1}", 0.05)

        return lih_op

    def _h2o_hamiltonian(self):
        """Create a simplified H2O Hamiltonian."""
        # Simplified H2O Hamiltonian for 14 qubits
        h2o_op = openfermion.QubitOperator()

        # Add main diagonal terms
        h2o_op += openfermion.QubitOperator("", -75.0)  # Identity
        for i in range(14):
            h2o_op += openfermion.QubitOperator(f"Z{i}", 0.2 * (-1) ** (i % 3))

        # Add some coupling terms
        for i in range(13):
            h2o_op += openfermion.QubitOperator(f"Z{i} Z{i + 1}", 0.1)

        return h2o_op

    def _initialize_ansatz(self):
        """Initialize the chosen ansatz circuit."""
        self.logger.info(
            f"Initializing {self.ansatz_type} ansatz with depth {self.depth}"
        )

        # Create qubits
        self.qubits = cirq.LineQubit.range(self.n_qubits)

        # Prepare kwargs for ansatz creation
        ansatz_params = {"depth": self.depth, **self.ansatz_kwargs}

        # Only add hamiltonian_terms for HVA ansatz
        if self.ansatz_type.lower() == "hva":
            if self.hamiltonian is not None:
                # Extract Pauli terms from the Hamiltonian
                hamiltonian_terms = []
                if hasattr(self.hamiltonian, "terms"):
                    # OpenFermion QubitOperator
                    for term, _coeff in self.hamiltonian.terms.items():
                        if term:  # Skip identity term
                            pauli_string = cirq.PauliString()
                            for qubit_idx, pauli in term:
                                if pauli == "X":
                                    pauli_string *= cirq.X(self.qubits[qubit_idx])
                                elif pauli == "Y":
                                    pauli_string *= cirq.Y(self.qubits[qubit_idx])
                                elif pauli == "Z":
                                    pauli_string *= cirq.Z(self.qubits[qubit_idx])
                            hamiltonian_terms.append(pauli_string)
                    ansatz_params["hamiltonian_terms"] = hamiltonian_terms
                else:
                    # Fallback: create simple Z-Z terms
                    hamiltonian_terms = []
                    for i in range(self.n_qubits - 1):
                        hamiltonian_terms.append(
                            cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i + 1])
                        )
                    ansatz_params["hamiltonian_terms"] = hamiltonian_terms
            else:
                # No Hamiltonian available, create default terms
                hamiltonian_terms = []
                for i in range(self.n_qubits - 1):
                    hamiltonian_terms.append(
                        cirq.Z(self.qubits[i]) * cirq.Z(self.qubits[i + 1])
                    )
                ansatz_params["hamiltonian_terms"] = hamiltonian_terms

        # Create the ansatz circuit
        self.ansatz_circuit = create_ansatz(
            self.ansatz_type, self.qubits, **ansatz_params
        )

        # Build the circuit
        self.circuit = self.ansatz_circuit.build_circuit()

        # Get parameter symbols
        self.symbols = sorted(cirq.parameter_names(self.circuit), key=lambda x: int(x.replace("θ_", "")))

        self.logger.info(f"Ansatz initialized with {len(self.symbols)} parameters")

    def random_params(self):
        """Generate random initial parameters."""
        return np.random.random(len(self.symbols)) * 2 * np.pi

    def param_count(self):
        """Get the number of parameters in the ansatz."""
        return len(self.symbols)

    def train(
        self, iterations=100, method="BFGS", verbose=True, callback=None, shots=None
    ):
        """
        Train the QNN to find the ground state energy.

        Args:
            iterations: Maximum number of optimization iterations
            method: Optimization method ('BFGS', 'COBYLA', etc.)
            verbose: Whether to print training progress
            callback: Optional callback function called after each iteration
            shots: Number of measurement shots for hardware execution

        Returns:
            Dictionary containing training results
        """
        start_time = time.time()

        if verbose:
            print(f"Starting training with {method} optimizer...")
            print(f"Initial energy: {self.get_energy():.6f} Hartree")

        # Define objective function
        def objective_fn(params):
            return self._energy_expectation(params, shots)

        # Internal callback for tracking energy history
        def internal_callback(params):
            energy = objective_fn(params)
            self.energy_history.append(energy)
            self.params = params.copy()

            if callback:
                callback(params)

        # Choose optimizer
        try:
            optimizer_to_use = None
            if self._optimizer:
                optimizer_to_use = self._optimizer
            elif (
                HARDWARE_AVAILABLE
                and isinstance(method, str)
                and method
                not in ["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell", "COBYLA"]
            ):
                # Try to create advanced optimizer by name
                try:
                    optimizer_to_use = create_optimizer(method)
                    if verbose:
                        print(f"Using advanced optimizer: {optimizer_to_use.name}")
                except ValueError:
                    if verbose:
                        print(
                            f"Optimizer '{method}' not found in advanced optimizers, falling back to SciPy."
                        )
                    optimizer_to_use = None  # Fallback to SciPy

            if optimizer_to_use:
                # Use the QuantumOptimizer instance
                result_dict = optimizer_to_use.minimize(
                    objective_fn,
                    self.params,
                    max_iterations=iterations,
                    callback=internal_callback,
                )
                self.final_params = result_dict["parameters"]
                self.final_energy = result_dict["energy"]
                self.energy_history = result_dict[
                    "energy_history"
                ]  # Optimizer might provide its own history
                success = result_dict.get("success", True)
                actual_iterations = len(self.energy_history)
                optimizer_name = optimizer_to_use.name

            else:
                # Use SciPy minimize
                optimizer_name = method
                if verbose:
                    print(f"Using SciPy optimizer: {optimizer_name}")
                options = {
                    "maxiter": iterations,
                    "disp": False,
                }  # Display is handled by callback

                # Callback wrapper for SciPy
                scipy_callback_wrapper = None
                if callback:

                    def scipy_callback(intermediate_result):
                        internal_callback(intermediate_result.x)

                    scipy_callback_wrapper = scipy_callback
                else:
                    scipy_callback_wrapper = internal_callback

                try:
                    result = minimize(
                        objective_fn,
                        self.params,
                        method=optimizer_name,
                        callback=scipy_callback_wrapper,
                        options=options,
                    )
                    self.final_params = result.x
                    self.final_energy = result.fun
                    success = result.success
                    actual_iterations = result.nit
                    if not success and verbose:
                        print(
                            f"Warning: SciPy optimization might not have converged: {result.message}"
                        )
                except ValueError as e:
                    if "Unknown solver" in str(e):
                        if verbose:
                            print(
                                f"Unknown SciPy solver: {optimizer_name}. Falling back to BFGS."
                            )
                        optimizer_name = "BFGS"
                        result = minimize(
                            objective_fn,
                            self.params,
                            method=optimizer_name,
                            callback=scipy_callback_wrapper,
                            options=options,
                        )
                        self.final_params = result.x
                        self.final_energy = result.fun
                        success = result.success
                        actual_iterations = result.nit
                        if not success and verbose:
                            print(
                                f"Warning: SciPy optimization (BFGS fallback) might not have converged: {result.message}"
                            )
                    else:
                        raise

            # Update QNN state
            self.params = self.final_params
            self._update_param_resolver()

            # Calculate training time
            end_time = time.time()
            self.training_time = end_time - start_time

            if verbose:
                print(
                    f"Training completed in {self.training_time:.2f} seconds using {optimizer_name}"
                )
                print(
                    f"Final energy: {self.final_energy:.6f} Hartree after {actual_iterations} iterations"
                )

                if self.exact_energy is not None:
                    error = abs(self.final_energy - self.exact_energy)
                    error_pct = (
                        100 * error / abs(self.exact_energy)
                        if abs(self.exact_energy) > 1e-9
                        else float("inf")
                    )
                    print(f"Exact energy: {self.exact_energy:.6f} Hartree")
                    print(f"Error: {error:.6f} Hartree ({error_pct:.4f}%)")

            # Return results
            return {
                "energy": self.final_energy,
                "parameters": self.final_params,
                "energy_history": self.energy_history,
                "training_time": self.training_time,
                "success": success,
                "method": optimizer_name,
                "iterations": actual_iterations,
            }

        except KeyboardInterrupt:
            # Handle user interruption gracefully
            end_time = time.time()
            self.training_time = end_time - start_time
            if verbose:
                print("\nTraining interrupted by user")

            # Save partial results if we have any
            if len(self.energy_history) > 0:
                self.final_energy = self.energy_history[-1]

                if verbose:
                    print(
                        f"Partial results - Final energy evaluated: {self.final_energy:.6f} Hartree"
                    )

                return {
                    "energy": self.final_energy,
                    "parameters": self.params,
                    "energy_history": self.energy_history,
                    "training_time": self.training_time,
                    "success": False,
                    "method": method,
                    "message": "Interrupted by user",
                }
            else:
                raise ValueError(
                    "Training interrupted before any iterations were completed"
                )

        except MemoryError:
            raise MemoryError(
                "Out of memory during training. Try reducing the number of qubits or circuit depth."
            )

        except Exception as e:
            if verbose:
                print(f"Error during training: {str(e)}")

            # Provide additional context for common errors
            if isinstance(e, QuantumHardwareError):
                raise RuntimeError(f"Hardware execution error: {str(e)}") from e
            else:
                # Print traceback for unexpected errors
                import traceback

                traceback.print_exc()
                raise

    def _energy_expectation(self, param_values, shots=None):
        """
        Calculate the energy expectation value for a given set of parameters.

        Args:
            param_values: Parameter values for the circuit
            shots: Number of measurement shots (for hardware execution)

        Returns:
            Energy expectation value
        """
        if self.use_hardware:
            # Use enhanced hardware expectation function
            if not HARDWARE_AVAILABLE:
                raise RuntimeError(
                    "Hardware execution requested but modules are not available."
                )
            try:
                local_shots = shots or self.measurements_per_circuit
                # Resolve parameters for the circuit
                param_resolver = cirq.ParamResolver(
                    dict(zip(self.symbols, param_values))
                )
                resolved_circuit = cirq.resolve_parameters(self.circuit, param_resolver)

                # Calculate expectation using the enhanced function
                energy = enhanced_expectation_with_hardware(
                    circuit=resolved_circuit,
                    observable=self.hamiltonian,  # Pass the PauliSum
                    backend=self._backend,  # Pass backend instance
                    error_mitigation=self._error_mitigation,  # Pass strategy instance
                    repetitions=local_shots,
                    provider=self.hardware_provider,
                )
                return energy
            except Exception as e:
                # Re-raise hardware errors with more context
                raise QuantumHardwareError(
                    f"Hardware error during energy calculation: {str(e)}"
                ) from e
        else:
            # Use simulator
            return self._simulator_energy_expectation(param_values)

    def _simulator_energy_expectation(self, param_values):
        """
        Calculate the energy expectation value using a simulator.

        Args:
            param_values: Parameter values for the ansatz circuit

        Returns:
            float: Energy expectation value
        """
        try:
            # Assign parameters to the circuit
            param_resolver = cirq.ParamResolver(dict(zip(self.symbols, param_values)))
            resolved_circuit = cirq.resolve_parameters(self.circuit, param_resolver)

            # Simulate the circuit to get the final state
            simulator = cirq.Simulator()
            result = simulator.simulate(resolved_circuit)
            final_state_vector = result.final_state_vector

            # Calculate the energy expectation value using the Hamiltonian
            # Manually compute expectation values using matrix operations

            if hasattr(self.hamiltonian, "terms"):
                # OpenFermion QubitOperator - convert to expectation value manually
                energy = 0.0
                for term, coeff in self.hamiltonian.terms.items():
                    if not term:  # Identity term
                        energy += coeff.real
                    else:
                        # Calculate expectation for this Pauli term manually
                        # Convert to matrix representation for expectation value calculation
                        term_matrix = np.eye(2 ** len(self.qubits), dtype=complex)

                        for qubit_idx, pauli in term:
                            # Create single-qubit Pauli matrix
                            if pauli == "X":
                                pauli_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
                            elif pauli == "Y":
                                pauli_matrix = np.array(
                                    [[0, -1j], [1j, 0]], dtype=complex
                                )
                            elif pauli == "Z":
                                pauli_matrix = np.array(
                                    [[1, 0], [0, -1]], dtype=complex
                                )
                            else:
                                pauli_matrix = np.eye(2, dtype=complex)

                            # Tensor product to get full matrix
                            full_matrix = np.eye(1, dtype=complex)
                            for i in range(len(self.qubits)):
                                if i == qubit_idx:
                                    full_matrix = np.kron(full_matrix, pauli_matrix)
                                else:
                                    full_matrix = np.kron(
                                        full_matrix, np.eye(2, dtype=complex)
                                    )

                            term_matrix = term_matrix @ full_matrix

                        # Calculate expectation value <ψ|term|ψ>
                        term_expectation = (
                            np.conj(final_state_vector)
                            @ term_matrix
                            @ final_state_vector
                        )
                        energy += (coeff * term_expectation).real

                return energy
            else:
                # For direct Cirq observables, use a simpler approach
                # This is a fallback - the OpenFermion path should be the main one
                return 0.0  # Placeholder

        except MemoryError as e:
            raise MemoryError(
                "Simulator ran out of memory. Circuit size likely too large."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error during simulator energy calculation: {str(e)}"
            ) from e

    def get_energy(self):
        """
        Get the current energy estimate for the optimized parameters.

        Returns:
            float: The estimated ground state energy in Hartree, or None if not calculated.
        """
        if self.final_energy is not None:
            return float(self.final_energy)
        elif self.energy_history:
            return float(self.energy_history[-1])
        else:
            # Calculate energy with current (potentially initial random) parameters
            try:
                return float(
                    self._energy_expectation(
                        self.params, shots=self.measurements_per_circuit
                    )
                )
            except Exception as e:
                print(f"Warning: Could not calculate initial energy: {e}")
                return None

    def get_circuit(self, resolved=True):
        """
        Get the quantum circuit with or without resolved parameters.

        Args:
            resolved: Whether to resolve the circuit with the current parameters

        Returns:
            cirq.Circuit: The quantum circuit
        """
        params_to_resolve = (
            self.final_params
            if resolved and self.final_params is not None
            else self.params
        )
        if resolved:
            try:
                param_dict = dict(zip(self.symbols, params_to_resolve))
                return cirq.resolve_parameters(self.circuit, param_dict)
            except Exception as e:
                print(
                    f"Warning: Could not resolve parameters: {e}. Returning unresolved circuit."
                )
                return self.circuit  # Fallback to unresolved
        else:
            return self.circuit

    def plot_energy_convergence(self, save_path=None):
        """
        Plot the energy convergence during training.

        Args:
            save_path: Path to save the plot

        Returns:
            matplotlib.figure.Figure: The generated figure, or None if no history.
        """
        if not self.energy_history:
            print("No energy history to plot. Train the model first.")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot energy convergence
        iterations = range(1, len(self.energy_history) + 1)
        ax.plot(
            iterations, self.energy_history, "b-", label=f"{self.ansatz_type} ansatz"
        )

        # Add exact energy line if available
        if hasattr(self, "exact_energy") and self.exact_energy is not None:
            ax.axhline(
                y=self.exact_energy,
                color="r",
                linestyle="--",
                label=f"Exact: {self.exact_energy:.6f}",
            )

        # Format the plot
        ax.set_title(
            f"Energy Convergence for {self.molecule} (Bond Length: {self.bond_length} Å)"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.legend()
        ax.grid(True)

        # Save the plot if requested
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        return fig

    def compare_ansatz_types(
        self,
        iterations=50,
        methods=None,
        save_path=None,
        callback=None,
        hardware_backend=None,
        error_mitigation=None,
        optimizer=None,
    ):
        """
        Compare different ansatz types for the same molecule and bond length.

        Args:
            iterations: Number of optimization iterations for each method
            methods: List of ansatz types to compare
            save_path: Path to save the comparison plot
            callback: Optional callback function to report progress (receives dict with method, iter, energy)
            hardware_backend: Quantum hardware backend to use for comparison (name or instance)
            error_mitigation: Error mitigation strategy to use (name or instance)
            optimizer: Optimizer to use for training (name or instance)

        Returns:
            results: Dictionary containing the results for each method
            fig: Matplotlib figure showing the convergence of each method
        """
        if methods is None:
            methods = ["hardware_efficient", "ucc", "chea", "hva"]
        comparison_results = {}
        if self.exact_energy is not None:
            comparison_results["exact_energy"] = self.exact_energy

        # Store original settings
        original_ansatz_type = self.ansatz_type
        original_ansatz_kwargs = self.ansatz_kwargs
        original_backend_input = self._backend_input
        original_em_input = self._error_mitigation_input
        original_optimizer_input = self._optimizer_input
        original_params = self.params.copy()  # Save initial params

        # Initialize the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Compare each method
        for method in methods:
            print(f"\n--- Testing Ansatz Type: {method} ---")
            _method_start_time = time.time()
            method_results = {
                "param_count": 0,
                "error": None,
                "energy_history": [],
                "training_time": 0,
            }

            try:
                # Create a temporary QNN instance for this method
                # Ensures clean slate for parameters and ansatz structure
                temp_qnn = MolecularQNN(
                    molecule=self.molecule,
                    bond_length=self.bond_length,
                    depth=self.depth,
                    seed=self.seed,
                    ansatz_type=method,
                    ansatz_kwargs=original_ansatz_kwargs,  # Use original kwargs if needed
                    hardware_backend=hardware_backend,  # Pass through comparison settings
                    error_mitigation=error_mitigation,
                    optimizer=optimizer,
                )
                method_results["param_count"] = temp_qnn.param_count

                # Define callback for this specific method
                def method_specific_callback(progress_info, _method=method):
                    if callback:
                        progress_info["method"] = _method  # Add method name to info
                        callback(progress_info)

                # Train the temporary QNN instance
                train_results = temp_qnn.train(
                    iterations=iterations,
                    verbose=False,  # Suppress verbose output from sub-train
                    callback=method_specific_callback,
                )

                # Store results
                method_results["final_energy"] = train_results["energy"]
                method_results["final_params"] = train_results["parameters"]
                method_results["energy_history"] = train_results["energy_history"]
                method_results["training_time"] = train_results["training_time"]
                method_results["success"] = train_results["success"]

                print(
                    f"  {method}: Final Energy = {method_results['final_energy']:.6f} Hartree, Time = {method_results['training_time']:.2f}s"
                )

                # Plot the convergence curve
                ax.plot(
                    range(1, len(method_results["energy_history"]) + 1),
                    method_results["energy_history"],
                    label=f"{method} ({method_results['param_count']} params, final: {method_results['final_energy']:.4f})",
                )

            except Exception as e:
                print(f"  Error testing {method}: {e}")
                method_results["error"] = str(e)
                # Print traceback for debugging
                import traceback

                traceback.print_exc()
            finally:
                comparison_results[method] = method_results

        # Restore original settings to the main instance
        self.ansatz_type = original_ansatz_type
        self.ansatz_kwargs = original_ansatz_kwargs
        self.set_hardware_backend(original_backend_input, original_em_input)
        self.set_optimizer(original_optimizer_input)
        self.params = original_params  # Restore original params
        self._initialize_ansatz()  # Re-initialize original ansatz structure
        self._update_param_resolver()

        # Add exact energy line if available
        if "exact_energy" in comparison_results:
            ax.axhline(
                y=comparison_results["exact_energy"],
                color="r",
                linestyle="--",
                label=f"Exact: {comparison_results['exact_energy']:.6f}",
            )

        # Format the plot
        ax.set_title(
            f"Energy Convergence Comparison for {self.molecule} (Bond Length: {self.bond_length} Å)"
        )
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy (Hartree)")
        ax.legend(loc="upper right")
        ax.grid(True)

        # Save the plot if requested
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Comparison plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving comparison plot: {e}")

        return comparison_results, fig

    def save_model(self, filepath):
        """
        Save the QNN model state to a file.

        Args:
            filepath: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Prepare data for saving
            # Store only the essential state, not hardware instances
            save_data = {
                "molecule": self.molecule,
                "bond_length": self.bond_length,
                "n_qubits": self.n_qubits,
                "depth": self.depth,
                "seed": self.seed,
                "ansatz_type": self.ansatz_type,
                "ansatz_kwargs": self.ansatz_kwargs,
                "symbols": self.symbols,  # Save the parameter symbols
                "params": self.params,  # Current parameters
                "final_params": self.final_params,  # Parameters after last training
                "final_energy": self.final_energy,
                "energy_history": self.energy_history,
                "training_time": self.training_time,
                "exact_energy": self.exact_energy,
                # Convert Hamiltonian PauliSum to a serializable format (list of tuples)
                "hamiltonian_terms": [
                    (str(term), float(coeff))
                    for term, coeff in self.hamiltonian.terms.items()
                ],
                # Save hardware/optimizer *inputs* for potential re-initialization
                "hardware_backend_input": self._backend_input,
                "error_mitigation_input": self._error_mitigation_input,
                "optimizer_input": self._optimizer_input,
                "measurements_per_circuit": self.measurements_per_circuit,
            }

            # Save model using pickle
            with open(filepath, "wb") as f:
                pickle.dump(save_data, f)
            print(f"Model state saved to {filepath}")
            return True

        except (OSError, PermissionError) as e:
            print(f"Error saving model: Permission denied or I/O error: {str(e)}")
            return False
        except Exception as e:
            print(f"Error saving model state: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    @classmethod
    def load_model(cls, filepath):
        """
        Load a QNN model state from a file.

        Args:
            filepath: Path to the model file

        Returns:
            MolecularQNN instance, or None if loading fails
        """
        try:
            # Load the saved data
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)  # noqa: S301
            print(f"Model data loaded from {filepath}")

            # Create a new QNN instance using saved configuration
            # Pass hardware/optimizer inputs directly to constructor
            qnn = cls(
                molecule=save_data["molecule"],
                bond_length=save_data["bond_length"],
                n_qubits=save_data.get("n_qubits"),  # Let constructor handle None
                depth=save_data.get("depth", 2),
                seed=save_data.get("seed", 42),
                ansatz_type=save_data.get("ansatz_type", "hardware_efficient"),
                ansatz_kwargs=save_data.get("ansatz_kwargs", {}),
                hardware_backend=save_data.get("hardware_backend_input"),
                error_mitigation=save_data.get("error_mitigation_input"),
                optimizer=save_data.get("optimizer_input"),
            )

            # Restore saved state (parameters, history, etc.)
            # Ensure parameter count matches before assigning
            loaded_symbols = save_data.get("symbols")
            if loaded_symbols == qnn.symbols:
                qnn.params = save_data.get("params", qnn.params)
                qnn.final_params = save_data.get("final_params")
            else:
                print(
                    "Warning: Loaded parameter symbols mismatch current ansatz. Using initial random parameters."
                )

            qnn.final_energy = save_data.get("final_energy")
            qnn.energy_history = save_data.get("energy_history", [])
            qnn.training_time = save_data.get("training_time", 0)
            qnn.exact_energy = save_data.get("exact_energy")
            qnn.measurements_per_circuit = save_data.get(
                "measurements_per_circuit", 1000
            )

            # Reconstruct Hamiltonian (optional, as constructor does this)
            # If needed: qnn.hamiltonian = cirq.PauliSum.from_pauli_strings(...)

            # Update parameter resolver
            qnn._update_param_resolver()
            print("Model state restored successfully.")
            return qnn

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except (pickle.UnpicklingError, EOFError, TypeError, KeyError) as e:
            raise ValueError(
                f"The file {filepath} is not a valid QNN model file or is corrupted/incompatible: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading model: {str(e)}")

    def _update_param_resolver(self):
        """Update the parameter resolver with current parameters."""
        try:
            # Ensure params has the correct length
            if len(self.params) != len(self.symbols):
                print(
                    f"Warning: Parameter count mismatch ({len(self.params)} vs {len(self.symbols)} symbols). Using initial random parameters."
                )
                self.params = self.random_params()
            self.param_resolver = dict(zip(self.symbols, self.params))
        except Exception as e:
            # Fallback: generate random parameters if update fails
            print(
                f"Error updating parameter resolver: {e}. Generating random parameters."
            )
            self.params = self.random_params()
            self.param_resolver = dict(zip(self.symbols, self.params))

    def set_hardware_backend(self, hardware_backend_input, error_mitigation_input=None):
        """
        Set the quantum hardware backend for execution using the enhanced provider.

        Args:
            hardware_backend_input: Hardware backend to use (name string, base QuantumBackend, or EnhancedQuantumBackend instance)
            error_mitigation_input: Error mitigation strategy to use (name string or strategy instance)

        Returns:
            True if successful, False otherwise
        """
        if not HARDWARE_AVAILABLE:
            print("Error: Quantum hardware modules not available.")
            self.use_hardware = False
            self._backend = None
            self._error_mitigation = None
            return False

        self._backend_input = hardware_backend_input  # Store input
        self._error_mitigation_input = error_mitigation_input

        try:
            # Get the backend instance using the default provider
            if isinstance(hardware_backend_input, str):
                # Get potentially enhanced backend by name
                self._backend = enhanced_default_provider.get_backend(
                    hardware_backend_input, enhanced=True
                )
            elif isinstance(hardware_backend_input, EnhancedQuantumBackend):
                self._backend = hardware_backend_input  # Already enhanced
            elif isinstance(hardware_backend_input, QuantumBackend):
                # Wrap base backend instance
                self._backend = EnhancedQuantumBackend(hardware_backend_input)
            else:
                raise TypeError("Invalid hardware_backend type.")

            # Set error mitigation strategy if provided
            if error_mitigation_input is not None:
                if isinstance(error_mitigation_input, str):
                    self._error_mitigation = (
                        enhanced_default_provider.get_error_mitigation(
                            error_mitigation_input
                        )
                    )
                elif isinstance(error_mitigation_input, ErrorMitigationStrategy):
                    self._error_mitigation = error_mitigation_input
                else:
                    raise TypeError("Invalid error_mitigation type.")
            else:
                self._error_mitigation = None

            # Enable hardware mode
            self.use_hardware = True
            print(f"Using quantum hardware backend: {self._backend.name}")
            if self._error_mitigation:
                print(f"Using error mitigation: {self._error_mitigation}")

            return True

        except (ImportError, ValueError, TypeError, QuantumHardwareError) as e:
            print(f"Error setting hardware backend: {e}")
            self.use_hardware = False
            self._backend = None
            self._error_mitigation = None
            return False
        except Exception as e:
            print(f"Unexpected error setting hardware backend: {str(e)}")
            self.use_hardware = False
            self._backend = None
            self._error_mitigation = None
            return False

    def set_optimizer(self, optimizer_input):
        """Set the optimizer for training."""
        if (
            not HARDWARE_AVAILABLE
        ):  # Optimizer module depends on hardware availability check
            print("Error: Optimizer modules not available.")
            self._optimizer = None
            return False

        self._optimizer_input = optimizer_input  # Store input
        try:
            if isinstance(optimizer_input, str):
                self._optimizer = create_optimizer(optimizer_input)
            elif isinstance(optimizer_input, QuantumOptimizer):
                self._optimizer = optimizer_input
            elif optimizer_input is None:
                self._optimizer = None  # Allow unsetting optimizer
            else:
                raise TypeError("Invalid optimizer type.")
            if self._optimizer:
                print(f"Optimizer set to: {self._optimizer.name}")
            else:
                print("Optimizer unset (will use SciPy default).")
            return True
        except (ValueError, TypeError) as e:
            print(f"Error setting optimizer: {e}")
            self._optimizer = None
            return False

    def set_shots(self, shots):
        """Set the number of measurement shots to use for hardware execution."""
        self.measurements_per_circuit = shots
