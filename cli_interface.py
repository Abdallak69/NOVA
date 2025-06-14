#!/usr/bin/env python3

"""
Command-Line Interface for Quantum Neural Network Molecular Energy Estimation

This module provides a simple command-line interface for testing and visualizing results
from the Quantum Neural Network (QNN) for molecular energy estimation.
"""

import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Debug print
print("Starting CLI interface...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    # Import the QNN implementation
    print("Attempting to import MolecularQNN...")
    from qnn_molecular_energy import MolecularQNN

    print("Successfully imported MolecularQNN")
except ImportError as e:
    print(f"ERROR importing MolecularQNN: {e}")
    print("Please check your environment setup and dependencies.")
    sys.exit(1)

# Import hardware and optimizer modules
try:
    from advanced_optimizers import create_optimizer
    from optimizer_benchmarks import BenchmarkSuite
    from quantum_hardware import QuantumHardwareProvider

    HARDWARE_AVAILABLE = True
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Some modules could not be imported: {e}")
    HARDWARE_AVAILABLE = False
    BENCHMARK_AVAILABLE = False


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title="Quantum Neural Network for Molecular Energy Estimation"):
    """Print the application header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def print_menu():
    """Print the main menu options."""
    print("\nMain Menu:")
    print("1. Create and train a new QNN model")
    print("2. Load a saved model")
    print("3. Compare ansatz types")
    if HARDWARE_AVAILABLE:
        print("4. Run on quantum hardware")
        print("5. Advanced optimizer testing")
        print("6. View system info")
        print("7. Exit")
    else:
        print("4. View system info")
        print("5. Exit")
    print()


def get_molecule_params():
    """
    Get molecule parameters from user input.

    Returns:
        tuple: (molecule, bond_length, depth, ansatz_type, ansatz_kwargs, hardware_backend, optimizer)
    """
    print("\nSelect molecule:")
    print("1. H2 (Hydrogen)")
    print("2. LiH (Lithium Hydride)")
    print("3. H2O (Water)")

    choice = input("\nEnter choice (1-3): ")

    if choice == "1":
        molecule = "H2"
        default_bond_length = 0.74
    elif choice == "2":
        molecule = "LiH"
        default_bond_length = 1.6
    elif choice == "3":
        molecule = "H2O"
        default_bond_length = 0.96
    else:
        print("Invalid choice, using H2 as default.")
        molecule = "H2"
        default_bond_length = 0.74

    # Get bond length
    bond_length_input = input(
        f"\nEnter bond length in Angstroms (default: {default_bond_length}): "
    )
    bond_length = (
        float(bond_length_input) if bond_length_input.strip() else default_bond_length
    )

    # Get circuit depth
    depth_input = input("\nEnter circuit depth (default: 2): ")
    depth = int(depth_input) if depth_input.strip() else 2

    # Get ansatz type
    print("\nSelect ansatz type:")
    print("1. Hardware-Efficient")
    print("2. Unitary Coupled Cluster (UCC)")
    print("3. Custom Hardware-Efficient Ansatz (CHEA)")
    print("4. Symmetry-Preserving")
    print("5. Hamiltonian Variational (HVA)")

    ansatz_choice = input("\nEnter choice (1-5): ")

    if ansatz_choice == "1":
        ansatz_type = "hardware_efficient"
        # Get rotation gates
        print("\nSelect rotation gates:")
        print("1. X rotations only")
        print("2. XY rotations")
        print("3. XYZ rotations (default)")

        rotation_choice = input("\nEnter choice (1-3): ")
        if rotation_choice == "1":
            rotation_gates = "X"
        elif rotation_choice == "2":
            rotation_gates = "XY"
        else:
            rotation_gates = "XYZ"

        # Get entanglement pattern
        print("\nSelect entanglement pattern:")
        print("1. Linear (default)")
        print("2. Full")

        entangle_choice = input("\nEnter choice (1-2): ")
        if entangle_choice == "2":
            entangle_pattern = "full"
        else:
            entangle_pattern = "linear"

        ansatz_kwargs = {
            "rotation_gates": rotation_gates,
            "entangle_pattern": entangle_pattern,
        }

    elif ansatz_choice == "2":
        ansatz_type = "ucc"
        # Get single/double excitations
        use_singles = input("\nInclude single excitations? (Y/n): ").lower() != "n"
        use_doubles = input("Include double excitations? (Y/n): ").lower() != "n"

        ansatz_kwargs = {"include_singles": use_singles, "include_doubles": use_doubles}

    elif ansatz_choice == "3":
        ansatz_type = "chea"
        # Get entanglement pattern
        print("\nSelect entanglement pattern:")
        print("1. Linear (default)")
        print("2. Full")

        entangle_choice = input("\nEnter choice (1-2): ")
        if entangle_choice == "2":
            entangle_pattern = "full"
        else:
            entangle_pattern = "linear"

        ansatz_kwargs = {"entangle_pattern": entangle_pattern}

    elif ansatz_choice == "4":
        ansatz_type = "symmetry_preserving"
        # Get conservation options
        conserve_number = input("\nConserve particle number? (Y/n): ").lower() != "n"

        ansatz_kwargs = {"conserve_particle_number": conserve_number}

    elif ansatz_choice == "5":
        ansatz_type = "hva"
        print(
            "\nNote: Hamiltonian terms will be automatically generated based on the molecule."
        )
        ansatz_kwargs = {}

    else:
        print("Invalid choice, using hardware-efficient ansatz as default.")
        ansatz_type = "hardware_efficient"
        ansatz_kwargs = {"rotation_gates": "XYZ", "entangle_pattern": "linear"}

    # Get hardware backend if available
    hardware_backend = None
    optimizer = None
    if HARDWARE_AVAILABLE:
        use_hardware = input("\nUse quantum hardware? (y/N): ").lower() == "y"

        if use_hardware:
            try:
                # Create hardware provider and list available backends
                provider = QuantumHardwareProvider()
                backends = provider.list_backends()

                print("\nAvailable hardware backends:")
                for i, backend in enumerate(backends):
                    print(f"{i + 1}. {backend}")

                backend_choice = input(
                    "\nEnter backend choice (or press Enter for simulator): "
                )
                if (
                    backend_choice.strip()
                    and backend_choice.isdigit()
                    and 1 <= int(backend_choice) <= len(backends)
                ):
                    hardware_backend = backends[int(backend_choice) - 1]
                else:
                    hardware_backend = "cirq_simulator"
                    print("Using Cirq simulator as default.")

                # Ask about error mitigation
                use_error_mitigation = (
                    input("\nUse error mitigation? (y/N): ").lower() == "y"
                )
                if use_error_mitigation:
                    strategies = provider.list_error_mitigation_strategies()
                    print("\nAvailable error mitigation strategies:")
                    for i, strategy in enumerate(strategies):
                        print(f"{i + 1}. {strategy}")

                    strategy_choice = input(
                        "\nEnter strategy choice (or press Enter for none): "
                    )
                    if (
                        strategy_choice.strip()
                        and strategy_choice.isdigit()
                        and 1 <= int(strategy_choice) <= len(strategies)
                    ):
                        error_mitigation = strategies[int(strategy_choice) - 1]
                        print(f"Using {error_mitigation} error mitigation.")
                    else:
                        error_mitigation = None
                        print("No error mitigation selected.")
                else:
                    error_mitigation = None

                # Ask about optimization method
                print("\nSelect optimization method:")
                print("1. BFGS (default)")
                print("2. Noise-aware BFGS")
                print("3. Gradient-free (Nelder-Mead)")
                print("4. Gradient-free (Differential Evolution)")
                print("5. Adaptive")
                print("6. Parallel Tempering")

                opt_choice = input("\nEnter choice (1-6): ")

                if opt_choice == "2":
                    optimizer = "noise_aware"
                elif opt_choice == "3":
                    optimizer = create_optimizer("gradient_free", method="nelder-mead")
                elif opt_choice == "4":
                    optimizer = create_optimizer(
                        "gradient_free", method="differential_evolution"
                    )
                elif opt_choice == "5":
                    optimizer = "adaptive"
                elif opt_choice == "6":
                    optimizer = "parallel_tempering"
                else:
                    optimizer = "bfgs"

                print(f"Using {optimizer} optimizer.")

            except Exception as e:
                print(f"Error setting up hardware: {e}")
                print("Falling back to simulator.")
                hardware_backend = None
                error_mitigation = None
                optimizer = None

    return (
        molecule,
        bond_length,
        depth,
        ansatz_type,
        ansatz_kwargs,
        hardware_backend,
        optimizer,
    )


def get_training_params():
    """
    Get training parameters from user input.

    Returns:
        tuple: (iterations, method, shots)
    """
    # Get number of iterations
    iterations_input = input("\nEnter maximum number of iterations (default: 100): ")
    iterations = int(iterations_input) if iterations_input.strip() else 100

    # Get optimization method
    print("\nSelect optimization method:")
    print("1. BFGS (default)")
    print("2. L-BFGS-B")
    print("3. Nelder-Mead")
    print("4. Powell")

    method_choice = input("\nEnter choice (1-4): ")

    if method_choice == "2":
        method = "L-BFGS-B"
    elif method_choice == "3":
        method = "Nelder-Mead"
    elif method_choice == "4":
        method = "Powell"
    else:
        method = "BFGS"

    # Get number of shots for hardware execution
    shots = None
    if HARDWARE_AVAILABLE:
        shots_input = input(
            "\nEnter number of measurement shots for hardware (default: 1000): "
        )
        shots = int(shots_input) if shots_input.strip() else 1000

    return (iterations, method, shots)


def create_and_train():
    """Create and train a new QNN model."""
    try:
        print_header("Create and Train QNN Model")

        # Get molecule and training parameters
        molecule_params = get_molecule_params()
        training_params = get_training_params()

        if molecule_params is None or training_params is None:
            print("\nError: Failed to get required parameters. Returning to main menu.")
            return

        (
            molecule,
            bond_length,
            depth,
            ansatz_type,
            ansatz_kwargs,
            hardware_backend,
            optimizer,
        ) = molecule_params
        iterations, method, verbose = training_params

        print(f"\nCreating QNN for {molecule} with {ansatz_type} ansatz...")

        try:
            qnn = MolecularQNN(
                molecule=molecule,
                bond_length=bond_length,
                depth=depth,
                ansatz_type=ansatz_type,
                ansatz_kwargs=ansatz_kwargs,
                hardware_backend=hardware_backend,
                optimizer=optimizer,
            )
        except Exception as e:
            print(f"\nError creating QNN model: {str(e)}")
            print("Please check your parameters and try again.")
            input("\nPress Enter to continue...")
            return

        print("\nInitial random parameters:")
        print(f"Number of parameters: {len(qnn.params)}")

        print(
            f"\nStarting training with {method} optimizer for {iterations} iterations..."
        )

        # Define callback to show progress
        def callback(params):
            iteration = len(qnn.energy_history)
            if iteration % 5 == 0:
                energy = qnn.get_energy()
                print(f"Iteration {iteration}: Energy = {energy:.6f} Hartree")

        try:
            qnn.train(
                iterations=iterations, method=method, verbose=verbose, callback=callback
            )

            print("\nTraining complete!")
            print(f"Final energy: {qnn.get_energy():.6f} Hartree")

            if qnn.exact_energy is not None:
                print(f"Exact energy: {qnn.exact_energy:.6f} Hartree")
                print(
                    f"Difference: {abs(qnn.get_energy() - qnn.exact_energy):.6f} Hartree"
                )

            # Plot energy convergence
            try:
                qnn.plot_energy_convergence()
                plt.show()
            except Exception as e:
                print(f"\nWarning: Could not display plot: {str(e)}")

            # Ask if user wants to save the model
            save = input("\nDo you want to save this model? (Y/n): ").lower() != "n"
            if save:
                filename = input("Enter filename to save (default: qnn_model.pkl): ")
                filename = filename.strip() or "qnn_model.pkl"

                try:
                    qnn.save_model(filename)
                    print(f"Model saved to {filename}")
                except Exception as e:
                    print(f"\nError saving model: {str(e)}")
                    print(
                        "Make sure you have write permissions for the specified path."
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            print("Partial results may still be available.")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print("You might want to try a different optimizer or parameters.")

        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Returning to main menu.")
        input("\nPress Enter to continue...")


def load_model():
    """Load a saved QNN model."""
    try:
        print_header("Load Saved Model")

        filename = input("Enter filename to load (default: qnn_model.pkl): ")
        filename = filename.strip() or "qnn_model.pkl"

        if not os.path.exists(filename):
            print(f"\nError: File '{filename}' does not exist.")
            input("\nPress Enter to continue...")
            return

        try:
            qnn = MolecularQNN.load_model(filename)
            print("\nModel loaded successfully!")
            print(f"Molecule: {qnn.molecule}")
            print(f"Bond length: {qnn.bond_length} Å")
            print(f"Ansatz type: {qnn.ansatz_type}")
            print(f"Number of qubits: {qnn.n_qubits}")
            print(f"Energy: {qnn.get_energy():.6f} Hartree")

            if qnn.exact_energy is not None:
                print(f"Exact energy: {qnn.exact_energy:.6f} Hartree")
                print(
                    f"Difference: {abs(qnn.get_energy() - qnn.exact_energy):.6f} Hartree"
                )

            # Plot energy convergence
            try:
                qnn.plot_energy_convergence()
                plt.show()
            except Exception as e:
                print(f"\nWarning: Could not display plot: {str(e)}")

        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            print(
                "The file might be corrupted or created with an incompatible version."
            )

        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        input("\nPress Enter to continue...")


def compare_ansatz_types():
    """Compare different ansatz types for the same molecule."""
    try:
        print_header("Compare Ansatz Types")

        # Get molecule parameters
        print("\nSelect molecule parameters:")
        molecule = input("Molecule (H2, LiH, H2O) [default: H2]: ").strip() or "H2"

        if molecule not in ["H2", "LiH", "H2O"]:
            print(f"\nError: Unsupported molecule '{molecule}'. Using H2 instead.")
            molecule = "H2"

        bond_length_input = input("Bond length in Angstroms [default: 0.74]: ").strip()
        try:
            bond_length = float(bond_length_input) if bond_length_input else 0.74
        except ValueError:
            print("\nError: Invalid bond length. Using default value of 0.74 Å.")
            bond_length = 0.74

        depth_input = input("Circuit depth [default: 2]: ").strip()
        try:
            depth = int(depth_input) if depth_input else 2
        except ValueError:
            print("\nError: Invalid depth. Using default value of 2.")
            depth = 2

        iterations_input = input(
            "Number of iterations per ansatz [default: 50]: "
        ).strip()
        try:
            iterations = int(iterations_input) if iterations_input else 50
        except ValueError:
            print("\nError: Invalid number of iterations. Using default value of 50.")
            iterations = 50

        # Select ansatz types to compare
        print("\nSelect ansatz types to compare (separated by commas):")
        print("Options: hardware_efficient, ucc, chea, hva, symmetry_preserving")
        ansatz_input = input("[default: hardware_efficient,ucc,chea,hva]: ").strip()

        if ansatz_input:
            ansatz_types = [a.strip() for a in ansatz_input.split(",")]
            valid_types = [
                "hardware_efficient",
                "ucc",
                "chea",
                "hva",
                "symmetry_preserving",
            ]
            ansatz_types = [a for a in ansatz_types if a in valid_types]

            if not ansatz_types:
                print(
                    "\nError: No valid ansatz types specified. Using default selection."
                )
                ansatz_types = ["hardware_efficient", "ucc", "chea", "hva"]
        else:
            ansatz_types = ["hardware_efficient", "ucc", "chea", "hva"]

        print(
            f"\nComparing {len(ansatz_types)} ansatz types: {', '.join(ansatz_types)}"
        )
        print(f"for {molecule} at bond length {bond_length} Å, depth {depth}")
        print(f"Running {iterations} iterations for each type")

        try:
            # Create a base QNN
            base_qnn = MolecularQNN(
                molecule=molecule,
                bond_length=bond_length,
                depth=depth,
                ansatz_type="hardware_efficient",  # Default type, will be changed
            )

            # Define callback for progress tracking
            def callback(progress_data):
                ansatz = progress_data["ansatz_type"]
                iteration = progress_data["iteration"]
                energy = progress_data["energy"]

                if iteration % 10 == 0:
                    print(
                        f"{ansatz} - Iteration {iteration}: Energy = {energy:.6f} Hartree"
                    )

            # Compare the ansatz types
            results = base_qnn.compare_ansatz_types(
                iterations=iterations, methods=ansatz_types, callback=callback
            )

            print("\nComparison complete!")
            print("\nFinal energies:")
            for ansatz, energy in results["final_energies"].items():
                print(f"{ansatz}: {energy:.6f} Hartree")

            # Display the plot
            try:
                plt.show()  # The compare_ansatz_types method creates a plot
            except Exception as e:
                print(f"\nWarning: Could not display plot: {str(e)}")

        except Exception as e:
            print(f"\nError during comparison: {str(e)}")
            print("Try adjusting parameters or selecting different ansatz types.")

        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        input("\nPress Enter to continue...")


def run_on_hardware():
    """Run a circuit on real quantum hardware or with noise simulation."""
    try:
        if not HARDWARE_AVAILABLE:
            print("\nError: Quantum hardware module is not available.")
            print("Please install the required dependencies and try again.")
            input("\nPress Enter to continue...")
            return

        print_header("Run on Quantum Hardware")

        # Get basic parameters
        print("\nSelect molecule parameters:")
        molecule = input("Molecule (H2, LiH, H2O) [default: H2]: ").strip() or "H2"

        if molecule not in ["H2", "LiH", "H2O"]:
            print(f"\nError: Unsupported molecule '{molecule}'. Using H2 instead.")
            molecule = "H2"

        bond_length_input = input("Bond length in Angstroms [default: 0.74]: ").strip()
        try:
            bond_length = float(bond_length_input) if bond_length_input else 0.74
        except ValueError:
            print("\nError: Invalid bond length. Using default value of 0.74 Å.")
            bond_length = 0.74

        # Select backend
        print("\nSelect quantum backend:")
        hardware_provider = QuantumHardwareProvider()
        available_backends = hardware_provider.list_backends()

        for i, backend in enumerate(available_backends, 1):
            is_simulator = hardware_provider.get_backend_info(backend).get(
                "is_simulator", True
            )
            backend_type = "Simulator" if is_simulator else "Hardware"
            print(f"{i}. {backend} ({backend_type})")

        backend_choice = input("\nEnter backend number [default: 1]: ").strip()
        try:
            backend_idx = int(backend_choice) - 1 if backend_choice else 0
            if backend_idx < 0 or backend_idx >= len(available_backends):
                print("\nError: Invalid selection. Using the first backend.")
                backend_idx = 0
        except ValueError:
            print("\nError: Invalid input. Using the first backend.")
            backend_idx = 0

        backend_name = available_backends[backend_idx]
        backend = hardware_provider.get_backend(backend_name)

        print(f"\nSelected backend: {backend_name}")

        # Select error mitigation if available
        error_mitigation = None
        available_error_mitigation = (
            hardware_provider.list_error_mitigation_strategies()
        )

        if available_error_mitigation:
            print("\nSelect error mitigation strategy:")
            print("0. None (no error mitigation)")

            for i, strategy in enumerate(available_error_mitigation, 1):
                print(f"{i}. {strategy}")

            em_choice = input("\nEnter strategy number [default: 0]: ").strip()
            try:
                em_idx = int(em_choice) if em_choice else 0
                if em_idx < 0 or em_idx > len(available_error_mitigation):
                    print("\nError: Invalid selection. Not using error mitigation.")
                    em_idx = 0

                if em_idx > 0:
                    strategy_name = available_error_mitigation[em_idx - 1]
                    error_mitigation = hardware_provider.get_error_mitigation(
                        strategy_name
                    )
                    print(f"Using {strategy_name} error mitigation")
            except ValueError:
                print("\nError: Invalid input. Not using error mitigation.")

        # Create QNN with hardware backend
        print("\nCreating QNN with hardware backend...")

        try:
            qnn = MolecularQNN(
                molecule=molecule,
                bond_length=bond_length,
                ansatz_type="hardware_efficient",  # Using simple ansatz for hardware
                hardware_backend=backend,
                error_mitigation=error_mitigation,
            )

            # Set the number of shots/measurements
            shots_input = input("Number of shots per circuit [default: 1024]: ").strip()
            try:
                shots = int(shots_input) if shots_input else 1024
                qnn.set_shots(shots)
            except ValueError:
                print("\nError: Invalid number of shots. Using default value of 1024.")
                qnn.set_shots(1024)

            # Run a small number of iterations
            iterations_input = input(
                "Number of optimization iterations [default: 20]: "
            ).strip()
            try:
                iterations = int(iterations_input) if iterations_input else 20
            except ValueError:
                print(
                    "\nError: Invalid number of iterations. Using default value of 20."
                )
                iterations = 20

            print(
                f"\nRunning optimization for {iterations} iterations on {backend_name}..."
            )
            print("This may take some time depending on the backend queue...")

            # Define callback to show progress
            def callback(params):
                iteration = len(qnn.energy_history)
                if iteration % 2 == 0:
                    energy = qnn.get_energy()
                    print(f"Iteration {iteration}: Energy = {energy:.6f} Hartree")

            try:
                # Choose an appropriate optimizer for hardware runs (fewer iterations)
                qnn.train(
                    iterations=iterations,
                    method="COBYLA",  # More robust for noisy results
                    verbose=True,
                    callback=callback,
                    shots=shots,
                )

                print("\nHardware run complete!")
                print(f"Final energy: {qnn.get_energy():.6f} Hartree")

                if qnn.exact_energy is not None:
                    print(f"Exact energy: {qnn.exact_energy:.6f} Hartree")
                    print(
                        f"Difference: {abs(qnn.get_energy() - qnn.exact_energy):.6f} Hartree"
                    )

                # Plot energy convergence
                try:
                    qnn.plot_energy_convergence()
                    plt.show()
                except Exception as e:
                    print(f"\nWarning: Could not display plot: {str(e)}")

            except Exception as e:
                print(f"\nError during hardware run: {str(e)}")
                print(
                    "Hardware runs might fail due to queue issues, hardware errors, or connectivity problems."
                )
                print("Consider using a simulator or trying again later.")

        except Exception as e:
            print(f"\nError setting up hardware: {str(e)}")
            print(
                "Check your connection and credentials for the quantum hardware provider."
            )

        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        input("\nPress Enter to continue...")


def test_optimizers():
    """Test different optimization strategies."""
    if not HARDWARE_AVAILABLE:
        print("\nAdvanced optimizer support is not available.")
        print("Please ensure 'advanced_optimizers.py' is accessible.")
        input("\nPress Enter to return to the main menu...")
        return

    print("\n=== Advanced Optimizer Testing ===")

    # Choose test function
    print("\nSelect test function:")
    print("1. Rosenbrock function")
    print("2. Quadratic function")
    print("3. QNN energy estimation")

    func_choice = input("\nEnter choice (1-3): ")

    if func_choice == "1":
        # Rosenbrock function
        def test_func(params):
            x, y = params
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        initial_params = [0.0, 0.0]
        param_names = ["x", "y"]
        true_min = [1.0, 1.0]
        true_value = 0.0
        test_name = "Rosenbrock function"

    elif func_choice == "2":
        # Simple quadratic function
        def test_func(params):
            return sum((p - 1) ** 2 for p in params)

        dim = int(input("\nEnter dimension (default: 4): ") or 4)
        initial_params = [0.0] * dim
        param_names = [f"x{i + 1}" for i in range(dim)]
        true_min = [1.0] * dim
        true_value = 0.0
        test_name = f"{dim}-dimensional quadratic function"

    else:
        # QNN energy estimation (real quantum objective)
        molecule, bond_length, depth, ansatz_type, ansatz_kwargs, _, _ = (
            get_molecule_params()
        )

        print(f"\nCreating QNN for {molecule} with bond length {bond_length} Å...")
        qnn = MolecularQNN(
            molecule=molecule,
            bond_length=bond_length,
            depth=depth,
            ansatz_type=ansatz_type,
            ansatz_kwargs=ansatz_kwargs,
        )

        initial_params = qnn.random_params()
        param_names = [f"θ{i + 1}" for i in range(len(initial_params))]
        true_min = None
        true_value = None
        test_name = f"QNN energy for {molecule}"

        # Define test function
        def test_func(params):
            return qnn._energy_expectation(params)

    # Select optimizers to test
    print("\nSelect optimizers to test:")
    print("1. BFGS (default)")
    print("2. Noise-aware")
    print("3. Gradient-free (Nelder-Mead)")
    print("4. Gradient-free (Differential Evolution)")
    print("5. Adaptive")
    print("6. Parallel Tempering")
    print("7. All of the above")

    opt_choice = input("\nEnter choice (1-7): ")

    if opt_choice == "7":
        optimizers = [
            create_optimizer("bfgs"),
            create_optimizer("noise_aware"),
            create_optimizer("gradient_free", method="nelder-mead"),
            create_optimizer("gradient_free", method="differential_evolution"),
            create_optimizer("adaptive"),
            create_optimizer("parallel_tempering"),
        ]
        optimizer_names = [
            "BFGS",
            "Noise-aware",
            "Nelder-Mead",
            "Differential Evolution",
            "Adaptive",
            "Parallel Tempering",
        ]
    else:
        optimizers = []
        optimizer_names = []

        if opt_choice == "1" or not opt_choice.strip():
            optimizers.append(create_optimizer("bfgs"))
            optimizer_names.append("BFGS")
        elif opt_choice == "2":
            optimizers.append(create_optimizer("noise_aware"))
            optimizer_names.append("Noise-aware")
        elif opt_choice == "3":
            optimizers.append(create_optimizer("gradient_free", method="nelder-mead"))
            optimizer_names.append("Nelder-Mead")
        elif opt_choice == "4":
            optimizers.append(
                create_optimizer("gradient_free", method="differential_evolution")
            )
            optimizer_names.append("Differential Evolution")
        elif opt_choice == "5":
            optimizers.append(create_optimizer("adaptive"))
            optimizer_names.append("Adaptive")
        elif opt_choice == "6":
            optimizers.append(create_optimizer("parallel_tempering"))
            optimizer_names.append("Parallel Tempering")

    # Get max iterations
    max_iters = int(input("\nEnter maximum iterations (default: 100): ") or 100)

    # Run comparison
    print(f"\nComparing {len(optimizers)} optimizers on {test_name}...")
    results = []

    for name, optimizer in zip(optimizer_names, optimizers):
        print(f"\nRunning {name} optimizer...")
        start_time = time.time()

        try:
            result = optimizer.minimize(
                test_func, initial_params.copy(), max_iterations=max_iters, verbose=True
            )

            duration = time.time() - start_time

            # Display results
            final_value = result["fun"]
            final_params = result["x"]
            iterations = result["nit"]

            print(f"Final value: {final_value:.6f}")
            print(f"Iterations: {iterations}")
            print(f"Time taken: {duration:.2f} seconds")

            # Calculate error if true minimum is known
            if true_value is not None:
                error = abs(final_value - true_value)
                print(f"Error: {error:.6e}")

                param_error = np.sqrt(
                    np.sum((np.array(final_params) - np.array(true_min)) ** 2)
                )
                print(f"Parameter error: {param_error:.6e}")

            # Store result for comparison
            results.append(
                {
                    "name": name,
                    "final_value": final_value,
                    "iterations": iterations,
                    "time": duration,
                    "params": final_params,
                }
            )

        except Exception as e:
            print(f"Error with {name} optimizer: {e}")

    # Compare results
    if len(results) > 1:
        print("\nOptimizer Comparison:")
        print(f"{'Optimizer':<25} {'Value':<15} {'Iterations':<12} {'Time (s)':<10}")
        print("-" * 65)

        for result in results:
            print(
                f"{result['name']:<25} {result['final_value']:<15.6f} {result['iterations']:<12} {result['time']:<10.2f}"
            )

        # Plot convergence if history is available
        try:
            plt.figure(figsize=(12, 6))

            for i, (name, optimizer) in enumerate(zip(optimizer_names, optimizers)):
                if hasattr(optimizer, "history") and optimizer.history:
                    values = optimizer.history["values"]
                    iterations = range(len(values))
                    plt.plot(iterations, values, label=name)

            plt.xlabel("Iteration")
            plt.ylabel("Function Value")
            plt.title(f"Optimization Convergence for {test_name}")
            plt.legend()
            plt.grid(True)

            # Save plot
            save_plot = (
                input("\nSave optimization comparison plot? (y/N): ").lower() == "y"
            )
            if save_plot:
                plot_filename = (
                    f"optimizer_comparison_{test_name.replace(' ', '_')}.png"
                )
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {plot_filename}")

            plt.show()

        except Exception as e:
            print(f"Error creating convergence plot: {e}")

    input("\nPress Enter to return to the main menu...")


def view_system_info():
    """Display system information."""
    print("\n=== System Information ===")

    # Python version
    print(f"Python version: {platform.python_version()}")

    # Cirq version
    try:
        import cirq

        print(f"Cirq version: {cirq.__version__}")
    except (ImportError, AttributeError):
        print("Cirq: Not installed or version not available")

    # NumPy version
    try:
        import numpy

        print(f"NumPy version: {numpy.__version__}")
    except (ImportError, AttributeError):
        print("NumPy: Not installed or version not available")

    # Check hardware support
    print(
        f"Quantum hardware support: {'Available' if HARDWARE_AVAILABLE else 'Not available'}"
    )

    if HARDWARE_AVAILABLE:
        try:
            provider = QuantumHardwareProvider()
            backends = provider.list_backends()

            print("\nAvailable quantum backends:")
            for backend in backends:
                print(f"- {backend}")
        except Exception as e:
            print(f"Error listing backends: {e}")

    input("\nPress Enter to return to the main menu...")


def run_benchmark_suite():
    """Run the Benchmark Suite to compare different optimizers on standard test functions."""
    clear_screen()
    print_header("Optimizer Benchmark Suite")

    # Create benchmark suite
    benchmark = BenchmarkSuite("Quantum Optimizer Benchmark")

    # Display available test functions
    print("\nAvailable Test Functions:")
    print(
        tabulate(
            benchmark.list_test_functions(),
            headers="keys",
            tablefmt="simple",
            showindex=False,
        )
    )

    # Select test functions
    print("\nSelect test functions to benchmark:")
    print("1. Rosenbrock Function (classic non-convex function)")
    print("2. Sphere Function (simple convex function)")
    print("3. Rastrigin Function (highly multimodal function)")
    print("4. Ackley Function (exponential function with many local minima)")
    print("5. Beale Function (2D function with sharp peaks)")
    print("6. Noisy Quadratic (tests noise robustness)")
    print("7. All of the above")

    func_choice = input("\nEnter your choice (1-7): ")

    if func_choice == "7":
        function_ids = [
            "rosenbrock",
            "sphere",
            "rastrigin",
            "ackley",
            "beale",
            "noisy_quadratic",
        ]
    else:
        function_map = {
            "1": ["rosenbrock"],
            "2": ["sphere"],
            "3": ["rastrigin"],
            "4": ["ackley"],
            "5": ["beale"],
            "6": ["noisy_quadratic"],
        }
        function_ids = function_map.get(func_choice, ["rosenbrock"])

    # Select dimensions
    print("\nSelect dimensions to test:")
    print("1. 2D (default for visualization)")
    print("2. 5D (medium complexity)")
    print("3. 10D (higher complexity)")
    print("4. All of the above")

    dim_choice = input("\nEnter your choice (1-4): ")

    if dim_choice == "4":
        dimensions = [2, 5, 10]
    else:
        dimension_map = {"1": [2], "2": [5], "3": [10]}
        dimensions = dimension_map.get(dim_choice, [2])

    # Select optimizers
    print("\nSelect optimizers to benchmark:")
    print("1. BFGS (standard)")
    print("2. Noise-aware")
    print("3. Gradient-free (Nelder-Mead)")
    print("4. Gradient-free (Differential Evolution)")
    print("5. Adaptive")
    print("6. Parallel Tempering")
    print("7. All of the above")

    opt_choice = input("\nEnter your choice (1-7): ")

    optimizers = []
    if opt_choice == "7":
        optimizers = [
            create_optimizer("bfgs"),
            create_optimizer("noise_aware"),
            create_optimizer("gradient_free", method="nelder-mead"),
            create_optimizer("gradient_free", method="differential_evolution"),
            create_optimizer("adaptive"),
            create_optimizer("parallel_tempering"),
        ]
    else:
        optimizer_map = {
            "1": [create_optimizer("bfgs")],
            "2": [create_optimizer("noise_aware")],
            "3": [create_optimizer("gradient_free", method="nelder-mead")],
            "4": [create_optimizer("gradient_free", method="differential_evolution")],
            "5": [create_optimizer("adaptive")],
            "6": [create_optimizer("parallel_tempering")],
        }
        optimizers = optimizer_map.get(opt_choice, [create_optimizer("bfgs")])

    # Configure benchmark parameters
    print("\nConfigure benchmark parameters:")
    runs_per_config = int(input("Number of runs per configuration (default: 3): ") or 3)
    max_iterations = int(input("Maximum iterations per run (default: 100): ") or 100)

    # Confirm before running
    functions_str = ", ".join(
        [benchmark.test_functions[f]["name"] for f in function_ids]
    )
    optimizers_str = ", ".join([opt.name for opt in optimizers])

    print("\nBenchmark Configuration:")
    print(f"Functions: {functions_str}")
    print(f"Dimensions: {dimensions}")
    print(f"Optimizers: {optimizers_str}")
    print(f"Runs per configuration: {runs_per_config}")
    print(f"Max iterations: {max_iterations}")

    confirm = input("\nRun benchmark with these settings? (y/n): ")
    if confirm.lower() != "y":
        print("Benchmark cancelled.")
        input("Press Enter to return to the main menu...")
        return

    # Run benchmark
    print("\nRunning benchmark... This may take a while.")
    results = benchmark.run_benchmark(
        optimizers=optimizers,
        function_ids=function_ids,
        dimensions=dimensions,
        runs_per_config=runs_per_config,
        max_iterations=max_iterations,
        save_results=True,
    )

    # Generate report
    print("\nGenerating benchmark report...")
    report_path = os.path.join(benchmark.output_dir, "benchmark_report.md")
    benchmark.generate_report(report_path)

    # Generate plots
    print("\nGenerating performance plots...")
    benchmark.plot_results(plot_type="all", save_plots=True)

    print(f"\nBenchmark completed! Results are saved in: {benchmark.output_dir}/")
    print(f"- Full report: {report_path}")
    print(f"- Plots: {os.path.join(benchmark.output_dir, 'plots')}/")

    # Show summary of results
    print("\nSummary of Results:")
    for config_id, config_summary in results["summary"].items():
        func_id, dim_str = config_id.split("_")
        func_name = benchmark.test_functions[func_id]["name"]

        print(f"\n{func_name} ({dim_str}):")
        for opt_name, stats in config_summary.items():
            if "status" in stats and stats["status"] == "all_failed":
                print(f"  {opt_name}: All runs failed")
                continue

            print(
                f"  {opt_name}: Mean value = {stats['mean_value']:.6e}, "
                + f"Time = {stats['mean_time']:.2f}s, "
                + f"Success rate = {stats['success_rate'] * 100:.1f}%"
            )

    input("\nPress Enter to continue...")


def main():
    """Main function to start the CLI interface."""
    try:
        # Setup environment
        clear_screen()
        print_header()

        # Main application loop
        while True:
            clear_screen()
            print_header()
            print_menu()

            try:
                choice = input("Enter your choice: ").strip()

                if not choice:
                    continue

                if choice == "1":
                    create_and_train()
                elif choice == "2":
                    load_model()
                elif choice == "3":
                    compare_ansatz_types()
                elif HARDWARE_AVAILABLE and choice == "4":
                    run_on_hardware()
                elif HARDWARE_AVAILABLE and choice == "5":
                    test_optimizers()
                elif HARDWARE_AVAILABLE and choice == "6":
                    view_system_info()
                elif (HARDWARE_AVAILABLE and choice == "7") or (
                    not HARDWARE_AVAILABLE and choice == "5"
                ):
                    print("\nExiting. Thank you for using the QNN framework!")
                    break
                else:
                    print(f"\nInvalid choice: {choice}")
                    input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\nOperation cancelled. Returning to main menu.")
                input("\nPress Enter to continue...")
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")
                input("\nPress Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting.")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please report this issue to the developers.")
        input("\nPress Enter to exit...")
    finally:
        print("\nGoodbye!")


if __name__ == "__main__":
    print("Starting CLI application...")
    main()
