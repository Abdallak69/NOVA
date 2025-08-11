#!/usr/bin/env python3
"""
NOVA Quantum Neural Network CLI

A command-line interface for quantum neural network molecular energy estimation.
"""

import sys

import click
import matplotlib.pyplot as plt
from tabulate import tabulate

try:
    from nova.core.qnn_molecular_energy import MolecularQNN
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    click.echo(f"Warning: Some dependencies not available: {e}", err=True)
    DEPENDENCIES_AVAILABLE = False


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="nova")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose):
    """NOVA Quantum Neural Network for Molecular Energy Estimation."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option("--molecule", "-m", default="H2",
              type=click.Choice(["H2", "LiH", "H2O"], case_sensitive=False),
              help="Molecule to simulate")
@click.option("--bond-length", "-b", default=None, type=float,
              help="Bond length in Angstroms")
@click.option("--depth", "-d", default=2, type=int,
              help="Circuit depth")
@click.option("--ansatz", "-a", default="hardware_efficient",
              type=click.Choice(["hardware_efficient", "ucc", "chea", "hva", "symmetry_preserving"]),
              help="Ansatz type")
@click.option("--iterations", "-i", default=100, type=int,
              help="Maximum training iterations")
@click.option("--method", default="BFGS",
              type=click.Choice(["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell"]),
              help="Optimization method")
@click.option("--backend", default=None,
              help="Quantum hardware backend")
@click.option("--shots", default=1000, type=int,
              help="Number of measurement shots")
@click.option("--save", "-s", default=None,
              help="Save trained model to file")
@click.option("--plot/--no-plot", default=True,
              help="Show convergence plot")
@click.pass_context
def train(ctx, molecule, bond_length, depth, ansatz, iterations, method,
          backend, shots, save, plot):
    """Train a quantum neural network for molecular energy estimation."""

    if not DEPENDENCIES_AVAILABLE:
        click.echo("Error: Required dependencies not available", err=True)
        sys.exit(1)

    # Access verbose flag
    verbose = ctx.obj.get('verbose', False)

    # Set default bond lengths
    if bond_length is None:
        defaults = {"H2": 0.74, "LiH": 1.6, "H2O": 0.96}
        bond_length = defaults.get(molecule.upper(), 0.74)

    if verbose:
        click.echo(f"Training {molecule} with {ansatz} ansatz")
        click.echo(f"Parameters: bond_length={bond_length}, depth={depth}")

    try:
        # Create QNN model
        qnn = MolecularQNN(
            molecule=molecule,
            bond_length=bond_length,
            depth=depth,
            ansatz_type=ansatz,
            hardware_backend=backend
        )

        click.echo(f"Created QNN with {len(qnn.params)} parameters")

        # Training callback
        def callback(params):
            iteration = len(qnn.energy_history)
            if iteration % 10 == 0:
                energy = qnn.get_energy()
                click.echo(f"Iteration {iteration}: Energy = {energy:.6f} Hartree")

        # Train the model
        with click.progressbar(length=iterations, label="Training") as bar:
            def progress_callback(params):
                callback(params)
                bar.update(1)

            qnn.train(
                iterations=iterations,
                method=method,
                verbose=verbose,
                callback=progress_callback,
                shots=shots
            )

        # Show results
        final_energy = qnn.get_energy()
        click.echo("\nTraining completed!")
        click.echo(f"Final energy: {final_energy:.6f} Hartree")

        if qnn.exact_energy is not None:
            error = abs(final_energy - qnn.exact_energy)
            click.echo(f"Exact energy: {qnn.exact_energy:.6f} Hartree")
            click.echo(f"Error: {error:.6f} Hartree")

        # Plot convergence
        if plot:
            try:
                qnn.plot_energy_convergence()
                plt.show()
            except Exception as e:
                click.echo(f"Warning: Could not display plot: {e}", err=True)

        # Save model
        if save:
            qnn.save_model(save)
            click.echo(f"Model saved to {save}")

    except Exception as e:
        click.echo(f"Error during training: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--plot/--no-plot", default=True, help="Show convergence plot")
@click.pass_context
def load(ctx, model_file, plot):
    """Load and inspect a saved model."""

    if not DEPENDENCIES_AVAILABLE:
        click.echo("Error: Required dependencies not available", err=True)
        sys.exit(1)

    try:
        qnn = MolecularQNN.load_model(model_file)

        # Display model info
        click.echo("Model loaded successfully!")
        click.echo(f"Molecule: {qnn.molecule}")
        click.echo(f"Bond length: {qnn.bond_length} Å")
        click.echo(f"Ansatz type: {qnn.ansatz_type}")
        click.echo(f"Number of qubits: {qnn.n_qubits}")
        click.echo(f"Energy: {qnn.get_energy():.6f} Hartree")

        if qnn.exact_energy is not None:
            error = abs(qnn.get_energy() - qnn.exact_energy)
            click.echo(f"Exact energy: {qnn.exact_energy:.6f} Hartree")
            click.echo(f"Error: {error:.6f} Hartree")

        if plot and len(qnn.energy_history) > 0:
            try:
                qnn.plot_energy_convergence()
                plt.show()
            except Exception as e:
                click.echo(f"Warning: Could not display plot: {e}", err=True)

    except Exception as e:
        click.echo(f"Error loading model: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--molecule", "-m", default="H2",
              type=click.Choice(["H2", "LiH", "H2O"], case_sensitive=False),
              help="Molecule to simulate")
@click.option("--bond-length", "-b", default=None, type=float,
              help="Bond length in Angstroms")
@click.option("--depth", "-d", default=2, type=int,
              help="Circuit depth")
@click.option("--ansatze", default="hardware_efficient,ucc,chea",
              help="Comma-separated list of ansatz types to compare")
@click.option("--iterations", "-i", default=50, type=int,
              help="Training iterations per ansatz")
@click.option("--save", "-s", default=None,
              help="Save comparison plot to file")
@click.pass_context
def compare(ctx, molecule, bond_length, depth, ansatze, iterations, save):
    """Compare different ansatz types for the same molecule."""

    if not DEPENDENCIES_AVAILABLE:
        click.echo("Error: Required dependencies not available", err=True)
        sys.exit(1)

    _ = ctx.obj.get('verbose', False)

    # Set default bond length
    if bond_length is None:
        defaults = {"H2": 0.74, "LiH": 1.6, "H2O": 0.96}
        bond_length = defaults.get(molecule.upper(), 0.74)

    ansatz_list = [a.strip() for a in ansatze.split(",")]
    click.echo(f"Comparing ansätze: {', '.join(ansatz_list)}")

    try:
        # Create base QNN for comparison
        qnn = MolecularQNN(
            molecule=molecule,
            bond_length=bond_length,
            depth=depth,
            ansatz_type=ansatz_list[0]  # Use first ansatz as base
        )

        # Run comparison
        results = qnn.compare_ansatz_types(
            iterations=iterations,
            methods=ansatz_list,
            save_path=save
        )

        # Display results table
        headers = ["Ansatz", "Final Energy", "Error", "Training Time"]
        table_data = []

        for ansatz, data in results.items():
            final_energy = data.get('final_energy', 'N/A')
            error = data.get('error', 'N/A')
            training_time = data.get('training_time', 'N/A')

            if isinstance(final_energy, float):
                final_energy = f"{final_energy:.6f}"
            if isinstance(error, float):
                error = f"{error:.6f}"
            if isinstance(training_time, float):
                training_time = f"{training_time:.2f}s"

            table_data.append([ansatz, final_energy, error, training_time])

        click.echo("\nComparison Results:")
        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))

        if save:
            click.echo(f"Comparison plot saved to {save}")

    except Exception as e:
        click.echo(f"Error during comparison: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def backends(ctx):
    """List available quantum backends."""

    try:
        from nova.hardware.quantum_hardware_enhanced import default_provider

        available_backends = default_provider.list_backends()

        click.echo("Available Quantum Backends:")
        for backend in available_backends:
            click.echo(f"  • {backend}")

        if not available_backends:
            click.echo("  No backends available")

    except ImportError:
        click.echo("Hardware modules not available", err=True)


@main.command()
@click.pass_context
def info(ctx):
    """Show system and dependency information."""

    click.echo("NOVA Quantum Neural Network")
    click.echo("=" * 50)

    # Python info
    click.echo(f"Python: {sys.version}")
    click.echo(f"Platform: {sys.platform}")

    # Dependency info
    dependencies = [
        ("numpy", "numpy"),
        ("cirq", "cirq"),
        ("openfermion", "openfermion"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("PyQt5", "PyQt5"),
        ("qiskit", "qiskit"),
    ]

    click.echo("\nDependencies:")
    for name, module in dependencies:
        try:
            __import__(module)
            click.echo(f"  ✓ {name}")
        except ImportError:
            click.echo(f"  ✗ {name}")

    # Hardware info
    try:
        from nova.hardware.quantum_hardware_enhanced import default_provider
        backends = default_provider.list_backends()
        click.echo(f"\nQuantum Backends: {len(backends)} available")
    except ImportError:
        click.echo("\nQuantum Backends: Not available")


@main.group()
def gui():
    """Launch graphical interfaces."""
    pass


@gui.command()
def main_gui():
    """Launch the main GUI interface."""
    try:
        from nova.gui.gui_interface import main as gui_main
        gui_main()
    except ImportError:
        click.echo("GUI dependencies not available. Install PyQt5.", err=True)
        sys.exit(1)


@gui.command()
def compare_gui():
    """Launch the ansatz comparison GUI."""
    try:
        from nova.gui.ansatz_comparison_gui import main as compare_main
        compare_main()
    except ImportError:
        click.echo("GUI dependencies not available. Install PyQt5.", err=True)
        sys.exit(1)


@main.command()
def test():
    """Run the test suite."""
    try:
        import pytest
        exit_code = pytest.main(["-q", "tests/"])
        sys.exit(exit_code)

    except FileNotFoundError:
        click.echo("pytest not found. Install with: pip install pytest", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
