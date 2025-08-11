#!/usr/bin/env python3
"""
Quantum Hardware Backend Definitions

This module defines the abstract base class for quantum backends and provides
concrete implementations for simulators like Cirq's simulator.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import cirq
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# --- Abstract Base Class ---


class QuantumBackend(ABC):
    """
    Abstract base class defining the standard interface for quantum backends
    (simulators or hardware).
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the quantum backend.

        Args:
            name: Name identifier for this backend.
            **kwargs: Additional configuration options.
        """
        self.name = name
        self.is_simulator = True
        self.metadata = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initializing backend: {self.name}")

    @abstractmethod
    def run_circuit(
        self, circuit: cirq.Circuit, repetitions: int = 1000, **kwargs
    ) -> Dict:
        """
        Run a quantum circuit and return measurement results.

        Args:
            circuit: The Cirq circuit to run. Must contain measurement gates.
            repetitions: Number of times to run the circuit (shots).
            **kwargs: Backend-specific execution options.

        Returns:
            A dictionary containing measurement results, typically keyed by
            measurement gate keys, with values being numpy arrays of shape
            (repetitions, num_qubits_measured). Should also include metadata
            like execution time if possible.
            Example: {'result': np.array([[0, 0], [1, 1], ...]), 'metadata': {...}}
        """
        pass

    @abstractmethod
    def get_device_properties(self) -> Dict:
        """
        Get the properties of the quantum backend (device or simulator).

        Should include information like qubit count, connectivity, gate set,
        error rates (if applicable), etc.

        Returns:
            Dictionary containing backend properties.
        """
        pass

    def supports_statevector(self) -> bool:
        """
        Check if the backend supports returning a final statevector.
        Defaults to False. Subclasses should override if they support it.
        """
        return False

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Simulate a circuit and return the final statevector.

        Args:
            circuit: The Cirq circuit to simulate. Should not contain measurements.

        Returns:
            A numpy array representing the final statevector.

        Raises:
            NotImplementedError: If the backend does not support statevector simulation.
            TypeError: If the circuit contains measurements.
        """
        if not self.supports_statevector():
            raise NotImplementedError(
                f"Backend {self.name} does not support statevector simulation."
            )
        if any(cirq.is_measurement(op) for op in circuit.all_operations()):
            raise TypeError(
                "Circuit must not contain measurements for statevector simulation."
            )
        # Subclasses supporting statevectors should implement the actual calculation here.
        raise NotImplementedError(
            "Statevector calculation not implemented in base class."
        )

    def __str__(self) -> str:
        """String representation of the backend."""
        return f"{self.name} ({'Simulator' if self.is_simulator else 'Hardware'})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(name='{self.name}', is_simulator={self.is_simulator})"


# --- Concrete Backend Implementations ---


class CirqSimulatorBackend(QuantumBackend):
    """
    Quantum backend implementation using Cirq's built-in simulator.
    Can simulate ideal quantum computation or include noise.
    """

    def __init__(
        self,
        name: str = "cirq_simulator",
        noise_model: Optional[cirq.NoiseModel] = None,
        t1_micros: Optional[float] = None,
        t2_micros: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the Cirq simulator backend.

        Args:
            name: Name identifier for this backend.
            noise_model: Optional pre-constructed Cirq noise model. If provided,
                         it overrides t1/t2 parameters.
            t1_micros: Optional T1 relaxation time in microseconds. If provided (and
                       noise_model is None), a thermal relaxation model is constructed.
            t2_micros: Optional T2 dephasing time in microseconds. Must be provided
                       if t1_micros is, and t2 <= 2*t1. If provided (and
                       noise_model is None), a thermal relaxation model is constructed.
            **kwargs: Additional configuration options for cirq.Simulator or
                      cirq.DensityMatrixSimulator (e.g., seed, dtype).
        """
        super().__init__(name=name, **kwargs)
        self.is_simulator = True
        self.max_qubits = 30  # Practical limit for simulation

        self._t1_micros = t1_micros
        self._t2_micros = t2_micros
        self._explicit_noise_model = noise_model
        self._effective_noise_model = None

        # Determine the effective noise model
        if self._explicit_noise_model is not None:
            self._effective_noise_model = self._explicit_noise_model
            self.logger.info(
                f"Using provided noise model: {type(self._effective_noise_model).__name__}"
            )
        elif self._t1_micros is not None and self._t2_micros is not None:
            if self._t2_micros > 2 * self._t1_micros:
                raise ValueError(
                    f"T2 time ({self._t2_micros} µs) cannot be greater than 2 * T1 time ({self._t1_micros} µs)."
                )
            # Build thermal relaxation model
            self._effective_noise_model = cirq.ThermalRelaxationNoiseModel(
                t1_ns=self._t1_micros * 1000, t2_ns=self._t2_micros * 1000
            )
            self.logger.info(
                f"Constructed ThermalRelaxationNoiseModel with T1={self._t1_micros} µs, T2={self._t2_micros} µs"
            )
        elif self._t1_micros is not None or self._t2_micros is not None:
            raise ValueError(
                "Both t1_micros and t2_micros must be provided to construct a thermal relaxation model."
            )

        # Choose the appropriate simulator
        # Pass relevant kwargs to the simulator constructor
        simulator_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["seed", "dtype", "split_untangled_states"]
        }
        if self._effective_noise_model is not None:
            self.simulator = cirq.DensityMatrixSimulator(**simulator_kwargs)
            self.logger.info("Using DensityMatrixSimulator for noisy simulation.")
        else:
            self.simulator = cirq.Simulator(**simulator_kwargs)
            self.logger.info("Using ideal StateVector Simulator.")

    def run_circuit(
        self, circuit: cirq.Circuit, repetitions: int = 1000, **kwargs
    ) -> Dict:
        """
        Execute a quantum circuit on the Cirq simulator, potentially with noise.

        Args:
            circuit: Cirq circuit to execute. Must contain measurements.
            repetitions: Number of repetitions/measurements.
            **kwargs: Additional execution options (currently ignored by simulators).

        Returns:
            Results dictionary containing measurement outcomes and metadata.
        """
        start_time = time.time()

        if not isinstance(circuit, cirq.Circuit):
            raise TypeError(f"Expected a cirq.Circuit, got {type(circuit)}")

        has_measurements = any(
            cirq.is_measurement(op) for op in circuit.all_operations()
        )
        if not has_measurements:
            raise ValueError("Circuit must contain measurement operations to run.")

        try:
            self.logger.info(
                f"Executing circuit on {self.name} with {repetitions} repetitions."
            )

            # Run with the simulator (which is already DensityMatrix if noise is active)
            if self._effective_noise_model is not None:
                # DensityMatrixSimulator uses the 'noise' argument
                result = self.simulator.run(
                    circuit, repetitions=repetitions, noise=self._effective_noise_model
                )
            else:
                # Standard Simulator doesn't use 'noise' argument
                result = self.simulator.run(circuit, repetitions=repetitions)

            execution_time = time.time() - start_time
            self.logger.info(f"Simulation completed in {execution_time:.4f} seconds.")

            # Prepare results dictionary
            noise_info = "None"
            if self._explicit_noise_model:
                noise_info = f"Explicit: {type(self._explicit_noise_model).__name__}"
            elif self._effective_noise_model:
                noise_info = f"Thermal Relaxation (T1={self._t1_micros}µs, T2={self._t2_micros}µs)"

            results_dict = {
                "metadata": {
                    "execution_time": execution_time,
                    "backend_name": self.name,
                    "repetitions": repetitions,
                    "timestamp": time.time(),
                    "noise_model_used": noise_info,
                }
            }
            # Add measurement results, handling potential multiple keys
            for key, measurement_data in result.measurements.items():
                results_dict[key] = measurement_data

            return results_dict

        except MemoryError as e:
            self.logger.error(f"MemoryError during simulation on {self.name}: {e}")
            raise MemoryError(
                f"Out of memory during simulation on {self.name}. Try reducing circuit size or repetitions."
            ) from e
        except Exception as e:
            self.logger.error(
                f"Error during simulation on {self.name}: {e}", exc_info=True
            )
            raise RuntimeError(f"Simulation failed on {self.name}: {e}") from e

    def supports_statevector(self) -> bool:
        """Cirq simulator supports statevector simulation only if no noise is active."""
        return self._effective_noise_model is None

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Simulate a circuit and return the final statevector. Only works for ideal simulation.

        Args:
            circuit: The Cirq circuit to simulate. Must not contain measurements.

        Returns:
            A numpy array representing the final statevector.

        Raises:
            NotImplementedError: If noise is active.
            TypeError: If the circuit contains measurements.
            RuntimeError: If simulation fails.
        """
        if not self.supports_statevector():
            raise NotImplementedError(
                f"Backend {self.name} with active noise model does not support statevector simulation."
            )
        if any(cirq.is_measurement(op) for op in circuit.all_operations()):
            raise TypeError(
                "Circuit must not contain measurements for statevector simulation."
            )

        start_time = time.time()
        try:
            self.logger.info(f"Calculating statevector on {self.name}.")
            # Ensure we are using the StateVector simulator (self.simulator should be correct type)
            if not isinstance(self.simulator, cirq.Simulator):
                # This shouldn't happen based on __init__ logic, but as a safeguard:
                raise RuntimeError(
                    "Incorrect simulator type for statevector calculation."
                )

            result = self.simulator.simulate(circuit)
            execution_time = time.time() - start_time
            self.logger.info(
                f"Statevector calculation completed in {execution_time:.4f} seconds."
            )
            return result.final_state_vector

        except MemoryError as e:
            self.logger.error(
                f"MemoryError during statevector simulation on {self.name}: {e}"
            )
            raise MemoryError(
                f"Out of memory during statevector simulation on {self.name}. Try reducing circuit size."
            ) from e
        except Exception as e:
            self.logger.error(
                f"Error during statevector simulation on {self.name}: {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Statevector simulation failed on {self.name}: {e}"
            ) from e

    def get_device_properties(self) -> Dict:
        """
        Get the properties of the Cirq simulator, including noise info if applicable.
        """
        noise_info = "None"
        noise_params = {}
        if self._explicit_noise_model:
            noise_info = f"Explicit: {type(self._explicit_noise_model).__name__}"
        elif self._effective_noise_model:
            noise_info = "Thermal Relaxation"
            noise_params = {"t1_micros": self._t1_micros, "t2_micros": self._t2_micros}

        props = {
            "name": self.name,
            "backend_description": "Cirq built-in simulator",
            "is_simulator": self.is_simulator,
            "max_qubits": self.max_qubits,
            "available_gates": self._get_simulator_gates(),
            "supports_statevector": self.supports_statevector(),
            "supports_density_matrix": True,
            "noise_model": noise_info,
            "noise_parameters": noise_params,
            "connectivity": "all-to-all",
        }
        return props

    def _get_simulator_gates(self) -> List[str]:
        """Return a list of common gates supported by Cirq simulators."""
        # This list can be expanded, but covers common gates.
        return [
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "T",
            "CNOT",
            "CZ",
            "SWAP",
            "ISWAP",
            "XPowGate",
            "YPowGate",
            "ZPowGate",
            "HPowGate",
            "CXPowGate",
            "CZPowGate",
            "Rx",
            "Ry",
            "Rz",
            "PhasedXPowGate",
            "PhasedXZGate",
            "MeasurementGate",
            "GlobalPhaseGate",
            "WaitGate",
            "Duration",  # Added Duration for thermal noise model
        ]


# Placeholder for Qiskit Backend if Qiskit is available
# We'll add this later if needed based on Qiskit integration status
# Example:
QISKIT_AVAILABLE = False  # Initialize to False
try:
    # Try importing a specific Qiskit backend class if needed for the check
    from qiskit_aer import AerSimulator  # noqa: F401
    QISKIT_AVAILABLE = True
except ImportError:
    # QISKIT_AVAILABLE remains False
    pass
#
# if QISKIT_AVAILABLE:
#     class QiskitSimulatorBackend(QuantumBackend):
#         # ... implementation ...
#         pass


# --- Old Interface Code Removed ---
# class QuantumHardwareInterface(ABC): ...
# class CirqHardwareInterface(QuantumHardwareInterface): ...
# class QiskitHardwareInterface(QuantumHardwareInterface): ... # (If it existed fully)
# def create_hardware_interface(...): ...
# class HardwareInterfaceManager: ...
# hardware_manager = HardwareInterfaceManager() ...

# Legacy compatibility for hardware_manager
try:
    from nova.hardware.quantum_hardware_enhanced import default_provider

    hardware_manager = default_provider  # Legacy alias for backward compatibility
except ImportError:
    # Create a dummy hardware_manager if enhanced module is not available
    class DummyHardwareManager:
        def get_interface(self, name=None):
            return CirqSimulatorBackend()

    hardware_manager = DummyHardwareManager()
