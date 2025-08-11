#!/usr/bin/env python3
"""
Enhanced Quantum Hardware Integration Module

This module integrates the enhanced error handling, logging, and error mitigation
components to provide a robust and improved quantum hardware interface.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cirq
import numpy as np

# Import the enhanced components
try:
    from nova.core.quantum_logger import configure_logging, quantum_logger
except ImportError:
    import logging

    logging.warning("Quantum logger module not found. Using basic logging instead.")
    quantum_logger = logging.getLogger("quantum_hardware")

    def configure_logging(*args, **kwargs):
        pass


try:
    from nova.hardware.quantum_error_handling import (
        QuantumHardwareError,
        check_backend_health,
        get_detailed_backend_diagnostics,
        retry_quantum_operation,
        validate_quantum_circuit,
        with_error_handling,
    )
except ImportError:
    # Define basic fallbacks if module not found
    class QuantumHardwareError(Exception):
        pass

    def retry_quantum_operation(max_retries=3):
        def decorator(func):
            return func

        return decorator

    def with_error_handling(func):
        return func

    def validate_quantum_circuit(circuit, backend):
        return True, None

    def check_backend_health(backend):
        return {"available": True}

    def get_detailed_backend_diagnostics(backend):
        return {}


try:
    from nova.core.quantum_logger import QuantumLogger, quantum_logger
    from nova.hardware.quantum_hardware_interface import QuantumBackend
    from nova.mitigation.quantum_error_mitigation import ErrorMitigationFactory
except ImportError:
    ErrorMitigationFactory = None
    QuantumBackend = None
    QuantumLogger = None
    quantum_logger = None

# Import the base QuantumBackend class
try:
    from nova.hardware.quantum_hardware_interface import (
        CirqSimulatorBackend,
        QuantumBackend,
    )
except ImportError as e:
    quantum_logger.error(f"Failed to import QuantumBackend base class: {e}")

    # Define a basic stub if import fails
    class QuantumBackend:
        def __init__(self, name: str, **kwargs):
            self.name = name

        def run_circuit(self, circuit, repetitions=1000, **kwargs):
            raise NotImplementedError

        def get_device_properties(self):
            return {"name": self.name, "error": "Base class import failed"}

    class CirqSimulatorBackend(QuantumBackend):
        pass


class EnhancedQuantumBackend:
    """
    Enhanced wrapper for quantum backends with improved error handling,
    logging, and validation.
    """

    def __init__(self, backend: QuantumBackend):
        """
        Initialize the enhanced quantum backend.

        Args:
            backend: Original quantum backend to enhance
        """
        if not isinstance(backend, QuantumBackend):
            raise TypeError(
                f"Expected an instance of QuantumBackend, got {type(backend)}"
            )
        self.backend = backend
        self.name = getattr(backend, "name", "Unknown Backend")
        self.health_check_interval = 3600  # seconds
        self.last_health_check = 0
        self.health_status = {}

        # Run initial health check
        self._check_health()

    def _check_health(self) -> Dict:
        """
        Check the health of the backend.

        Returns:
            Dictionary with health status
        """
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                self.health_status = check_backend_health(self.backend)
                self.last_health_check = current_time
            except Exception as e:
                quantum_logger.log_error(
                    component="hardware",
                    error_type="health_check_error",
                    message=f"Health check failed: {str(e)}",
                )
                self.health_status = {
                    "available": False,
                    "status": "error",
                    "errors": [str(e)],
                    "timestamp": current_time,
                }

        # If health status doesn't have 'available' key, set it to True by default
        if "available" not in self.health_status:
            self.health_status["available"] = True

        return self.health_status

    @retry_quantum_operation(max_retries=3)
    @with_error_handling
    def run_circuit(
        self,
        circuit: cirq.Circuit,
        repetitions: int = 1000,
        error_mitigation: Any = None,
        **kwargs,
    ) -> Dict:
        """
        Run a quantum circuit with enhanced error handling and logging.

        Args:
            circuit: Cirq circuit to run
            repetitions: Number of repetitions (shots)
            error_mitigation: Optional error mitigation strategy
            **kwargs: Additional arguments

        Returns:
            Dictionary with measurement results
        """
        operation_id = quantum_logger.start_operation(f"run_circuit_{self.name}")
        execution_metadata = {
            "backend": self.name,
            "circuit_qubits": len(list(circuit.all_qubits())),
            "circuit_depth": len(circuit),
            "repetitions": repetitions,
            "error_mitigation": str(error_mitigation) if error_mitigation else None,
        }

        start_time = time.time()

        try:
            # Check backend health
            health = self._check_health()
            if not health.get("available", True):  # Default to available if check fails
                raise QuantumHardwareError(
                    f"Backend {self.name} is not available: {health.get('status', 'unknown')}"
                )

            # Validate circuit
            is_valid, error_message = validate_quantum_circuit(circuit, self.backend)
            if not is_valid:
                raise QuantumHardwareError(
                    f"Circuit validation failed: {error_message}"
                )

            # Apply error mitigation if provided
            mitigated_circuit = circuit
            active_strategy = None
            if error_mitigation:
                if isinstance(error_mitigation, str):
                    if ErrorMitigationFactory:
                        active_strategy = ErrorMitigationFactory.create(
                            error_mitigation, **kwargs
                        )
                        mitigated_circuit = active_strategy.mitigate(
                            circuit, self.backend, **kwargs
                        )
                    else:
                        quantum_logger.warning(
                            "ErrorMitigationFactory not available, cannot create strategy by name."
                        )
                elif hasattr(error_mitigation, "mitigate"):
                    active_strategy = error_mitigation
                    mitigated_circuit = active_strategy.mitigate(
                        circuit, self.backend, **kwargs
                    )
                else:
                    quantum_logger.warning(
                        f"Invalid error mitigation object: {error_mitigation}"
                    )

            # Handle multi-circuit case from some error mitigation strategies
            if isinstance(mitigated_circuit, list):
                subcircuit_results = []
                for idx, subcircuit in enumerate(mitigated_circuit):
                    quantum_logger.logger.debug(
                        f"Running sub-circuit {idx + 1}/{len(mitigated_circuit)} for mitigation strategy {str(active_strategy)}"
                    )

                    # Run the subcircuit
                    sub_result = self.backend.run_circuit(
                        subcircuit, repetitions=repetitions
                    )
                    subcircuit_results.append(sub_result)

                # Process results using the error mitigation strategy
                if active_strategy and hasattr(active_strategy, "process_results"):
                    final_results = active_strategy.process_results(
                        subcircuit_results, **kwargs
                    )
                else:
                    # Just return the first result as fallback
                    quantum_logger.warning(
                        "Multi-circuit result processing not found, returning first result."
                    )
                    final_results = subcircuit_results[0]
            else:
                # Log the execution attempt
                quantum_logger.logger.info(
                    f"Executing circuit on backend: {self.name}, "
                    f"Qubits: {len(list(mitigated_circuit.all_qubits()))}, "
                    f"Depth: {len(mitigated_circuit)}, Reps: {repetitions}"
                )

                # Run the circuit
                results = self.backend.run_circuit(
                    mitigated_circuit, repetitions=repetitions
                )

                # Apply result processing if using error mitigation
                if active_strategy and hasattr(active_strategy, "process_results"):
                    final_results = active_strategy.process_results(results, **kwargs)
                else:
                    final_results = results

            # Calculate execution time
            execution_time = time.time() - start_time

            # Log successful execution
            quantum_logger.log_circuit_execution(
                backend_name=self.name,
                n_qubits=len(list(circuit.all_qubits())),
                depth=len(circuit),
                shots=repetitions,
                success=True,
                execution_time=execution_time,
                error_data=None,
            )

            # Add execution metadata to results
            if isinstance(final_results, dict):
                if "metadata" not in final_results:
                    final_results["metadata"] = {}
                final_results["metadata"].update(
                    {
                        "execution_time": execution_time,
                        "backend": self.name,
                        "shots": repetitions,
                        "error_mitigation": str(active_strategy)
                        if active_strategy
                        else None,
                        "timestamp": time.time(),
                    }
                )

            # End operation logging
            quantum_logger.end_operation(
                operation_id,
                {
                    "success": True,
                    "execution_time": execution_time,
                    **execution_metadata,
                },
            )

            return final_results

        except Exception as e:
            # Log failure
            execution_time = time.time() - start_time
            error_data = {"message": str(e), "type": type(e).__name__}

            quantum_logger.log_circuit_execution(
                backend_name=self.name,
                n_qubits=len(list(circuit.all_qubits())),
                depth=len(circuit),
                shots=repetitions,
                success=False,
                execution_time=execution_time,
                error_data=error_data,
            )

            quantum_logger.end_operation(
                operation_id,
                {
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time,
                    **execution_metadata,
                },
            )

            # Re-raise the exception as QuantumHardwareError if not already
            if isinstance(e, QuantumHardwareError):
                raise
            else:
                raise QuantumHardwareError(
                    f"Error during circuit execution: {str(e)}"
                ) from e

    @with_error_handling
    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Get the statevector output of a circuit with enhanced error handling.

        Args:
            circuit: Cirq circuit to run

        Returns:
            Final statevector
        """
        operation_id = quantum_logger.start_operation(f"get_statevector_{self.name}")
        execution_metadata = {
            "backend": self.name,
            "circuit_qubits": len(list(circuit.all_qubits())),
            "circuit_depth": len(circuit),
        }

        start_time = time.time()

        try:
            # Check backend health
            health = self._check_health()
            if not health.get("available", True):  # Default to available
                raise QuantumHardwareError(
                    f"Backend {self.name} is not available: {health.get('status', 'unknown')}"
                )

            # Validate circuit
            is_valid, error_message = validate_quantum_circuit(circuit, self.backend)
            if not is_valid:
                raise QuantumHardwareError(
                    f"Circuit validation failed: {error_message}"
                )

            # Check if backend supports statevector
            if (
                not hasattr(self.backend, "supports_statevector")
                or not self.backend.supports_statevector()
            ):
                raise QuantumHardwareError(
                    f"Backend {self.name} does not support statevector simulation"
                )

            # Get statevector
            statevector = self.backend.get_statevector(circuit)

            # Calculate execution time
            execution_time = time.time() - start_time

            # End operation logging
            quantum_logger.end_operation(
                operation_id,
                {
                    "success": True,
                    "execution_time": execution_time,
                    **execution_metadata,
                },
            )

            return statevector

        except Exception as e:
            # Log failure
            execution_time = time.time() - start_time

            quantum_logger.end_operation(
                operation_id,
                {
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time,
                    **execution_metadata,
                },
            )

            # Re-raise the exception
            if isinstance(e, QuantumHardwareError):
                raise
            else:
                raise QuantumHardwareError(
                    f"Error getting statevector: {str(e)}"
                ) from e

    def supports_statevector(self) -> bool:
        """Check if backend supports statevector output."""
        if hasattr(self.backend, "supports_statevector"):
            return self.backend.supports_statevector()
        return False

    def get_device_properties(self) -> Dict:
        """Get properties of the device with enhanced diagnostics."""
        operation_id = quantum_logger.start_operation(
            f"get_device_properties_{self.name}"
        )

        try:
            # Get base properties
            if hasattr(self.backend, "get_device_properties"):
                properties = self.backend.get_device_properties()
            else:
                properties = {
                    "name": self.name,
                    "is_simulator": getattr(self.backend, "is_simulator", True),
                    "max_qubits": getattr(self.backend, "max_qubits", 0),
                    "available_gates": getattr(self.backend, "available_gates", []),
                }

            # Add detailed diagnostics
            properties["diagnostics"] = get_detailed_backend_diagnostics(self.backend)
            properties["health"] = self._check_health()

            quantum_logger.end_operation(operation_id, {"success": True})

            return properties
        except Exception as e:
            quantum_logger.end_operation(
                operation_id, {"success": False, "error": str(e)}
            )

            # Return basic info on error
            return {"name": self.name, "error": str(e)}

    def __str__(self) -> str:
        """String representation of the enhanced backend."""
        health = (
            "healthy" if self.health_status.get("available", False) else "unhealthy"
        )
        return f"Enhanced {self.name} ({health})"


class EnhancedExecutionResult:
    """Enhanced execution result with additional metadata and analysis capabilities."""

    def __init__(
        self,
        raw_results: Dict,
        circuit: cirq.Circuit = None,
        backend_name: str = None,
        execution_time: float = None,
    ):
        """
        Initialize enhanced execution result.

        Args:
            raw_results: Raw results from backend execution
            circuit: Original circuit
            backend_name: Name of the backend used
            execution_time: Execution time in seconds
        """
        self.raw_results = raw_results
        self.circuit = circuit
        self.backend_name = backend_name
        self.execution_time = execution_time
        self.timestamp = time.time()

        # Extract metadata if available
        if isinstance(raw_results, dict) and "metadata" in raw_results:
            self.metadata = raw_results["metadata"]
            if "execution_time" in self.metadata and self.execution_time is None:
                self.execution_time = self.metadata["execution_time"]
            if "backend" in self.metadata and self.backend_name is None:
                self.backend_name = self.metadata["backend"]
        else:
            self.metadata = {}

    def get_counts(self, key: str = None) -> Dict[str, int]:
        """
        Get counts of measurement outcomes.

        Args:
            key: Measurement key (if None, use first available key)

        Returns:
            Dictionary mapping bitstrings to counts
        """
        if not isinstance(self.raw_results, dict):
            return {}

        # Determine which measurement key to use
        if key is None:
            # Use first key in results that is not 'metadata'
            available_keys = [
                k
                for k in self.raw_results
                if k != "metadata" and isinstance(self.raw_results.get(k), np.ndarray)
            ]
            if not available_keys:
                # Check for ZNE structure
                if "result" in self.raw_results and isinstance(
                    self.raw_results["result"], dict
                ):
                    zne_counts = self.raw_results["result"].get("counts")
                    if isinstance(zne_counts, dict):
                        return zne_counts
                return {}
            key = available_keys[0]

        if key not in self.raw_results or not isinstance(
            self.raw_results[key], np.ndarray
        ):
            return {}

        # Count occurrences of each bitstring
        measurements = self.raw_results[key]
        counts = {}
        for result in measurements:
            bitstring = "".join(map(str, result))
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts

    def get_expectation(self, observable: Any = None) -> float:
        """
        Calculate expectation value of an observable.

        Args:
            observable: Observable to calculate expectation value for
                       (if None, use Z on all qubits)

        Returns:
            Expectation value
        """
        # Special case for ZNE results
        if (
            isinstance(self.raw_results, dict)
            and "result" in self.raw_results
            and isinstance(self.raw_results["result"], dict)
            and "expectation" in self.raw_results["result"]
        ):
            return float(self.raw_results["result"]["expectation"])

        # If observable is not specified, use Z on all qubits
        if observable is None:
            counts = self.get_counts()
            if not counts:
                return 0.0

            # Calculate expectation from counts for Z-basis
            total_shots = sum(counts.values())
            expectation = 0.0
            for bitstring, count in counts.items():
                parity = 1 if (bitstring.count("1") % 2 == 0) else -1
                expectation += parity * count
            return expectation / total_shots if total_shots > 0 else 0.0

        # Handle PauliSum observable
        if isinstance(observable, cirq.PauliSum):
            total_expectation = 0.0
            counts = self.get_counts()
            if not counts:
                return 0.0
            total_shots = sum(counts.values())

            for term in observable:
                term_expectation = 0.0
                qubit_indices = {q: i for i, q in enumerate(self.circuit.all_qubits())}

                for bitstring, count in counts.items():
                    parity = 1
                    for qubit, pauli_op in term.items():
                        bit_index = qubit_indices.get(qubit)
                        if bit_index is None:
                            continue

                        bit_value = int(bitstring[bit_index])
                        if pauli_op == cirq.Z:
                            if bit_value == 1:
                                parity *= -1
                        elif pauli_op in (cirq.X, cirq.Y):
                            parity = 0
                            break
                    term_expectation += parity * count
                total_expectation += term.coefficient * (term_expectation / total_shots)
            return total_expectation.real

        # Handle other observable types if needed
        quantum_logger.warning(
            f"Unsupported observable type for expectation calculation: {type(observable)}"
        )
        return 0.0

    def plot_histogram(
        self, key: str = None, figsize: Tuple[int, int] = (10, 6), top_n: int = 10
    ) -> Any:
        """
        Plot histogram of measurement outcomes.

        Args:
            key: Measurement key
            figsize: Figure size
            top_n: Number of top results to show

        Returns:
            Matplotlib figure or None if plotting not available
        """
        try:
            import matplotlib.pyplot as plt

            counts = self.get_counts(key)
            if not counts:
                return None

            # Sort by counts and take top N
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            if top_n > 0 and len(sorted_counts) > top_n:
                other_count = sum(c for _, c in sorted_counts[top_n:])
                sorted_counts = sorted_counts[:top_n]
                if other_count > 0:
                    sorted_counts.append(("Others", other_count))

            # Create the plot
            fig, ax = plt.subplots(figsize=figsize)
            labels = [x[0] for x in sorted_counts]
            values = [x[1] for x in sorted_counts]

            ax.bar(labels, values)
            ax.set_xlabel("Bitstring")
            ax.set_ylabel("Counts")
            title = f"Measurement Results on {self.backend_name}"
            if key:
                title += f" (Key: {key})"
            ax.set_title(title)

            plt.xticks(rotation=70)
            plt.tight_layout()

            return fig
        except ImportError:
            quantum_logger.log_error(
                component="visualization",
                error_type="import_error",
                message="matplotlib is required for plotting histograms",
            )
            return None

    def __str__(self) -> str:
        """String representation of the result."""
        counts_summary = ""
        counts = self.get_counts()
        if counts:
            # Get top 3 results
            top_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            counts_summary = ", ".join(f"'{k}': {v}" for k, v in top_counts)
            if len(counts) > 3:
                counts_summary += f", ... ({len(counts) - 3} more)"

        return (
            f"EnhancedExecutionResult on {self.backend_name}, "
            f"execution time: {self.execution_time:.4f}s, "
            f"top counts: {{{counts_summary}}}"
        )


class EnhancedQuantumHardwareProvider:
    """Enhanced provider for quantum hardware backends with improved capabilities."""

    def __init__(self):
        """Initialize the enhanced quantum hardware provider."""
        # Registry for backend constructors
        self._backend_constructors: Dict[str, Callable[..., QuantumBackend]] = {}
        # Registry for enhanced backends (cache)
        self._enhanced_backends: Dict[Tuple[str, tuple], EnhancedQuantumBackend] = {}
        # Registry for error mitigation strategy constructors
        self._error_mitigation_constructors: Dict[str, Callable[..., Any]] = {}

        # Auto-register default backends (e.g., CirqSimulator)
        self.register_backend("cirq_simulator", CirqSimulatorBackend)

        # Configure logging
        log_dir = os.environ.get("QUANTUM_LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"quantum_hardware_{time.strftime('%Y%m%d_%H%M%S')}.log"
            )
            configure_logging(log_file=log_file)
        except Exception as e:
            quantum_logger.log_error(
                component="provider",
                error_type="logging_setup_error",
                message=f"Failed to configure logging: {str(e)}",
            )

    def register_backend(
        self, name: str, constructor: Callable[..., QuantumBackend]
    ) -> None:
        """
        Register a quantum backend constructor.

        Args:
            name: Name of the backend
            constructor: Function that creates the backend instance
        """
        if not callable(constructor):
            raise TypeError("Constructor must be callable")
        self._backend_constructors[name] = constructor
        quantum_logger.logger.info(f"Registered backend: {name}")

    def register_error_mitigation(
        self, name: str, constructor: Callable[..., Any]
    ) -> None:
        """
        Register an error mitigation strategy constructor.

        Args:
            name: Name of the strategy
            constructor: Function that creates the strategy instance
        """
        if ErrorMitigationFactory:
            # Use the factory if available (delegates registration)
            pass  # Factory handles registration implicitly via class definition
        else:
            # Manual registration if factory not available
            if not callable(constructor):
                raise TypeError("Constructor must be callable")
            self._error_mitigation_constructors[name] = constructor
            quantum_logger.logger.info(f"Registered error mitigation strategy: {name}")

    @with_error_handling
    def get_backend(
        self, name: str, enhanced: bool = True, **kwargs
    ) -> Union[QuantumBackend, EnhancedQuantumBackend]:
        """
        Get a quantum backend instance by name.

        Args:
            name: Name of the backend
            enhanced: Whether to return an enhanced (wrapped) backend
            **kwargs: Additional arguments for the backend constructor

        Returns:
            Quantum backend instance
        """
        operation_id = quantum_logger.start_operation(f"get_backend_{name}")

        try:
            if name not in self._backend_constructors:
                raise QuantumHardwareError(f"Backend '{name}' not registered.")

            # Create a hashable key for caching kwargs
            kwargs_key = tuple(sorted(kwargs.items()))
            cache_key = (name, kwargs_key)

            if enhanced:
                if cache_key in self._enhanced_backends:
                    backend = self._enhanced_backends[cache_key]
                else:
                    # Create base backend and wrap it
                    constructor = self._backend_constructors[name]
                    base_backend = constructor(**kwargs)
                    backend = EnhancedQuantumBackend(base_backend)
                    self._enhanced_backends[cache_key] = backend  # Cache it
            else:
                # Return the base backend directly
                constructor = self._backend_constructors[name]
                backend = constructor(**kwargs)

            quantum_logger.end_operation(
                operation_id,
                {"success": True, "backend_name": name, "enhanced": enhanced},
            )

            return backend
        except Exception as e:
            quantum_logger.end_operation(
                operation_id, {"success": False, "error": str(e), "backend_name": name}
            )

            # Re-raise the exception
            if isinstance(e, QuantumHardwareError):
                raise
            else:
                raise QuantumHardwareError(
                    f"Error getting backend '{name}': {str(e)}"
                ) from e

    def get_error_mitigation(self, name: str, **kwargs) -> Any:
        """
        Get an error mitigation strategy instance by name.

        Args:
            name: Name of the strategy
            **kwargs: Additional arguments for the strategy constructor

        Returns:
            Error mitigation strategy instance
        """
        if ErrorMitigationFactory:
            # Use factory if available
            return ErrorMitigationFactory.create(name, **kwargs)
        else:
            # Use manual registry
            if name not in self._error_mitigation_constructors:
                raise QuantumHardwareError(
                    f"Error mitigation strategy '{name}' not registered."
                )
            constructor = self._error_mitigation_constructors[name]
            return constructor(**kwargs)

    def list_backends(self) -> List[str]:
        """List available backend names."""
        return list(self._backend_constructors.keys())

    def list_error_mitigation_strategies(self) -> List[str]:
        """List available error mitigation strategy names."""
        if ErrorMitigationFactory:
            # Assuming factory has a method to list strategies
            if hasattr(ErrorMitigationFactory, "list_strategies"):
                return ErrorMitigationFactory.list_strategies()
            else:
                # Fallback: return known strategies manually if factory exists but no list method
                return [
                    "EnhancedReadoutErrorMitigation",
                    "DynamicalDecouplingMitigation",
                    "ZeroNoiseExtrapolation",
                ]
        else:
            return list(self._error_mitigation_constructors.keys())


@retry_quantum_operation(max_retries=3)
@with_error_handling
def enhanced_execute_with_hardware(
    circuit: cirq.Circuit,
    backend: Union[str, QuantumBackend, EnhancedQuantumBackend],
    error_mitigation: Any = None,
    repetitions: int = 1000,
    provider: Optional[EnhancedQuantumHardwareProvider] = None,
    **kwargs,
) -> EnhancedExecutionResult:
    """
    Execute a circuit on hardware with enhanced capabilities.

    Args:
        circuit: Cirq circuit to execute
        backend: Quantum backend to use (name or instance)
        error_mitigation: Optional error mitigation strategy
        repetitions: Number of repetitions (shots)
        provider: Optional quantum hardware provider
        **kwargs: Additional arguments

    Returns:
        Enhanced execution result
    """
    operation_id = quantum_logger.start_operation("enhanced_execute_with_hardware")
    start_time = time.time()

    try:
        # Create provider if not provided
        if provider is None:
            provider = default_provider  # Use the globally defined default provider

        # Get backend if string name provided
        if isinstance(backend, str):
            # Pass kwargs to get_backend for potential backend initialization
            backend_instance = provider.get_backend(backend, enhanced=False, **kwargs)
        elif isinstance(backend, EnhancedQuantumBackend):
            backend_instance = backend  # Already enhanced
        elif isinstance(backend, QuantumBackend):
            # Wrap base backend instance
            backend_instance = EnhancedQuantumBackend(backend)
        else:
            raise TypeError(
                f"Invalid backend type: {type(backend)}. Expected str, QuantumBackend, or EnhancedQuantumBackend."
            )

        # Ensure we have an enhanced backend instance
        if not isinstance(backend_instance, EnhancedQuantumBackend):
            # This case should ideally not happen due to the logic above, but as a safeguard:
            if isinstance(backend_instance, QuantumBackend):
                backend_instance = EnhancedQuantumBackend(backend_instance)
            else:
                # If it's still not the right type, raise error
                raise TypeError(
                    f"Failed to obtain EnhancedQuantumBackend. Got {type(backend_instance)}."
                )

        # Get error mitigation strategy if string name provided
        if isinstance(error_mitigation, str):
            try:
                error_mitigation = provider.get_error_mitigation(
                    error_mitigation, **kwargs
                )
            except Exception as e:
                quantum_logger.warning(
                    f"Could not create error mitigation strategy '{error_mitigation}': {e}"
                )
                error_mitigation = None  # Proceed without mitigation

        # Execute circuit using the enhanced backend instance
        raw_results = backend_instance.run_circuit(
            circuit,
            repetitions=repetitions,
            error_mitigation=error_mitigation,
            **kwargs,
        )

        # Create enhanced result
        execution_time = time.time() - start_time
        result = EnhancedExecutionResult(
            raw_results=raw_results,
            circuit=circuit,
            backend_name=backend_instance.name,
            execution_time=execution_time,
        )

        # Log success
        quantum_logger.end_operation(
            operation_id,
            {
                "success": True,
                "backend_name": backend_instance.name,
                "has_error_mitigation": error_mitigation is not None,
                "shots": repetitions,
                "execution_time": execution_time,
                "circuit_qubits": len(list(circuit.all_qubits())),
                "circuit_depth": len(circuit),
            },
        )

        return result

    except Exception as e:
        # Log failure
        execution_time = time.time() - start_time
        backend_name = (
            getattr(backend, "name", str(backend))
            if isinstance(backend, (QuantumBackend, EnhancedQuantumBackend))
            else (backend if isinstance(backend, str) else "unknown")
        )

        quantum_logger.end_operation(
            operation_id,
            {
                "success": False,
                "error": str(e),
                "backend_name": backend_name,
                "execution_time": execution_time,
            },
        )

        # Re-raise the exception
        if isinstance(e, QuantumHardwareError):
            raise
        else:
            raise QuantumHardwareError(f"Enhanced execution failed: {str(e)}") from e


@retry_quantum_operation(max_retries=3)
@with_error_handling
def enhanced_expectation_with_hardware(
    circuit: cirq.Circuit,
    observable: Any,
    backend: Union[str, QuantumBackend, EnhancedQuantumBackend],
    error_mitigation: Any = None,
    repetitions: int = 1000,
    provider: Optional[EnhancedQuantumHardwareProvider] = None,
    **kwargs,
) -> float:
    """
    Calculate expectation value of an observable on hardware with enhanced capabilities.

    Args:
        circuit: Cirq circuit to execute
        observable: Observable (Pauli string or list of terms with coefficients)
        backend: Quantum backend to use (name or instance)
        error_mitigation: Optional error mitigation strategy
        repetitions: Number of repetitions (shots)
        provider: Optional quantum hardware provider
        **kwargs: Additional arguments

    Returns:
        Expectation value
    """
    # Execute circuit and get enhanced result
    result = enhanced_execute_with_hardware(
        circuit=circuit,
        backend=backend,
        error_mitigation=error_mitigation,
        repetitions=repetitions,
        provider=provider,
        **kwargs,
    )

    # Calculate expectation value
    expectation = result.get_expectation(observable)

    return expectation


def validate_and_repair_circuit(
    circuit: cirq.Circuit,
    backend: Union[str, QuantumBackend, EnhancedQuantumBackend],
    provider: Optional[EnhancedQuantumHardwareProvider] = None,
) -> Tuple[cirq.Circuit, bool, str]:
    """
    Validate a circuit against backend constraints and attempt to repair if needed.

    Args:
        circuit: Circuit to validate
        backend: Quantum backend to validate against
        provider: Optional quantum hardware provider

    Returns:
        Tuple of (repaired circuit, is_valid, message)
    """
    # Get provider if not provided
    if provider is None:
        provider = default_provider

    # Get backend instance if name provided
    if isinstance(backend, str):
        backend_instance = provider.get_backend(backend, enhanced=False)
    elif isinstance(backend, EnhancedQuantumBackend):
        backend_instance = backend.backend  # Use the underlying base backend
    elif isinstance(backend, QuantumBackend):
        backend_instance = backend
    else:
        raise TypeError("Invalid backend type for validation")

    # Validate the circuit using the base backend instance
    is_valid, error_message = validate_quantum_circuit(circuit, backend_instance)

    if is_valid:
        return circuit, True, "Circuit is valid"

    # Attempt to repair the circuit (basic repair logic)
    repaired_circuit = circuit.copy()
    repaired = False
    repair_message = error_message

    # Check for common issues and try to fix them
    if (
        "Circuit uses" in error_message
        and "qubits, but backend supports at most" in error_message
    ):
        # Too many qubits - simple truncation
        try:
            max_qubits_str = error_message.split("supports at most ")[-1].split(
                " qubits"
            )[0]
            if max_qubits_str and str(max_qubits_str).isdigit():
                max_qubits = int(max_qubits_str)
                if max_qubits > 0:
                    qubits = sorted(circuit.all_qubits())
                    if len(qubits) > max_qubits:
                        keep_qubits = qubits[:max_qubits]
                        new_circuit = cirq.Circuit(
                            op
                            for moment in circuit
                            for op in moment
                            if all(q in keep_qubits for q in op.qubits)
                        )
                        # Check if measurements were removed unintentionally
                        if any(
                            cirq.is_measurement(op) for op in circuit.all_operations()
                        ) and not any(
                            cirq.is_measurement(op) for op in new_circuit.all_operations()
                        ):
                            # Re-add measurements for kept qubits
                            new_circuit.append(cirq.measure(*keep_qubits, key="result"))

                        repaired_circuit = new_circuit
                        repaired = True
                        repair_message = f"Reduced circuit to use {max_qubits} qubits"
        except Exception as repair_err:
            quantum_logger.warning(
                f"Failed to apply qubit reduction repair: {repair_err}"
            )

    elif "Circuit uses unsupported gates" in error_message:
        # Unsupported gates - potentially attempt decomposition later
        repair_message = (
            "Cannot automatically repair unsupported gates (decomposition needed)"
        )

    # Validate the repaired circuit if repair was attempted
    if repaired:
        is_valid_after_repair, new_error_message = validate_quantum_circuit(
            repaired_circuit, backend_instance
        )
        if not is_valid_after_repair:
            repair_message += (
                f", but still invalid after repair attempt: {new_error_message}"
            )
            repaired = False  # Repair failed
            is_valid = False
        else:
            is_valid = True  # Repair succeeded

    return repaired_circuit, is_valid, repair_message


# Create default provider instance
default_provider = EnhancedQuantumHardwareProvider()


# Export for easy imports
__all__ = [
    "EnhancedQuantumBackend",
    "EnhancedExecutionResult",
    "EnhancedQuantumHardwareProvider",
    "enhanced_execute_with_hardware",
    "enhanced_expectation_with_hardware",
    "validate_and_repair_circuit",
    "default_provider",
]
