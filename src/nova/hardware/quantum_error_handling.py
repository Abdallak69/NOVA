#!/usr/bin/env python3
"""
Quantum Hardware Error Handling Module

This module provides enhanced error handling for quantum hardware operations,
including connection failures, timeout management, and retry mechanisms.
"""

import functools
import socket
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import cirq
import requests

# Import the quantum logger if available, otherwise use basic logging
try:
    from nova.core.quantum_logger import quantum_logger as logger
except ImportError:
    import logging

    logger = logging.getLogger("quantum_hardware.error_handling")


class QuantumErrorType(Enum):
    """Enumeration of quantum hardware error types for classification."""

    CONNECTION_ERROR = "connection_error"
    AUTHENTICATION_ERROR = "authentication_error"
    TIMEOUT_ERROR = "timeout_error"
    HARDWARE_ERROR = "hardware_error"
    CALIBRATION_ERROR = "calibration_error"
    VALIDATION_ERROR = "validation_error"
    COMPATIBILITY_ERROR = "compatibility_error"
    RESOURCE_ERROR = "resource_error"
    QUOTA_ERROR = "quota_error"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_ERROR = "unknown_error"


class QuantumHardwareError(Exception):
    """Base exception class for quantum hardware errors with enhanced metadata."""

    def __init__(
        self,
        message: str,
        error_type: QuantumErrorType = QuantumErrorType.UNKNOWN_ERROR,
        error_data: Dict = None,
        original_exception: Exception = None,
        backend_name: str = None,
        circuit_info: Dict = None,
        retry_info: Dict = None,
    ):
        """
        Initialize the quantum hardware error.

        Args:
            message: Error message
            error_type: Type of error from QuantumErrorType enum
            error_data: Additional error data
            original_exception: Original exception if this is a wrapper
            backend_name: Name of the quantum backend
            circuit_info: Information about the circuit being executed
            retry_info: Information about retry attempts
        """
        self.error_type = error_type
        self.error_data = error_data or {}
        self.original_exception = original_exception
        self.backend_name = backend_name
        self.circuit_info = circuit_info
        self.retry_info = retry_info
        self.timestamp = time.time()

        # Build detailed error message
        detailed_message = f"{error_type.value}: {message}"
        if backend_name:
            detailed_message += f" (Backend: {backend_name})"

        super().__init__(detailed_message)

        # Log the error
        try:
            logger.log_error(
                component="hardware",
                error_type=error_type.value,
                message=message,
                exception=original_exception,
                circuit_info=circuit_info,
            )
        except Exception:
            # Fallback logging if the logger fails
            print(f"ERROR: {detailed_message}")
            if original_exception:
                print(f"Caused by: {original_exception}")


class QuantumConnectionError(QuantumHardwareError):
    """Error raised when connection to quantum hardware fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_type=QuantumErrorType.CONNECTION_ERROR, **kwargs
        )


class QuantumAuthenticationError(QuantumHardwareError):
    """Error raised when authentication to quantum hardware fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_type=QuantumErrorType.AUTHENTICATION_ERROR, **kwargs
        )


class QuantumTimeoutError(QuantumHardwareError):
    """Error raised when a quantum hardware operation times out."""

    def __init__(self, message: str, timeout_value: float = None, **kwargs):
        if timeout_value:
            kwargs["error_data"] = kwargs.get("error_data", {})
            kwargs["error_data"]["timeout_value"] = timeout_value
        super().__init__(message, error_type=QuantumErrorType.TIMEOUT_ERROR, **kwargs)


class QuantumHardwareUnavailableError(QuantumHardwareError):
    """Error raised when quantum hardware is unavailable."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type=QuantumErrorType.HARDWARE_ERROR, **kwargs)


class QuantumCalibrationError(QuantumHardwareError):
    """Error raised when quantum hardware calibration fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_type=QuantumErrorType.CALIBRATION_ERROR, **kwargs
        )


class QuantumValidationError(QuantumHardwareError):
    """Error raised when circuit validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_type=QuantumErrorType.VALIDATION_ERROR, **kwargs
        )


class QuantumBackendCompatibilityError(QuantumHardwareError):
    """Error raised when a circuit is incompatible with a backend."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, error_type=QuantumErrorType.COMPATIBILITY_ERROR, **kwargs
        )


class QuantumQuotaError(QuantumHardwareError):
    """Error raised when user exceeds quantum hardware usage quota."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_type=QuantumErrorType.QUOTA_ERROR, **kwargs)


# Error mapping dictionary for common errors
ERROR_MAPPING = {
    ConnectionError: QuantumConnectionError,
    socket.timeout: QuantumTimeoutError,
    socket.error: QuantumConnectionError,
    TimeoutError: QuantumTimeoutError,
    PermissionError: QuantumAuthenticationError,
    requests.exceptions.Timeout: QuantumTimeoutError,
    requests.exceptions.ConnectionError: QuantumConnectionError,
    requests.exceptions.HTTPError: QuantumConnectionError,
    ValueError: QuantumValidationError,
}


def map_exception(
    exception: Exception,
    default_message: str = "Unknown quantum hardware error",
    backend_name: str = None,
    circuit_info: Dict = None,
) -> QuantumHardwareError:
    """
    Map a standard exception to a quantum hardware exception.

    Args:
        exception: Original exception to map
        default_message: Default error message
        backend_name: Name of the quantum backend
        circuit_info: Information about the circuit being executed

    Returns:
        Mapped quantum hardware exception
    """
    # Extract message from original exception
    message = str(exception) or default_message

    # Check for direct exception type match
    for exception_type, quantum_error_class in ERROR_MAPPING.items():
        if isinstance(exception, exception_type):
            return quantum_error_class(
                message=message,
                original_exception=exception,
                backend_name=backend_name,
                circuit_info=circuit_info,
            )

    # If no match found, return generic quantum hardware error
    return QuantumHardwareError(
        message=message,
        original_exception=exception,
        backend_name=backend_name,
        circuit_info=circuit_info,
    )


def retry_quantum_operation(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    allowed_exceptions: List[Type[Exception]] = None,
    retry_on_result: Callable[[Any], bool] = None,
):
    """
    Decorator to retry quantum operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (will increase exponentially)
        allowed_exceptions: List of exception types to retry on (defaults to connection errors)
        retry_on_result: Optional function to evaluate if result should trigger retry

    Returns:
        Decorated function
    """
    if allowed_exceptions is None:
        # Default to common connection-related exceptions
        allowed_exceptions = [
            QuantumConnectionError,
            QuantumTimeoutError,
            QuantumHardwareUnavailableError,
            ConnectionError,
            TimeoutError,
            socket.timeout,
            socket.error,
            requests.exceptions.RequestException,
        ]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract backend name for error reporting if available
            backend_name = None
            if args and hasattr(args[0], "name"):
                backend_name = args[0].name
            elif "backend" in kwargs and hasattr(kwargs["backend"], "name"):
                backend_name = kwargs["backend"].name

            # Circuit info for error reporting
            circuit_info = None
            if "circuit" in kwargs and isinstance(kwargs["circuit"], cirq.Circuit):
                circuit = kwargs["circuit"]
                circuit_info = {
                    "n_qubits": len(list(circuit.all_qubits())),
                    "n_moments": len(circuit),
                    "has_measurements": any(
                        isinstance(op.gate, cirq.MeasurementGate)
                        for op in circuit.all_operations()
                    ),
                }

            retries = 0
            last_exception = None

            while retries <= max_retries:
                try:
                    # Attempt the operation
                    result = func(*args, **kwargs)

                    # Check if result should trigger retry
                    if retry_on_result and retry_on_result(result):
                        if retries >= max_retries:
                            logger.logger.warning(
                                f"Operation {func.__name__} returned unsuccessful result "
                                f"after {retries} retries, giving up."
                            )
                            break
                        retries += 1
                        delay = retry_delay * (
                            2 ** (retries - 1)
                        )  # Exponential backoff
                        logger.logger.info(
                            f"Retrying operation {func.__name__} due to unsuccessful result "
                            f"(attempt {retries}/{max_retries}, waiting {delay:.2f}s)"
                        )
                        time.sleep(delay)
                        continue

                    # Operation was successful but indicated failure
                    logger.logger.warning(
                        f"Operation {func.__name__} returned unsuccessful result "
                        f"with status {result.get('status', 'unknown')}"
                    )
                    return result

                except tuple(allowed_exceptions) as e:
                    last_exception = e
                    if retries >= max_retries:
                        break

                    retries += 1
                    delay = retry_delay * (2 ** (retries - 1))  # Exponential backoff

                    logger.logger.warning(
                        f"Operation {func.__name__} failed with error: {e}. "
                        f"Retrying ({retries}/{max_retries})..."
                    )
                    time.sleep(delay)

            # If we get here, all retries failed
            retry_info = {
                "attempts": retries,
                "max_retries": max_retries,
                "total_delay": sum(retry_delay * (2**i) for i in range(retries)),
            }

            if last_exception is None:
                raise QuantumHardwareError(
                    message=f"Operation {func.__name__} failed after {retries} retries",
                    error_type=QuantumErrorType.UNKNOWN_ERROR,
                    backend_name=backend_name,
                    circuit_info=circuit_info,
                    retry_info=retry_info,
                )

            # Map the last exception to a quantum hardware error
            if isinstance(last_exception, QuantumHardwareError):
                # If it's already a quantum hardware error, just update retry info
                last_exception.retry_info = retry_info
                raise last_exception
            else:
                # Map standard exception to quantum hardware error
                quantum_error = map_exception(
                    last_exception,
                    default_message=f"Operation {func.__name__} failed after {retries} retries",
                    backend_name=backend_name,
                    circuit_info=circuit_info,
                )
                quantum_error.retry_info = retry_info
                raise quantum_error

        return wrapper

    return decorator


def with_error_handling(
    func=None,
    timeout: float = None,
    error_mapping: Dict[Type[Exception], Type[QuantumHardwareError]] = None,
):
    """
    Decorator to add error handling to quantum operations.

    Args:
        func: Function to decorate
        timeout: Optional timeout in seconds
        error_mapping: Optional custom error mapping

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract backend name for error reporting if available
            backend_name = None
            if args and hasattr(args[0], "name"):
                backend_name = args[0].name
            elif "backend" in kwargs and hasattr(kwargs["backend"], "name"):
                backend_name = kwargs["backend"].name

            # Circuit info for error reporting
            circuit_info = None
            if "circuit" in kwargs and isinstance(kwargs["circuit"], cirq.Circuit):
                circuit = kwargs["circuit"]
                circuit_info = {
                    "n_qubits": len(list(circuit.all_qubits())),
                    "n_moments": len(circuit),
                    "has_measurements": any(
                        isinstance(op.gate, cirq.MeasurementGate)
                        for op in circuit.all_operations()
                    ),
                }

            try:
                # Handle timeout if specified
                if timeout:
                    import signal

                    def timeout_handler(signum, frame):
                        raise QuantumTimeoutError(
                            message=f"Operation {func.__name__} timed out after {timeout}s",
                            timeout_value=timeout,
                            backend_name=backend_name,
                            circuit_info=circuit_info,
                        )

                    # Set up timeout handler
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(timeout))

                    try:
                        result = func(*args, **kwargs)
                    finally:
                        # Restore previous handler and cancel alarm
                        signal.signal(signal.SIGALRM, old_handler)
                        signal.alarm(0)
                else:
                    result = func(*args, **kwargs)

                return result

            except QuantumHardwareError:
                # Already a quantum hardware error, re-raise
                raise
            except Exception as e:
                # Map the exception using custom mapping or default mapping
                mapping = error_mapping or ERROR_MAPPING

                for exception_type, quantum_error_class in mapping.items():
                    if isinstance(e, exception_type):
                        raise quantum_error_class(
                            message=str(e) or f"Error in {func.__name__}",
                            original_exception=e,
                            backend_name=backend_name,
                            circuit_info=circuit_info,
                        )

                # If no match found, raise generic quantum hardware error
                raise QuantumHardwareError(
                    message=str(e) or f"Unknown error in {func.__name__}",
                    original_exception=e,
                    backend_name=backend_name,
                    circuit_info=circuit_info,
                )

        return wrapper

    # Handle both @with_error_handling and @with_error_handling(timeout=10)
    if func is None:
        return decorator
    else:
        return decorator(func)


def validate_quantum_circuit(
    circuit: cirq.Circuit, backend: Any
) -> Tuple[bool, Optional[str]]:
    """
    Validate a quantum circuit against backend constraints.

    Args:
        circuit: Cirq circuit to validate
        backend: Quantum backend

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if circuit is valid
        if not isinstance(circuit, cirq.Circuit):
            return False, "Invalid circuit type, expected cirq.Circuit"

        # Check if circuit has operations
        if len(circuit) == 0:
            return False, "Empty circuit"

        # Get all qubits in the circuit
        qubits = list(circuit.all_qubits())

        # Check number of qubits against backend limit
        if hasattr(backend, "max_qubits") and len(qubits) > backend.max_qubits:
                return (
                    False,
                    f"Circuit uses {len(qubits)} qubits, but backend supports at most {backend.max_qubits}",
                )

        # Check available gates
        if hasattr(backend, "available_gates"):
            available_gates = backend.available_gates

            # Always allow measurement gates regardless of available_gates list
            available_gates = list(available_gates) + ["Measurement"]

            unsupported_gates = []

            for op in circuit.all_operations():
                gate_name = None
                if hasattr(op.gate, "name"):
                    gate_name = op.gate.name
                elif isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
                    gate_name = "X"
                elif isinstance(op.gate, cirq.YPowGate) and op.gate.exponent == 1:
                    gate_name = "Y"
                elif isinstance(op.gate, cirq.ZPowGate) and op.gate.exponent == 1:
                    gate_name = "Z"
                elif isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1:
                    gate_name = "H"
                elif isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1:
                    gate_name = "CZ"
                elif isinstance(op.gate, cirq.CXPowGate) and op.gate.exponent == 1:
                    gate_name = "CNOT"
                elif isinstance(op.gate, cirq.MeasurementGate):
                    gate_name = "Measurement"
                else:
                    gate_name = str(op.gate)

                if (
                    gate_name not in available_gates
                    and gate_name not in unsupported_gates
                ):
                    unsupported_gates.append(gate_name)

            if unsupported_gates:
                return (
                    False,
                    f"Circuit uses unsupported gates: {', '.join(unsupported_gates)}",
                )

        # Check if backend has a device with connectivity constraints
        if hasattr(backend, "device") and hasattr(backend.device, "validate_operation"):
            try:
                for moment in circuit:
                    for op in moment:
                        # Skip measurement gate validation as it's always allowed
                        if isinstance(op.gate, cirq.MeasurementGate):
                            continue

                        if not backend.device.validate_operation(op):
                            return (
                                False,
                                f"Operation {op} is not supported by the device",
                            )
            except Exception as e:
                return False, f"Circuit validation failed: {str(e)}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def check_backend_health(backend: Any) -> Dict[str, Any]:
    """
    Check the health and availability of a quantum backend.

    Args:
        backend: Quantum backend to check

    Returns:
        Dictionary with health check results
    """
    health_data = {
        "name": getattr(backend, "name", "Unknown"),
        "available": False,
        "status": "unknown",
        "errors": [],
        "timestamp": time.time(),
    }

    try:
        # Check if backend has a status or is_online method
        if hasattr(backend, "status"):
            status = backend.status()
            health_data["status"] = status
            health_data["available"] = status == "online"
        elif hasattr(backend, "is_online"):
            online = backend.is_online()
            health_data["available"] = online
            health_data["status"] = "online" if online else "offline"
        else:
            # Try to run a simple test circuit
            test_circuit = cirq.Circuit(
                cirq.H(cirq.LineQubit(0)), cirq.measure(cirq.LineQubit(0), key="result")
            )

            if hasattr(backend, "run_circuit"):
                _ = backend.run_circuit(test_circuit, repetitions=10)
                health_data["available"] = True
                health_data["status"] = "online"

        # Add backend properties if available
        if hasattr(backend, "get_device_properties"):
            health_data["properties"] = backend.get_device_properties()

    except Exception as e:
        health_data["available"] = False
        health_data["status"] = "error"
        health_data["errors"].append(str(e))

    return health_data


def get_detailed_backend_diagnostics(backend: Any) -> Dict[str, Any]:
    """
    Get detailed diagnostics for a quantum backend.

    Args:
        backend: Quantum backend to check

    Returns:
        Dictionary with detailed diagnostics
    """
    diagnostics = {
        "backend": getattr(backend, "name", "Unknown"),
        "timestamp": time.time(),
        "errors": [],
        "warnings": [],
        "connectivity": {},
        "gate_errors": {},
        "readout_errors": {},
        "qubit_t1_times": {},
        "qubit_t2_times": {},
    }

    try:
        # Check basic health first
        health = check_backend_health(backend)
        diagnostics.update(
            {"available": health["available"], "status": health["status"]}
        )

        if not health["available"]:
            diagnostics["errors"].append("Backend is not available")
            return diagnostics

        # Get backend properties if available
        if hasattr(backend, "properties"):
            properties = backend.properties()

            # Extract qubit properties
            if hasattr(properties, "qubits"):
                for i, qubit_props in enumerate(properties.qubits):
                    for prop in qubit_props:
                        if prop.name == "T1":
                            diagnostics["qubit_t1_times"][i] = prop.value
                        elif prop.name == "T2":
                            diagnostics["qubit_t2_times"][i] = prop.value
                        elif prop.name == "readout_error":
                            diagnostics["readout_errors"][i] = prop.value

            # Extract gate properties
            if hasattr(properties, "gates"):
                for gate_props in properties.gates:
                    gate_name = gate_props.gate
                    params = gate_props.parameters
                    for param in params:
                        if param.name == "gate_error":
                            diagnostics["gate_errors"][gate_name] = param.value

            # Extract connectivity
            if hasattr(properties, "coupling_map"):
                diagnostics["connectivity"]["coupling_map"] = properties.coupling_map

        # Check for device-specific properties
        if hasattr(backend, "device"):
            device = backend.device
            if hasattr(device, "qubits"):
                diagnostics["connectivity"]["n_qubits"] = len(device.qubits)

            # Get connectivity graph if available
            if hasattr(device, "qubit_pairs"):
                diagnostics["connectivity"]["connected_pairs"] = [
                    (q1.x, q1.y, q2.x, q2.y) for q1, q2 in device.qubit_pairs()
                ]

    except Exception as e:
        diagnostics["errors"].append(f"Error getting diagnostics: {str(e)}")

    return diagnostics


def global_error_handler(func):
    """
    Decorator to provide global error handling for any function.

    This decorator wraps a function with comprehensive error handling
    that maps generic exceptions to quantum-specific ones with user-friendly
    messages and provides appropriate fallbacks where possible.

    Args:
        func: The function to wrap with error handling

    Returns:
        Decorated function with error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except QuantumHardwareError:
            # Already a quantum error, just re-raise it
            raise
        except KeyboardInterrupt:
            logger.logger.warning("Operation interrupted by user")
            raise QuantumHardwareError(
                message="Operation interrupted by user",
                error_type=QuantumErrorType.INTERNAL_ERROR,
            )
        except MemoryError:
            logger.log_error(
                "error_handler", "MemoryError", "Out of memory during quantum operation"
            )
            raise QuantumHardwareError(
                message="Out of memory. Try reducing circuit size or number of qubits.",
                error_type=QuantumErrorType.RESOURCE_ERROR,
            )
        except ValueError as e:
            # Check for common value errors
            message = str(e)
            if "parameter" in message.lower():
                logger.log_error(
                    "error_handler",
                    "ValueError",
                    f"Invalid parameter: {message}",
                    exception=e,
                )
                raise QuantumValidationError(
                    message=f"Invalid parameter: {message}", original_exception=e
                )
            elif "circuit" in message.lower() and "valid" in message.lower():
                logger.log_error(
                    "error_handler",
                    "ValueError",
                    f"Invalid circuit: {message}",
                    exception=e,
                )
                raise QuantumValidationError(
                    message=f"Invalid quantum circuit: {message}", original_exception=e
                )
            else:
                logger.log_error(
                    "error_handler",
                    "ValueError",
                    f"Validation error: {message}",
                    exception=e,
                )
                raise QuantumValidationError(message=message, original_exception=e)
        except TypeError as e:
            message = str(e)
            logger.log_error(
                "error_handler",
                "TypeError",
                f"Type error in quantum operation: {message}",
                exception=e,
            )
            raise QuantumValidationError(
                message=f"Type error: {message}", original_exception=e
            )
        except (OSError, ConnectionError, requests.exceptions.ConnectionError) as e:
            logger.log_error(
                "error_handler",
                "ConnectionError",
                f"Connection error: {str(e)}",
                exception=e,
            )
            raise QuantumConnectionError(
                message=f"Failed to connect to quantum hardware: {str(e)}",
                original_exception=e,
            )
        except (TimeoutError, socket.timeout, requests.exceptions.Timeout) as e:
            logger.log_error(
                "error_handler", "TimeoutError", f"Timeout error: {str(e)}", exception=e
            )
            raise QuantumTimeoutError(
                message=f"Timeout while communicating with quantum hardware: {str(e)}",
                original_exception=e,
            )
        except PermissionError as e:
            logger.log_error(
                "error_handler",
                "PermissionError",
                f"Authentication error: {str(e)}",
                exception=e,
            )
            raise QuantumAuthenticationError(
                message=f"Authentication failed for quantum hardware access: {str(e)}",
                original_exception=e,
            )
        except NotImplementedError as e:
            logger.log_error(
                "error_handler",
                "NotImplementedError",
                f"Unsupported operation: {str(e)}",
                exception=e,
            )
            raise QuantumHardwareError(
                message=f"Unsupported operation: {str(e)}",
                error_type=QuantumErrorType.UNSUPPORTED_OPERATION,
                original_exception=e,
            )
        except FileNotFoundError as e:
            logger.log_error(
                "error_handler",
                "FileNotFoundError",
                f"File not found: {str(e)}",
                exception=e,
            )
            raise QuantumHardwareError(
                message=f"File not found: {str(e)}",
                error_type=QuantumErrorType.INTERNAL_ERROR,
                original_exception=e,
            )
        except Exception as e:
            # Generic fallback for other exceptions
            logger.log_error(
                "error_handler",
                "Exception",
                f"Unexpected error in quantum operation: {str(e)}",
                exception=e,
            )
            # No direct equivalent to debug in QuantumLogger, use logger.logger.debug instead
            logger.logger.debug(f"Exception details: {traceback.format_exc()}")

            # Try to map to a specific quantum error type based on the exception message
            message = str(e).lower()

            if "memory" in message or "resources" in message or "too large" in message:
                raise QuantumHardwareError(
                    message=f"Resource limit exceeded: {str(e)}",
                    error_type=QuantumErrorType.RESOURCE_ERROR,
                    original_exception=e,
                )
            elif "calibration" in message:
                raise QuantumCalibrationError(
                    message=f"Calibration error: {str(e)}", original_exception=e
                )
            elif "quota" in message or "limit" in message:
                raise QuantumQuotaError(
                    message=f"Quota exceeded: {str(e)}", original_exception=e
                )
            elif "timeout" in message or "timed out" in message:
                raise QuantumTimeoutError(
                    message=f"Operation timed out: {str(e)}", original_exception=e
                )
            elif (
                "authentication" in message
                or "permission" in message
                or "access" in message
            ):
                raise QuantumAuthenticationError(
                    message=f"Authentication error: {str(e)}", original_exception=e
                )
            elif (
                "connection" in message
                or "network" in message
                or "unreachable" in message
            ):
                raise QuantumConnectionError(
                    message=f"Connection error: {str(e)}", original_exception=e
                )
            elif "compatibility" in message or "not supported" in message:
                raise QuantumBackendCompatibilityError(
                    message=f"Compatibility error: {str(e)}", original_exception=e
                )
            else:
                # Default to generic hardware error if we can't identify a specific type
                raise QuantumHardwareError(
                    message=f"Unexpected error during quantum operation: {str(e)}",
                    error_type=QuantumErrorType.INTERNAL_ERROR,
                    original_exception=e,
                )

    return wrapper


def user_friendly_error_message(error: Exception) -> str:
    """
    Convert any exception to a user-friendly error message.

    Args:
        error: The exception to convert

    Returns:
        User-friendly error message string
    """
    if isinstance(error, QuantumHardwareError):
        # If it's already a quantum error, use its message and add recommendations
        message = str(error)

        # Add recommendations based on error type
        if error.error_type == QuantumErrorType.CONNECTION_ERROR:
            return f"{message}\n\nRecommendation: Check your internet connection and ensure the quantum hardware service is available."
        elif error.error_type == QuantumErrorType.AUTHENTICATION_ERROR:
            return f"{message}\n\nRecommendation: Check your credentials and ensure you have proper access permissions."
        elif error.error_type == QuantumErrorType.TIMEOUT_ERROR:
            return f"{message}\n\nRecommendation: The operation took too long. Try again later or with a smaller circuit."
        elif error.error_type == QuantumErrorType.HARDWARE_ERROR:
            return f"{message}\n\nRecommendation: The quantum hardware may be experiencing issues. Try again later or use a different backend."
        elif error.error_type == QuantumErrorType.CALIBRATION_ERROR:
            return f"{message}\n\nRecommendation: The quantum hardware may need recalibration. Try again later or use a different backend."
        elif error.error_type == QuantumErrorType.VALIDATION_ERROR:
            return f"{message}\n\nRecommendation: Check your circuit construction and parameters for errors."
        elif error.error_type == QuantumErrorType.COMPATIBILITY_ERROR:
            return f"{message}\n\nRecommendation: Modify your circuit to be compatible with the selected backend's capabilities."
        elif error.error_type == QuantumErrorType.RESOURCE_ERROR:
            return f"{message}\n\nRecommendation: Try reducing the size of your circuit or the number of qubits."
        elif error.error_type == QuantumErrorType.QUOTA_ERROR:
            return f"{message}\n\nRecommendation: You've exceeded your usage quota. Wait until your quota refreshes or request an increase."
        elif error.error_type == QuantumErrorType.UNSUPPORTED_OPERATION:
            return f"{message}\n\nRecommendation: This operation is not supported. Check the documentation for supported features."
        else:
            return message

    # For memory errors
    if isinstance(error, MemoryError):
        return "Out of memory error. Try reducing the number of qubits, circuit depth, or shots."

    # For keyboard interrupts
    if isinstance(error, KeyboardInterrupt):
        return "Operation cancelled by user."

    # For specific error types
    error_message = str(error)
    error_type = type(error).__name__

    # Standard error conversions
    if isinstance(error, ValueError):
        if "parameter" in error_message.lower():
            return f"Invalid parameter value: {error_message}"
        elif "circuit" in error_message.lower():
            return f"Invalid circuit configuration: {error_message}"
        return f"Invalid value: {error_message}"

    if isinstance(error, TypeError):
        return f"Type error: {error_message}"

    if isinstance(error, (ConnectionError, socket.error)):
        return "Connection error: Failed to connect to quantum hardware. Check your internet connection."

    if isinstance(error, (TimeoutError, socket.timeout)):
        return (
            "Timeout error: The operation took too long to complete. Try again later."
        )

    if isinstance(error, PermissionError):
        return (
            "Authentication error: You do not have permission to access this resource."
        )

    if isinstance(error, NotImplementedError):
        return f"Unsupported operation: {error_message}"

    if isinstance(error, FileNotFoundError):
        return f"File not found: {error_message}"

    # Generic message for other exceptions
    return f"Error ({error_type}): {error_message}"


def handle_errors_gracefully(func, fallback_value=None, log_error=True):
    """
    Execute a function with graceful error handling and fallback.

    Args:
        func: Function to execute
        fallback_value: Value to return if function raises an exception
        log_error: Whether to log the error

    Returns:
        Result of the function or fallback value if an error occurs
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.log_error(
                "error_handler",
                "Exception",
                f"Error in function {func.__name__}: {str(e)}",
                exception=e,
            )
            # No direct equivalent to debug in QuantumLogger, use logger.logger.debug instead
            logger.logger.debug(f"Exception details: {traceback.format_exc()}")

        # Create user-friendly error message
        friendly_message = user_friendly_error_message(e)

        # Return fallback value with error information
        if isinstance(fallback_value, dict):
            fallback_value.update(
                {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "friendly_error": friendly_message,
                    "success": False,
                }
            )
            return fallback_value
        else:
            return fallback_value


# Export for easy imports
__all__ = [
    "QuantumErrorType",
    "QuantumHardwareError",
    "QuantumConnectionError",
    "QuantumAuthenticationError",
    "QuantumTimeoutError",
    "QuantumHardwareUnavailableError",
    "QuantumCalibrationError",
    "QuantumValidationError",
    "QuantumBackendCompatibilityError",
    "QuantumQuotaError",
    "map_exception",
    "retry_quantum_operation",
    "with_error_handling",
    "validate_quantum_circuit",
    "check_backend_health",
    "get_detailed_backend_diagnostics",
    "global_error_handler",
    "user_friendly_error_message",
    "handle_errors_gracefully",
]
