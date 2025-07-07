#!/usr/bin/env python3
"""
Quantum Hardware Logging Module

This module provides logging utilities specifically designed for quantum hardware operations.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

# Set up logging configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create logger
logger = logging.getLogger("quantum_hardware")
logger.setLevel(DEFAULT_LOG_LEVEL)

# Add console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(console_handler)

# Create file handler if log directory exists
LOG_DIR = os.environ.get(
    "QUANTUM_LOG_DIR", os.path.join(os.path.dirname(__file__), "logs")
)
if not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
    except Exception as e:
        logger.warning(f"Could not create log directory at {LOG_DIR}: {e}")

LOG_FILE = os.path.join(
    LOG_DIR, f"quantum_hardware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create log file at {LOG_FILE}: {e}")


class QuantumLogger:
    """Class for logging quantum hardware operations with performance metrics."""

    def __init__(self, component: str = "general", log_level: int = None):
        """
        Initialize the quantum logger.

        Args:
            component: Component name for this logger instance
            log_level: Optional log level to override default
        """
        self.component = component
        self.logger = logging.getLogger(f"quantum_hardware.{component}")
        if log_level is not None:
            self.logger.setLevel(log_level)
        self.operation_timers = {}

    def start_operation(self, operation_name: str) -> str:
        """
        Start timing an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Operation ID for use with end_operation
        """
        operation_id = f"{operation_name}_{time.time()}"
        self.operation_timers[operation_id] = {
            "start_time": time.time(),
            "name": operation_name,
        }
        self.logger.debug(f"Started operation: {operation_name}")
        return operation_id

    def end_operation(self, operation_id: str, metadata: Dict = None) -> Dict:
        """
        End timing an operation and log results.

        Args:
            operation_id: Operation ID from start_operation
            metadata: Additional metadata to log

        Returns:
            Dictionary with operation timing information
        """
        if operation_id not in self.operation_timers:
            self.logger.warning(f"Unknown operation ID: {operation_id}")
            return {}

        timer_data = self.operation_timers.pop(operation_id)
        end_time = time.time()
        duration = end_time - timer_data["start_time"]

        result = {
            "operation": timer_data["name"],
            "duration_seconds": duration,
            "start_time": timer_data["start_time"],
            "end_time": end_time,
        }

        if metadata:
            result["metadata"] = metadata

        self.logger.info(
            f"Completed operation: {timer_data['name']} in {duration:.4f}s"
        )
        if metadata:
            self.logger.debug(
                f"Operation metadata: {json.dumps(metadata, default=str)}"
            )

        return result

    def log_circuit_execution(
        self,
        backend_name: str,
        n_qubits: int,
        depth: int,
        shots: int,
        success: bool,
        execution_time: float,
        error_data: Dict = None,
    ) -> None:
        """
        Log information about a circuit execution.

        Args:
            backend_name: Name of the quantum backend
            n_qubits: Number of qubits in the circuit
            depth: Circuit depth
            shots: Number of shots executed
            success: Whether execution was successful
            execution_time: Time taken for execution
            error_data: Error data if execution failed
        """
        log_data = {
            "backend": backend_name,
            "n_qubits": n_qubits,
            "circuit_depth": depth,
            "shots": shots,
            "success": success,
            "execution_time": execution_time,
        }

        if error_data:
            log_data["error_data"] = error_data

        if success:
            self.logger.info(
                f"Circuit execution successful on {backend_name}: {n_qubits} qubits, {depth} depth, {shots} shots, {execution_time:.4f}s"
            )
        else:
            self.logger.error(
                f"Circuit execution failed on {backend_name}: {error_data.get('message', 'Unknown error')}"
            )

        # Log detailed data at debug level
        self.logger.debug(
            f"Circuit execution details: {json.dumps(log_data, default=str)}"
        )

    def log_error(
        self,
        component: str,
        error_type: str,
        message: str,
        exception: Exception = None,
        circuit_info: Dict = None,
    ) -> None:
        """
        Log an error with detailed information.

        Args:
            component: Component where the error occurred
            error_type: Type of error
            message: Error message
            exception: Exception object if available
            circuit_info: Circuit information if relevant
        """
        error_data = {
            "component": component,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }

        if exception:
            error_data["exception"] = str(exception)
            error_data["exception_type"] = exception.__class__.__name__

        if circuit_info:
            error_data["circuit_info"] = circuit_info

        self.logger.error(f"{component} - {error_type}: {message}")
        if exception:
            self.logger.error(f"Exception details: {exception}")

        # Log full error data at debug level
        self.logger.debug(f"Error details: {json.dumps(error_data, default=str)}")

    def log_calibration(
        self,
        backend_name: str,
        calibration_type: str,
        qubits: List[int],
        success: bool,
        metrics: Dict = None,
        duration: float = None,
    ) -> None:
        """
        Log calibration information.

        Args:
            backend_name: Name of the quantum backend
            calibration_type: Type of calibration performed
            qubits: List of qubits calibrated
            success: Whether calibration was successful
            metrics: Calibration metrics if available
            duration: Duration of calibration if available
        """
        log_data = {
            "backend": backend_name,
            "calibration_type": calibration_type,
            "qubits": qubits,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }

        if metrics:
            log_data["metrics"] = metrics

        if duration:
            log_data["duration"] = duration

        if success:
            self.logger.info(
                f"Calibration {calibration_type} successful on {backend_name}: {len(qubits)} qubits"
            )
        else:
            self.logger.error(
                f"Calibration {calibration_type} failed on {backend_name}"
            )

        # Log detailed data at debug level
        self.logger.debug(f"Calibration details: {json.dumps(log_data, default=str)}")


# Singleton instance for global use
quantum_logger = QuantumLogger()


def set_log_level(level: Union[int, str]) -> None:
    """
    Set the log level for all quantum loggers.

    Args:
        level: Logging level (can be integer or string like 'DEBUG', 'INFO', etc.)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_logging(
    log_file: Optional[str] = None,
    log_level: Union[int, str] = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """
    Configure logging settings.

    Args:
        log_file: Path to log file (if None, use default)
        log_level: Logging level
        log_format: Log format string
    """
    # Convert string log level to int if needed
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Set log level
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure handlers
    for handler in logger.handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)

    # Add file handler if specified
    if log_file:
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to configure log file {log_file}: {e}")


# Export for easy imports
__all__ = ["QuantumLogger", "quantum_logger", "set_log_level", "configure_logging"]
