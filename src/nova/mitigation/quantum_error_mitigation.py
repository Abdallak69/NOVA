#!/usr/bin/env python3
"""
Enhanced Quantum Error Mitigation Module

This module provides improved error mitigation strategies with better calibration routines
for quantum hardware execution.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import cirq
import numpy as np

# Import the quantum logger if available
try:
    from nova.core.quantum_logger import quantum_logger
except ImportError:
    import logging

    quantum_logger = logging.getLogger("quantum_hardware.error_mitigation")

# Import the error handling module if available
try:
    from nova.hardware.quantum_error_handling import (
        QuantumCalibrationError,
        retry_quantum_operation,
        with_error_handling,
    )
except ImportError:
    # Define fallbacks if module not available
    class QuantumCalibrationError(Exception):
        pass

    def with_error_handling(func):
        return func

    def retry_quantum_operation(max_retries=3):
        def decorator(func):
            return func

        return decorator


class CalibrationMethod(Enum):
    """Enumeration of calibration methods for error mitigation."""

    STANDARD = "standard"
    EXTENDED = "extended"
    INTERLEAVED = "interleaved"
    RANDOM = "random"
    CYCLE_BENCHMARKING = "cycle_benchmarking"


class ErrorMitigationStrategy:
    """Base class for enhanced error mitigation strategies."""

    def __init__(self, name: str = "base", calibration_shots: int = 1000):
        """
        Initialize the error mitigation strategy.

        Args:
            name: Name of the strategy
            calibration_shots: Number of shots to use for calibration
        """
        self.name = name
        self.calibration_shots = calibration_shots
        self.is_calibrated = False
        self.calibration_data = {}
        self.last_calibration_time = None

    def calibrate(self, backend: Any, qubits: List[cirq.Qid] = None, **kwargs) -> bool:
        """
        Calibrate the error mitigation strategy.

        Args:
            backend: Quantum backend to use
            qubits: Qubits to calibrate (if None, use all qubits)
            **kwargs: Additional arguments

        Returns:
            True if calibration was successful
        """
        raise NotImplementedError("Subclasses must implement calibrate method")

    def mitigate(self, circuit: cirq.Circuit, backend: Any, **kwargs) -> cirq.Circuit:
        """
        Apply error mitigation to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            Mitigated circuit or list of circuits
        """
        raise NotImplementedError("Subclasses must implement mitigate method")

    def process_results(self, results: Union[Dict, List[Dict]], **kwargs) -> Dict:
        """
        Process results after running the mitigated circuit.

        Args:
            results: Results from running the circuit
            **kwargs: Additional arguments

        Returns:
            Processed results
        """
        return results

    def get_calibration_status(self) -> Dict:
        """
        Get the calibration status.

        Returns:
            Dictionary with calibration status
        """
        return {
            "is_calibrated": self.is_calibrated,
            "strategy_name": self.name,
            "calibration_shots": self.calibration_shots,
            "last_calibration_time": self.last_calibration_time,
            "calibration_data_size": len(self.calibration_data)
            if self.calibration_data
            else 0,
        }

    def __str__(self) -> str:
        """String representation of the strategy."""
        status = "calibrated" if self.is_calibrated else "not calibrated"
        return f"{self.name} Error Mitigation ({status})"


class EnhancedReadoutErrorMitigation(ErrorMitigationStrategy):
    """Enhanced readout error mitigation with improved calibration."""

    def __init__(
        self,
        calibration_shots: int = 1000,
        calibration_method: CalibrationMethod = CalibrationMethod.STANDARD,
    ):
        """
        Initialize enhanced readout error mitigation.

        Args:
            calibration_shots: Number of shots for calibration
            calibration_method: Method to use for calibration
        """
        super().__init__(
            name="Enhanced Readout Error", calibration_shots=calibration_shots
        )
        self.calibration_method = calibration_method
        self.confusion_matrices = {}
        self.inverse_matrices = {}

    @retry_quantum_operation(max_retries=3)
    @with_error_handling
    def calibrate(self, backend: Any, qubits: List[cirq.Qid] = None, **kwargs) -> bool:
        """
        Calibrate readout errors for the given qubits.

        Args:
            backend: Quantum backend to use
            qubits: List of qubits to calibrate (if None, use all qubits)
            **kwargs: Additional arguments

        Returns:
            True if calibration was successful
        """
        # Start logging the calibration
        start_time = time.time()
        operation_id = quantum_logger.start_operation(f"calibrate_{self.name}")

        try:
            # Get qubits if not provided
            if qubits is None:
                if hasattr(backend, "device") and hasattr(backend.device, "qubits"):
                    qubits = backend.device.qubits
                else:
                    # Default to first 5 qubits as fallback
                    qubits = [cirq.LineQubit(i) for i in range(5)]

            n_qubits = len(qubits)
            quantum_logger.log_calibration(
                backend_name=getattr(backend, "name", "unknown"),
                calibration_type=f"{self.calibration_method.value}_readout",
                qubits=[q.x for q in qubits if hasattr(q, "x")],
                success=False,  # Will update to True if successful
                metrics={"n_qubits": n_qubits, "shots": self.calibration_shots},
            )

            # Reset calibration data
            self.confusion_matrices = {}
            self.inverse_matrices = {}

            # Determine calibration circuits based on method
            if self.calibration_method == CalibrationMethod.STANDARD:
                # Create calibration circuits for all computational basis states
                for state_idx in range(2**n_qubits):
                    # Convert state_idx to binary representation
                    binary = format(state_idx, f"0{n_qubits}b")

                    # Create circuit to prepare this state
                    circuit = cirq.Circuit()
                    for i, bit in enumerate(binary):
                        if bit == "1":
                            circuit.append(cirq.X(qubits[i]))

                    # Add measurements
                    circuit.append(cirq.measure(*qubits, key=f"state_{state_idx}"))

                    # Run the circuit
                    results = backend.run_circuit(
                        circuit, repetitions=self.calibration_shots
                    )

                    # Process results to build confusion matrix
                    for result_key, result_vals in results.items():
                        # Count occurrences of each outcome
                        counts = {}
                        for result in result_vals:
                            # Convert result to integer
                            result_int = int("".join(map(str, result)), 2)
                            counts[result_int] = counts.get(result_int, 0) + 1

                        # Normalize counts to get probabilities
                        for k in counts:
                            counts[k] /= self.calibration_shots

                        # Store in confusion matrix
                        self.confusion_matrices[state_idx] = counts

            elif self.calibration_method == CalibrationMethod.EXTENDED:
                # Extended method: run multiple times and average
                repetitions = 5
                for rep in range(repetitions):
                    # Create temporary matrices for this repetition
                    temp_matrices = {}

                    for state_idx in range(2**n_qubits):
                        # Convert state_idx to binary representation
                        binary = format(state_idx, f"0{n_qubits}b")

                        # Create circuit to prepare this state
                        circuit = cirq.Circuit()
                        for i, bit in enumerate(binary):
                            if bit == "1":
                                circuit.append(cirq.X(qubits[i]))

                        # Add measurements
                        circuit.append(
                            cirq.measure(*qubits, key=f"state_{state_idx}_rep{rep}")
                        )

                        # Run the circuit
                        results = backend.run_circuit(
                            circuit, repetitions=self.calibration_shots // repetitions
                        )

                        # Process results
                        for result_key, result_vals in results.items():
                            # Count occurrences
                            counts = {}
                            for result in result_vals:
                                result_int = int("".join(map(str, result)), 2)
                                counts[result_int] = counts.get(result_int, 0) + 1

                            # Normalize
                            for k in counts:
                                counts[k] /= len(result_vals)

                            # Store in temp matrices
                            if state_idx not in temp_matrices:
                                temp_matrices[state_idx] = {}

                            for k, v in counts.items():
                                if k not in temp_matrices[state_idx]:
                                    temp_matrices[state_idx][k] = []
                                temp_matrices[state_idx][k].append(v)

                    # Average the results across repetitions
                    for state_idx in temp_matrices:
                        if state_idx not in self.confusion_matrices:
                            self.confusion_matrices[state_idx] = {}

                        for outcome, values in temp_matrices[state_idx].items():
                            self.confusion_matrices[state_idx][outcome] = sum(
                                values
                            ) / len(values)

            else:
                # Other methods would be implemented here
                raise NotImplementedError(
                    f"Calibration method {self.calibration_method.value} not implemented"
                )

            # Compute inverse matrices for error correction
            self._compute_inverse_matrices()

            # Update calibration status
            self.is_calibrated = True
            self.last_calibration_time = time.time()

            # Log successful calibration
            duration = time.time() - start_time
            quantum_logger.log_calibration(
                backend_name=getattr(backend, "name", "unknown"),
                calibration_type=f"{self.calibration_method.value}_readout",
                qubits=[q.x for q in qubits if hasattr(q, "x")],
                success=True,
                metrics={
                    "n_qubits": n_qubits,
                    "shots": self.calibration_shots,
                    "matrix_size": len(self.confusion_matrices),
                    "method": self.calibration_method.value,
                },
                duration=duration,
            )

            quantum_logger.end_operation(
                operation_id,
                {
                    "success": True,
                    "duration": duration,
                    "n_qubits": n_qubits,
                    "method": self.calibration_method.value,
                },
            )

            return True

        except Exception as e:
            # Log failure
            duration = time.time() - start_time
            quantum_logger.log_calibration(
                backend_name=getattr(backend, "name", "unknown"),
                calibration_type=f"{self.calibration_method.value}_readout",
                qubits=[q.x for q in qubits if hasattr(q, "x")] if qubits else [],
                success=False,
                metrics={"error": str(e)},
                duration=duration,
            )

            quantum_logger.end_operation(
                operation_id, {"success": False, "duration": duration, "error": str(e)}
            )

            # Re-raise as calibration error
            raise QuantumCalibrationError(
                message=f"Readout error calibration failed: {str(e)}",
                backend_name=getattr(backend, "name", "unknown"),
            )

    def _compute_inverse_matrices(self) -> None:
        """Compute inverse matrices from confusion matrices for error correction."""
        try:
            # For each state, compute the inverse matrix
            for state_idx in self.confusion_matrices:
                # Create a normalized matrix
                matrix_size = max(self.confusion_matrices[state_idx].keys()) + 1
                matrix = np.zeros((matrix_size, matrix_size))

                # Fill in the matrix with measured probabilities
                for i in range(matrix_size):
                    for j in self.confusion_matrices.get(i, {}):
                        if j < matrix_size:
                            matrix[j, i] = self.confusion_matrices[i].get(j, 0)

                # Ensure each column sums to 1 (normalization)
                for i in range(matrix_size):
                    col_sum = matrix[:, i].sum()
                    if col_sum > 0:
                        matrix[:, i] /= col_sum

                # Compute inverse using pseudo-inverse for stability
                try:
                    inverse = np.linalg.pinv(matrix)
                    self.inverse_matrices[state_idx] = inverse
                except np.linalg.LinAlgError:
                    # Fallback to identity if inversion fails
                    quantum_logger.log_error(
                        component="error_mitigation",
                        error_type="matrix_inversion_error",
                        message=f"Failed to invert confusion matrix for state {state_idx}",
                    )
                    self.inverse_matrices[state_idx] = np.eye(matrix_size)
        except Exception as e:
            quantum_logger.log_error(
                component="error_mitigation",
                error_type="computation_error",
                message=f"Error computing inverse matrices: {str(e)}",
            )

    def mitigate(self, circuit: cirq.Circuit, backend: Any, **kwargs) -> cirq.Circuit:
        """
        Apply readout error mitigation to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            Mitigated circuit (unchanged, mitigation happens in process_results)
        """
        if not self.is_calibrated:
            # Auto-calibrate if needed
            qubits = list(circuit.all_qubits())
            self.calibrate(backend, qubits)

        # The circuit remains unchanged, but we track qubits for result processing
        kwargs["qubits"] = list(circuit.all_qubits())
        kwargs["original_circuit"] = circuit

        return circuit

    def process_results(self, results: Dict, **kwargs) -> Dict:
        """
        Process results using readout error mitigation.

        Args:
            results: Results from running the circuit
            **kwargs: Additional arguments

        Returns:
            Mitigated results
        """
        if not self.is_calibrated or not self.inverse_matrices:
            quantum_logger.log_error(
                component="error_mitigation",
                error_type="uncalibrated_error",
                message="Attempted to process results without calibration",
            )
            return results

        # Extract qubits from kwargs
        qubits = kwargs.get("qubits", [])
        if not qubits:
            return results

        # Apply inverse matrices to correct readout errors
        mitigated_results = {}

        for key, vals in results.items():
            # Original measured counts
            measured_counts = {}
            for result in vals:
                # Convert result to integer
                result_int = int("".join(map(str, result)), 2)
                measured_counts[result_int] = measured_counts.get(result_int, 0) + 1

            # Normalize to get probabilities
            total_counts = sum(measured_counts.values())
            measured_probs = {k: v / total_counts for k, v in measured_counts.items()}

            # Apply correction using inverse matrices
            max_state = max(measured_counts.keys(), default=0)
            matrix_size = max(max_state + 1, len(self.inverse_matrices))

            # Create vector of measured probabilities
            meas_vec = np.zeros(matrix_size)
            for state, prob in measured_probs.items():
                if state < matrix_size:
                    meas_vec[state] = prob

            # Apply correction
            if matrix_size in self.inverse_matrices:
                inverse = self.inverse_matrices[matrix_size]
                corrected_vec = np.dot(inverse, meas_vec)

                # Ensure non-negative probabilities
                corrected_vec = np.maximum(corrected_vec, 0)

                # Renormalize
                sum_probs = np.sum(corrected_vec)
                if sum_probs > 0:
                    corrected_vec /= sum_probs

                # Convert back to counts
                corrected_counts = {
                    i: int(p * total_counts + 0.5)
                    for i, p in enumerate(corrected_vec)
                    if p > 0
                }
            else:
                # Fallback if no exact matrix size match
                # Find closest matrix size
                closest_size = min(
                    self.inverse_matrices.keys(), key=lambda x: abs(x - matrix_size)
                )
                inverse = self.inverse_matrices[closest_size]

                # Resize vectors if needed
                if len(meas_vec) > len(inverse):
                    meas_vec = meas_vec[: len(inverse)]
                elif len(meas_vec) < len(inverse):
                    meas_vec = np.pad(meas_vec, (0, len(inverse) - len(meas_vec)))

                corrected_vec = np.dot(inverse, meas_vec)
                corrected_vec = np.maximum(corrected_vec, 0)
                corrected_vec /= (
                    np.sum(corrected_vec) if np.sum(corrected_vec) > 0 else 1
                )
                corrected_counts = {
                    i: int(p * total_counts + 0.5)
                    for i, p in enumerate(corrected_vec)
                    if p > 0
                }

            # Convert corrected counts back to measurement format
            mitigated_vals = np.copy(vals)

            # Distribute the corrected counts
            count_idx = 0
            for state, count in sorted(corrected_counts.items()):
                binary = format(state, f"0{len(qubits)}b")
                binary_array = np.array([int(b) for b in binary])

                # Fill in the mitigated values
                for i in range(count):
                    if count_idx < len(mitigated_vals):
                        if len(binary_array) < len(mitigated_vals[count_idx]):
                            # Pad if needed
                            binary_array = np.pad(
                                binary_array,
                                (0, len(mitigated_vals[count_idx]) - len(binary_array)),
                            )
                        mitigated_vals[count_idx] = binary_array[
                            : len(mitigated_vals[count_idx])
                        ]
                        count_idx += 1

            mitigated_results[key] = mitigated_vals

        return mitigated_results


class DynamicalDecouplingMitigation(ErrorMitigationStrategy):
    """Dynamical decoupling for mitigating coherent errors during idle times."""

    def __init__(self, sequence_type: str = "XY4"):
        """
        Initialize dynamical decoupling error mitigation.

        Args:
            sequence_type: Type of DD sequence to use (XY4, CPMG, etc.)
        """
        super().__init__(name="Dynamical Decoupling")
        self.sequence_type = sequence_type
        self.is_calibrated = True  # No calibration needed for basic DD

    def calibrate(self, backend: Any, qubits: List[cirq.Qid] = None, **kwargs) -> bool:
        """
        DD doesn't require calibration for basic implementation.

        Args:
            backend: Quantum backend to use
            qubits: List of qubits to calibrate
            **kwargs: Additional arguments

        Returns:
            True (always calibrated)
        """
        self.is_calibrated = True
        self.last_calibration_time = time.time()
        return True

    def mitigate(self, circuit: cirq.Circuit, backend: Any, **kwargs) -> cirq.Circuit:
        """
        Apply dynamical decoupling to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            Mitigated circuit with DD sequences inserted
        """
        operation_id = quantum_logger.start_operation(f"mitigate_{self.name}")

        try:
            # Create a copy of the circuit
            mitigated_circuit = cirq.Circuit()

            # Get all qubits in the circuit
            qubits = list(circuit.all_qubits())

            # Track the last operation time for each qubit
            last_op_moment = dict.fromkeys(qubits, 0)

            # Process circuit moments
            for moment_idx, moment in enumerate(circuit):
                # Get qubits that have operations in this moment
                active_qubits = set(q for op in moment for q in op.qubits)

                # For each qubit, check if it's been idle
                for qubit in qubits:
                    if qubit in active_qubits:
                        # Update last operation time
                        last_op_moment[qubit] = moment_idx
                    else:
                        # Check if qubit has been idle for a while
                        idle_duration = moment_idx - last_op_moment[qubit]

                        # Apply DD sequence if idle for 2+ moments
                        if idle_duration >= 2:
                            # Insert DD sequence before the current moment
                            if self.sequence_type == "XY4":
                                # XY4 sequence: X - Y - X - Y
                                mitigated_circuit.append(cirq.X(qubit))
                                mitigated_circuit.append(cirq.Y(qubit))
                                mitigated_circuit.append(cirq.X(qubit))
                                mitigated_circuit.append(cirq.Y(qubit))
                            elif self.sequence_type == "CPMG":
                                # CPMG sequence: X - X
                                mitigated_circuit.append(cirq.X(qubit))
                                mitigated_circuit.append(cirq.X(qubit))
                            # Update last operation time
                            last_op_moment[qubit] = moment_idx

                # Add the original moment
                mitigated_circuit.append(moment)

            # Log success
            quantum_logger.end_operation(
                operation_id,
                {
                    "success": True,
                    "n_qubits": len(qubits),
                    "sequence_type": self.sequence_type,
                    "original_circuit_depth": len(circuit),
                    "mitigated_circuit_depth": len(mitigated_circuit),
                },
            )

            return mitigated_circuit

        except Exception as e:
            # Log failure
            quantum_logger.log_error(
                component="error_mitigation",
                error_type="dd_mitigation_error",
                message=f"Dynamical decoupling mitigation failed: {str(e)}",
                circuit_info={
                    "n_qubits": len(list(circuit.all_qubits())),
                    "depth": len(circuit),
                },
            )

            quantum_logger.end_operation(
                operation_id, {"success": False, "error": str(e)}
            )

            # Return original circuit if mitigation fails
            return circuit

    def process_results(self, results: Dict, **kwargs) -> Dict:
        """
        Process results after dynamical decoupling.

        Args:
            results: Results from running the circuit
            **kwargs: Additional arguments

        Returns:
            Processed results (unchanged for DD)
        """
        # No post-processing needed for dynamical decoupling
        return results


class ZeroNoiseExtrapolation(ErrorMitigationStrategy):
    """Zero-noise extrapolation for error mitigation."""

    def __init__(
        self, scale_factors: List[float] = None, extrapolation_method: str = "linear"
    ):
        """
        Initialize zero-noise extrapolation.

        Args:
            scale_factors: Noise scale factors to use
            extrapolation_method: Method for extrapolation
        """
        super().__init__(name="Zero Noise Extrapolation")
        self.scale_factors = scale_factors or [1.0, 2.0, 3.0]
        self.extrapolation_method = extrapolation_method
        self.is_calibrated = True  # No explicit calibration needed

    def calibrate(self, backend: Any, qubits: List[cirq.Qid] = None, **kwargs) -> bool:
        """
        ZNE doesn't require explicit calibration.

        Args:
            backend: Quantum backend to use
            qubits: List of qubits to calibrate
            **kwargs: Additional arguments

        Returns:
            True (always calibrated)
        """
        self.is_calibrated = True
        self.last_calibration_time = time.time()
        return True

    def mitigate(
        self, circuit: cirq.Circuit, backend: Any, **kwargs
    ) -> List[cirq.Circuit]:
        """
        Apply ZNE to a circuit by generating scaled noise circuits.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            List of circuits with different noise scaling
        """
        operation_id = quantum_logger.start_operation(f"mitigate_{self.name}")
        scaled_circuits = []

        try:
            # Store original circuit as first scale factor (1.0)
            scaled_circuits.append(circuit)

            # Generate circuits for additional scale factors
            for scale in self.scale_factors[1:]:
                # Create a scaled circuit
                scaled_circuit = self._scale_noise(circuit, scale)
                scaled_circuits.append(scaled_circuit)

            # Store scale factors for processing results
            kwargs["scale_factors"] = self.scale_factors
            kwargs["extrapolation_method"] = self.extrapolation_method

            # Log success
            quantum_logger.end_operation(
                operation_id,
                {
                    "success": True,
                    "scale_factors": self.scale_factors,
                    "extrapolation_method": self.extrapolation_method,
                    "n_circuits": len(scaled_circuits),
                },
            )

            return scaled_circuits

        except Exception as e:
            # Log failure
            quantum_logger.log_error(
                component="error_mitigation",
                error_type="zne_mitigation_error",
                message=f"Zero noise extrapolation failed: {str(e)}",
                circuit_info={
                    "n_qubits": len(list(circuit.all_qubits())),
                    "depth": len(circuit),
                },
            )

            quantum_logger.end_operation(
                operation_id, {"success": False, "error": str(e)}
            )

            # Return original circuit if scaling fails
            return [circuit]

    def _scale_noise(self, circuit: cirq.Circuit, scale_factor: float) -> cirq.Circuit:
        """
        Scale the noise in a circuit by a given factor.

        Args:
            circuit: Original circuit
            scale_factor: Factor to scale the noise by

        Returns:
            Circuit with scaled noise
        """
        if scale_factor == 1.0:
            return circuit

        # Method 1: Gate insertion
        # For each gate, insert identity operations to extend duration
        scaled_circuit = cirq.Circuit()

        for moment in circuit:
            scaled_circuit.append(moment)

            # For scale factor > 1, insert additional operations
            if scale_factor > 1:
                # Insert identity-equivalent operations
                n_extras = int(scale_factor - 1)
                for _ in range(n_extras):
                    identity_moment = cirq.Moment()

                    for op in moment:
                        # Skip measurement gates
                        if isinstance(op.gate, cirq.MeasurementGate):
                            continue

                        # Add identity equivalent (X^2 = I)
                        for qubit in op.qubits:
                            identity_moment = identity_moment.with_operation(
                                cirq.X(qubit) ** 0  # Identity operation
                            )

                    if len(identity_moment) > 0:
                        scaled_circuit.append(identity_moment)

        return scaled_circuit

    def process_results(self, results: List[Dict], **kwargs) -> Dict:
        """
        Process results using zero-noise extrapolation.

        Args:
            results: List of results from different noise scales
            **kwargs: Additional arguments

        Returns:
            Extrapolated results
        """
        if not isinstance(results, list) or len(results) != len(self.scale_factors):
            quantum_logger.log_error(
                component="error_mitigation",
                error_type="invalid_results",
                message=f"Expected {len(self.scale_factors)} results, got {len(results) if isinstance(results, list) else 'non-list'}",
            )
            return results[0] if isinstance(results, list) and results else {}

        # Get scale factors
        scale_factors = kwargs.get("scale_factors", self.scale_factors)
        extrapolation_method = kwargs.get(
            "extrapolation_method", self.extrapolation_method
        )

        # Extract expectation values for each scale factor
        expectation_values = []

        # Handle different result formats
        for result in results:
            if isinstance(result, dict) and "result" in result:
                # Convert bit strings to expectation value (+1 for 0, -1 for 1)
                measurements = result["result"]
                exp_val = 0.0
                for shot in measurements:
                    # Compute parity
                    parity = (-1) ** sum(shot)
                    exp_val += parity
                exp_val /= len(measurements)
                expectation_values.append(exp_val)
            elif isinstance(result, (int, float)):
                # Direct expectation value
                expectation_values.append(result)
            elif hasattr(result, "expectation") and callable(result.expectation):
                # Result object with expectation method
                expectation_values.append(result.expectation())

        # Perform extrapolation
        if extrapolation_method == "linear":
            # Linear extrapolation to zero noise
            if len(expectation_values) >= 2:
                # Use numpy for least squares fit
                z = np.polyfit(scale_factors, expectation_values, 1)
                # Extrapolate to zero noise
                zero_noise_value = np.polyval(z, 0.0)

                # Create result dictionary
                extrapolated_result = {
                    "expectation": zero_noise_value,
                    "raw_values": dict(zip(scale_factors, expectation_values)),
                    "extrapolation_method": extrapolation_method,
                    "fit_parameters": z.tolist(),
                }

                return {"result": extrapolated_result}
            else:
                return results[0]
        elif extrapolation_method == "exponential":
            # Exponential extrapolation
            if len(expectation_values) >= 2:
                # Take log of absolute values for exponential fit
                log_vals = np.log(np.abs(expectation_values))
                z = np.polyfit(scale_factors, log_vals, 1)
                # Extrapolate to zero noise and convert back from log
                zero_noise_log = np.polyval(z, 0.0)
                zero_noise_value = np.exp(zero_noise_log)
                # Restore sign
                if expectation_values[0] < 0:
                    zero_noise_value = -zero_noise_value

                extrapolated_result = {
                    "expectation": zero_noise_value,
                    "raw_values": dict(zip(scale_factors, expectation_values)),
                    "extrapolation_method": extrapolation_method,
                    "fit_parameters": z.tolist(),
                }

                return {"result": extrapolated_result}
            else:
                return results[0]
        else:
            # Unknown method, return first result
            return results[0]


class ErrorMitigationFactory:
    """Factory for creating error mitigation strategies."""

    @staticmethod
    def create(strategy_type: str, **kwargs) -> ErrorMitigationStrategy:
        """
        Create an error mitigation strategy.

        Args:
            strategy_type: Type of strategy to create
            **kwargs: Additional arguments for the strategy

        Returns:
            Error mitigation strategy instance
        """
        if strategy_type == "readout":
            return EnhancedReadoutErrorMitigation(**kwargs)
        elif strategy_type == "dynamical_decoupling":
            return DynamicalDecouplingMitigation(**kwargs)
        elif strategy_type == "zne":
            return ZeroNoiseExtrapolation(**kwargs)
        else:
            raise ValueError(f"Unknown error mitigation strategy: {strategy_type}")


# Default instance for simple usage
default_error_mitigation = EnhancedReadoutErrorMitigation()


# Helper functions for easy use


def mitigate_circuit(
    circuit: cirq.Circuit, backend: Any, strategy_type: str = "readout", **kwargs
) -> Tuple[cirq.Circuit, Callable]:
    """
    Apply error mitigation to a circuit.

    Args:
        circuit: Circuit to mitigate
        backend: Quantum backend to use
        strategy_type: Type of mitigation strategy to use
        **kwargs: Additional arguments for the strategy

    Returns:
        Tuple of (mitigated circuit, results processor function)
    """
    # Create the strategy
    strategy = ErrorMitigationFactory.create(strategy_type, **kwargs)

    # Apply mitigation
    mitigated_circuit = strategy.mitigate(circuit, backend, **kwargs)

    # Create processor function
    def process_results(results):
        return strategy.process_results(results, **kwargs)

    return mitigated_circuit, process_results


# Export for easy imports
__all__ = [
    "CalibrationMethod",
    "ErrorMitigationStrategy",
    "EnhancedReadoutErrorMitigation",
    "DynamicalDecouplingMitigation",
    "ZeroNoiseExtrapolation",
    "ErrorMitigationFactory",
    "default_error_mitigation",
    "mitigate_circuit",
]
