#!/usr/bin/env python3
"""
Quantum Hardware Integration Module

This module provides interfaces to connect the QNN with real quantum hardware
through various providers, as well as error mitigation techniques to improve results.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import cirq
import numpy as np

# Try to import provider-specific modules
try:
    import cirq_google

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import qiskit
    from qiskit import IBMQ

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


# Base class for quantum hardware backends
class QuantumBackend:
    """Base class for quantum hardware backends."""

    def __init__(self, name: str = "base"):
        """Initialize the quantum backend."""
        self.name = name
        self.supports_noise_simulation = False
        self.is_simulator = True
        self.max_qubits = float("inf")
        self.available_gates = []

    def run_circuit(self, circuit: cirq.Circuit, repetitions: int = 1000) -> np.ndarray:
        """
        Run a quantum circuit on the backend.

        Args:
            circuit: Cirq circuit to run
            repetitions: Number of repetitions (shots)

        Returns:
            Measurement results
        """
        raise NotImplementedError("Subclasses must implement run_circuit method")

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Get the statevector output of a circuit (simulator-only).

        Args:
            circuit: Cirq circuit to run

        Returns:
            Final statevector
        """
        raise NotImplementedError("Only simulators support statevector output")

    def supports_statevector(self) -> bool:
        """Check if backend supports statevector output."""
        return False

    def get_device_properties(self) -> Dict:
        """Get properties of the device."""
        return {
            "name": self.name,
            "is_simulator": self.is_simulator,
            "max_qubits": self.max_qubits,
            "available_gates": self.available_gates,
            "supports_noise_simulation": self.supports_noise_simulation,
        }

    def __str__(self) -> str:
        """String representation of the backend."""
        return f"{self.name} ({'Simulator' if self.is_simulator else 'Hardware'})"


class CirqSimulatorBackend(QuantumBackend):
    """Cirq simulator backend."""

    def __init__(self, noise_model=None):
        """
        Initialize Cirq simulator backend.

        Args:
            noise_model: Optional noise model for noisy simulation
        """
        super().__init__(name="Cirq Simulator")
        self.simulator = cirq.Simulator()
        self.noise_model = noise_model
        self.supports_noise_simulation = noise_model is not None
        self.is_simulator = True
        self.max_qubits = 30  # Practical limit
        self.available_gates = [
            "X",
            "Y",
            "Z",
            "H",
            "CZ",
            "CNOT",
            "SWAP",
            "T",
            "S",
            "Rx",
            "Ry",
            "Rz",
        ]

    def run_circuit(self, circuit: cirq.Circuit, repetitions: int = 1000) -> Dict:
        """
        Run a quantum circuit on the Cirq simulator.

        Args:
            circuit: Cirq circuit to run
            repetitions: Number of repetitions (shots)

        Returns:
            Dictionary with measurement results
        """
        # Apply noise model if provided
        if self.noise_model:
            noisy_circuit = cirq.Circuit()
            for moment in circuit:
                noisy_circuit.append(moment)
                noisy_circuit.append(self.noise_model.on_each(moment.qubits))
            circuit = noisy_circuit

        # Check if circuit has measurements
        has_measurements = any(
            isinstance(op.gate, cirq.MeasurementGate) for op in circuit.all_operations()
        )

        if not has_measurements:
            # Add measurements of all qubits if none exist
            qubits = sorted(circuit.all_qubits())
            circuit = circuit.copy()
            circuit.append(cirq.measure(*qubits, key="result"))

        # Run the simulation
        result = self.simulator.run(circuit, repetitions=repetitions)
        return {key: np.array(result.measurements[key]) for key in result.measurements}

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Get the statevector output of a circuit.

        Args:
            circuit: Cirq circuit to run

        Returns:
            Final statevector
        """
        result = self.simulator.simulate(circuit)
        return result.final_state_vector

    def supports_statevector(self) -> bool:
        """Check if backend supports statevector output."""
        return True


class CirqGoogleBackend(QuantumBackend):
    """Google Quantum hardware backend using Cirq."""

    def __init__(self, device_name: str = "rainbow"):
        """
        Initialize Google Quantum hardware backend.

        Args:
            device_name: Name of the Google Quantum device
        """
        super().__init__(name=f"Google {device_name}")

        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "cirq_google package is required for Google Quantum hardware"
            )

        if device_name == "rainbow":
            self.device = cirq_google.devices.rainbow.get_rainbow_device()
        elif device_name == "weber":
            self.device = cirq_google.devices.weber.get_weber_device()
        elif device_name == "simulator":
            self.device = cirq_google.devices.foxtail.get_foxtail_device()
            self.name = "Google Quantum Simulator"
            self.is_simulator = True
        else:
            raise ValueError(f"Unknown Google device: {device_name}")

        self.is_simulator = device_name == "simulator"
        self.sampler = cirq_google.get_engine_sampler(
            processor_id=device_name, gate_set=cirq_google.FSIM_GATESET
        )

        # Set device properties
        self.max_qubits = len(self.device.qubits)
        self.available_gates = ["X", "Y", "Z", "H", "CZ", "SQRT_ISWAP"]

    def run_circuit(self, circuit: cirq.Circuit, repetitions: int = 1000) -> Dict:
        """
        Run a quantum circuit on Google hardware.

        Args:
            circuit: Cirq circuit to run
            repetitions: Number of repetitions (shots)

        Returns:
            Dictionary with measurement results
        """
        # Validate circuit against device constraints
        if not self.is_simulator:
            cirq.validate_circuit(circuit, self.device)

        # Check if circuit has measurements
        has_measurements = any(
            isinstance(op.gate, cirq.MeasurementGate) for op in circuit.all_operations()
        )

        if not has_measurements:
            # Add measurements of all qubits if none exist
            qubits = sorted(circuit.all_qubits())
            circuit = circuit.copy()
            circuit.append(cirq.measure(*qubits, key="result"))

        # Run on hardware
        result = self.sampler.run(circuit, repetitions=repetitions)
        return {key: np.array(result.measurements[key]) for key in result.measurements}

    def supports_statevector(self) -> bool:
        """Check if backend supports statevector output."""
        return False


class QiskitIBMBackend(QuantumBackend):
    """IBM Quantum hardware backend using Qiskit."""

    def __init__(
        self, device_name: str = "ibmq_qasm_simulator", token: Optional[str] = None
    ):
        """
        Initialize IBM Quantum hardware backend.

        Args:
            device_name: Name of the IBM Quantum device
            token: IBM Quantum Experience token
        """
        super().__init__(name=f"IBM {device_name}")

        if not QISKIT_AVAILABLE:
            raise ImportError("qiskit package is required for IBM Quantum hardware")

        # Load IBMQ account if token provided
        if token:
            IBMQ.save_account(token, overwrite=True)

        try:
            IBMQ.load_account()
            provider = IBMQ.get_provider()
            self.backend = provider.get_backend(device_name)
        except:
            # Fallback to simulator if no account or connection issues
            from qiskit import Aer

            self.backend = Aer.get_backend("qasm_simulator")
            self.name = "IBM QASM Simulator (local)"

        self.is_simulator = "simulator" in device_name.lower()

        # Set device properties
        if hasattr(self.backend, "configuration"):
            self.max_qubits = self.backend.configuration().n_qubits
            self.available_gates = self.backend.configuration().basis_gates
        else:
            self.max_qubits = 5
            self.available_gates = ["x", "y", "z", "h", "cx"]

    def _cirq_to_qiskit(self, circuit: cirq.Circuit) -> "qiskit.QuantumCircuit":
        """Convert Cirq circuit to Qiskit circuit."""
        from qiskit import QuantumCircuit

        # Get all qubits in the circuit
        cirq_qubits = sorted(circuit.all_qubits())
        n_qubits = len(cirq_qubits)

        # Create Qiskit circuit
        qiskit_circuit = QuantumCircuit(n_qubits, n_qubits)

        # Map Cirq qubits to Qiskit qubits
        qubit_map = {q: i for i, q in enumerate(cirq_qubits)}

        # Convert operations
        for moment in circuit:
            for op in moment:
                # Get qubits
                qiskit_qubits = [qubit_map[q] for q in op.qubits]

                # Convert gate
                if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.x(qiskit_qubits[0])
                elif isinstance(op.gate, cirq.YPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.y(qiskit_qubits[0])
                elif isinstance(op.gate, cirq.ZPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.z(qiskit_qubits[0])
                elif isinstance(op.gate, cirq.HPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.h(qiskit_qubits[0])
                elif isinstance(op.gate, cirq.CZPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.cz(qiskit_qubits[0], qiskit_qubits[1])
                elif isinstance(op.gate, cirq.CXPowGate) and op.gate.exponent == 1:
                    qiskit_circuit.cx(qiskit_qubits[0], qiskit_qubits[1])
                elif isinstance(op.gate, cirq.MeasurementGate):
                    for i, q in enumerate(qiskit_qubits):
                        qiskit_circuit.measure(q, q)
                elif isinstance(op.gate, cirq.XPowGate):
                    # Rx gate with parameter
                    angle = op.gate.exponent * np.pi
                    qiskit_circuit.rx(angle, qiskit_qubits[0])
                elif isinstance(op.gate, cirq.YPowGate):
                    # Ry gate with parameter
                    angle = op.gate.exponent * np.pi
                    qiskit_circuit.ry(angle, qiskit_qubits[0])
                elif isinstance(op.gate, cirq.ZPowGate):
                    # Rz gate with parameter
                    angle = op.gate.exponent * np.pi
                    qiskit_circuit.rz(angle, qiskit_qubits[0])
                else:
                    raise ValueError(f"Unsupported gate: {op.gate}")

        return qiskit_circuit

    def run_circuit(self, circuit: cirq.Circuit, repetitions: int = 1000) -> Dict:
        """
        Run a quantum circuit on IBM hardware.

        Args:
            circuit: Cirq circuit to run
            repetitions: Number of repetitions (shots)

        Returns:
            Dictionary with measurement results
        """
        # Convert circuit to Qiskit
        qiskit_circuit = self._cirq_to_qiskit(circuit)

        # Run on backend
        job = qiskit.execute(qiskit_circuit, self.backend, shots=repetitions)
        result = job.result()
        counts = result.get_counts()

        # Convert results back to Cirq format
        # This is simplified and might need adaptation for specific measurement keys
        measurements = {}
        n_qubits = qiskit_circuit.num_qubits
        for bitstring, count in counts.items():
            bitarray = np.array([int(b) for b in bitstring.zfill(n_qubits)])
            if "result" not in measurements:
                measurements["result"] = np.zeros(
                    (repetitions, n_qubits), dtype=np.int8
                )

            # Fill the measurements array with the results
            # This is approximate as we're distributing counts
            start_idx = sum(
                measurements["result"].shape[0] for k in measurements if k != "result"
            )
            end_idx = start_idx + count
            if end_idx > repetitions:
                end_idx = repetitions

            for i in range(start_idx, end_idx):
                if i < repetitions:
                    measurements["result"][i] = bitarray

        return measurements

    def supports_statevector(self) -> bool:
        """Check if backend supports statevector output."""
        return self.is_simulator and "statevector" in self.backend.name()

    def get_statevector(self, circuit: cirq.Circuit) -> np.ndarray:
        """
        Get the statevector output of a circuit (simulator-only).

        Args:
            circuit: Cirq circuit to run

        Returns:
            Final statevector
        """
        if not self.supports_statevector():
            raise ValueError("This backend does not support statevector output")

        # Convert circuit to Qiskit
        qiskit_circuit = self._cirq_to_qiskit(circuit)

        # Run on statevector simulator
        from qiskit import Aer

        backend = Aer.get_backend("statevector_simulator")
        job = qiskit.execute(qiskit_circuit, backend)
        result = job.result()

        # Get statevector
        statevector = result.get_statevector()
        return np.array(statevector)


# Define error mitigation strategies


class ErrorMitigationStrategy:
    """Base class for error mitigation strategies."""

    def __init__(self, name: str = "base"):
        """Initialize the error mitigation strategy."""
        self.name = name

    def mitigate(
        self, circuit: cirq.Circuit, backend: QuantumBackend, **kwargs
    ) -> cirq.Circuit:
        """
        Apply error mitigation to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            Mitigated circuit
        """
        raise NotImplementedError("Subclasses must implement mitigate method")

    def process_results(self, results: Dict, **kwargs) -> Dict:
        """
        Process results after running the mitigated circuit.

        Args:
            results: Results from running the circuit
            **kwargs: Additional arguments

        Returns:
            Processed results
        """
        return results

    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name} Error Mitigation"


class ReadoutErrorMitigation(ErrorMitigationStrategy):
    """Readout error mitigation using calibration circuits."""

    def __init__(self):
        """Initialize readout error mitigation."""
        super().__init__(name="Readout Error")
        self.calibration_matrices = {}

    def calibrate(
        self, qubits: List[cirq.Qid], backend: QuantumBackend, shots: int = 1000
    ) -> None:
        """
        Calibrate readout errors for the given qubits.

        Args:
            qubits: List of qubits to calibrate
            backend: Quantum backend to use
            shots: Number of shots for calibration
        """
        n_qubits = len(qubits)

        # Create calibration circuits
        # For each computational basis state, prepare and measure
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
            results = backend.run_circuit(circuit, repetitions=shots)

            # Process results to build calibration matrix
            # This is a simplified version
            for result_key, result_vals in results.items():
                counts = {}
                for result in result_vals:
                    # Convert result to integer
                    result_int = int("".join(map(str, result)), 2)
                    counts[result_int] = counts.get(result_int, 0) + 1

                # Normalize counts
                for k in counts:
                    counts[k] /= shots

                # Store in calibration matrices
                if state_idx not in self.calibration_matrices:
                    self.calibration_matrices[state_idx] = {}

                self.calibration_matrices[state_idx].update(counts)

    def mitigate(
        self, circuit: cirq.Circuit, backend: QuantumBackend, **kwargs
    ) -> cirq.Circuit:
        """
        Apply readout error mitigation to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            Mitigated circuit (unchanged, mitigation happens in process_results)
        """
        qubits = list(circuit.all_qubits())
        if not self.calibration_matrices:
            self.calibrate(qubits, backend)

        # The circuit remains unchanged, but we need the qubits order
        kwargs["qubits"] = qubits
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
        if not self.calibration_matrices:
            return results

        # Extract qubits from kwargs
        qubits = kwargs.get("qubits", [])
        if not qubits:
            return results

        # Apply inverse of calibration matrix to correct for readout errors
        # This is a simplified implementation
        mitigated_results = {}

        for key, vals in results.items():
            mitigated_vals = np.copy(vals)

            # Apply correction for each shot
            for i, result in enumerate(vals):
                # Convert result to integer
                result_int = int("".join(map(str, result)), 2)

                # Use calibration matrix to correct
                # This is simplified and would need more sophisticated matrix inversion in practice
                if result_int in self.calibration_matrices:
                    # Find most likely original state
                    most_likely = max(
                        self.calibration_matrices,
                        key=lambda x: self.calibration_matrices[x].get(result_int, 0),
                    )

                    # Convert back to bit array
                    corrected = np.array(
                        [int(b) for b in format(most_likely, f"0{len(qubits)}b")]
                    )
                    mitigated_vals[i] = corrected

            mitigated_results[key] = mitigated_vals

        return mitigated_results


class CircuitKnittingStrategy(ErrorMitigationStrategy):
    """Circuit knitting to decompose large circuits into smaller ones."""

    def __init__(self):
        """Initialize circuit knitting strategy."""
        super().__init__(name="Circuit Knitting")
        self.subcircuits = []

    def mitigate(
        self, circuit: cirq.Circuit, backend: QuantumBackend, **kwargs
    ) -> List[cirq.Circuit]:
        """
        Apply circuit knitting to a circuit.

        Args:
            circuit: Circuit to mitigate
            backend: Quantum backend to use
            **kwargs: Additional arguments

        Returns:
            List of subcircuits
        """
        # Get max qubits supported by the backend
        max_qubits = backend.max_qubits

        # Get all qubits in the circuit
        all_qubits = sorted(circuit.all_qubits())
        n_qubits = len(all_qubits)

        # If circuit fits on device, return as is
        if n_qubits <= max_qubits:
            return [circuit]

        # Split into subcircuits that fit on the device
        # This is a simplified approach - real circuit knitting is more complex
        self.subcircuits = []
        qubit_groups = [
            all_qubits[i : i + max_qubits] for i in range(0, n_qubits, max_qubits)
        ]

        for group in qubit_groups:
            # Create subcircuit for this group
            subcircuit = cirq.Circuit()

            # Add operations that only involve qubits in this group
            for moment in circuit:
                subcircuit_moment = cirq.Moment()
                for op in moment:
                    if all(q in group for q in op.qubits):
                        subcircuit_moment = subcircuit_moment.with_operation(op)

                if len(subcircuit_moment) > 0:
                    subcircuit.append(subcircuit_moment)

            self.subcircuits.append(subcircuit)

        # Store qubit mapping for result processing
        self.qubit_mapping = {
            q: (i, j)
            for i, group in enumerate(qubit_groups)
            for j, q in enumerate(group)
        }

        kwargs["qubit_mapping"] = self.qubit_mapping
        return self.subcircuits

    def process_results(self, results: List[Dict], **kwargs) -> Dict:
        """
        Process results from subcircuits.

        Args:
            results: List of results from running the subcircuits
            **kwargs: Additional arguments

        Returns:
            Combined results
        """
        # This is a simplified version of combining results
        # Real circuit knitting would use more sophisticated techniques

        qubit_mapping = kwargs.get("qubit_mapping", {})
        if not qubit_mapping or not self.subcircuits:
            return results[0] if results else {}

        # Combine results from subcircuits
        combined_results = {}

        # For each measurement key, combine results
        # This is a very simplified approach
        for subcircuit_idx, subcircuit_results in enumerate(results):
            for key, vals in subcircuit_results.items():
                if key not in combined_results:
                    # Initialize with empty results
                    max_idx = max(
                        i for i, _ in qubit_mapping.values() if i == subcircuit_idx
                    )
                    n_qubits = max_idx + 1
                    n_shots = vals.shape[0]
                    combined_results[key] = np.zeros((n_shots, n_qubits), dtype=np.int8)

                # Fill in results for this subcircuit
                for q, (sc_idx, q_idx) in qubit_mapping.items():
                    if sc_idx == subcircuit_idx and key in subcircuit_results:
                        combined_results[key][:, q_idx] = subcircuit_results[key][
                            :, q_idx
                        ]

        return combined_results


# Define a hardware provider class to manage connections


class QuantumHardwareProvider:
    """Provider for quantum hardware backends."""

    def __init__(self):
        """Initialize the quantum hardware provider."""
        self.backends = {}
        self.error_mitigation_strategies = {}

        # Register default backends
        self.register_backend("cirq_simulator", lambda: CirqSimulatorBackend())

        # Register backends that require additional packages
        if GOOGLE_AVAILABLE:
            self.register_backend(
                "google_simulator", lambda: CirqGoogleBackend("simulator")
            )
            self.register_backend(
                "google_rainbow", lambda: CirqGoogleBackend("rainbow")
            )
            self.register_backend("google_weber", lambda: CirqGoogleBackend("weber"))

        if QISKIT_AVAILABLE:
            self.register_backend(
                "ibm_simulator", lambda: QiskitIBMBackend("ibmq_qasm_simulator")
            )
            self.register_backend(
                "ibm_hardware", lambda token: QiskitIBMBackend("ibmq_lima", token)
            )

        # Register error mitigation strategies
        self.register_error_mitigation("readout", lambda: ReadoutErrorMitigation())
        self.register_error_mitigation(
            "circuit_knitting", lambda: CircuitKnittingStrategy()
        )

    def register_backend(self, name: str, constructor: Callable) -> None:
        """
        Register a quantum backend.

        Args:
            name: Name of the backend
            constructor: Function that creates the backend
        """
        self.backends[name] = constructor

    def register_error_mitigation(self, name: str, constructor: Callable) -> None:
        """
        Register an error mitigation strategy.

        Args:
            name: Name of the strategy
            constructor: Function that creates the strategy
        """
        self.error_mitigation_strategies[name] = constructor

    def get_backend(self, name: str, **kwargs) -> QuantumBackend:
        """
        Get a quantum backend by name.

        Args:
            name: Name of the backend
            **kwargs: Additional arguments for the backend constructor

        Returns:
            Quantum backend instance
        """
        if name not in self.backends:
            raise ValueError(f"Unknown backend: {name}")

        # Create the backend
        constructor = self.backends[name]
        return constructor(**kwargs)

    def get_error_mitigation(self, name: str, **kwargs) -> ErrorMitigationStrategy:
        """
        Get an error mitigation strategy by name.

        Args:
            name: Name of the strategy
            **kwargs: Additional arguments for the strategy constructor

        Returns:
            Error mitigation strategy instance
        """
        if name not in self.error_mitigation_strategies:
            raise ValueError(f"Unknown error mitigation strategy: {name}")

        # Create the strategy
        constructor = self.error_mitigation_strategies[name]
        return constructor(**kwargs)

    def list_backends(self) -> List[str]:
        """List available backend names."""
        return list(self.backends.keys())

    def list_error_mitigation_strategies(self) -> List[str]:
        """List available error mitigation strategy names."""
        return list(self.error_mitigation_strategies.keys())


# Define optimized execution functions


def execute_with_hardware(
    circuit: cirq.Circuit,
    backend: QuantumBackend,
    error_mitigation: Optional[ErrorMitigationStrategy] = None,
    repetitions: int = 1000,
    **kwargs,
) -> Dict:
    """
    Execute a circuit on hardware with optional error mitigation.

    Args:
        circuit: Cirq circuit to execute
        backend: Quantum backend to use
        error_mitigation: Optional error mitigation strategy
        repetitions: Number of repetitions (shots)
        **kwargs: Additional arguments

    Returns:
        Execution results
    """
    # Apply error mitigation if provided
    if error_mitigation:
        mitigated_circuit = error_mitigation.mitigate(circuit, backend, **kwargs)

        # Check if we got a list of circuits (e.g. from circuit knitting)
        if isinstance(mitigated_circuit, list):
            subcircuit_results = []
            for subcircuit in mitigated_circuit:
                result = backend.run_circuit(subcircuit, repetitions=repetitions)
                subcircuit_results.append(result)

            # Process results from subcircuits
            return error_mitigation.process_results(subcircuit_results, **kwargs)
        else:
            # Single circuit case
            result = backend.run_circuit(mitigated_circuit, repetitions=repetitions)
            return error_mitigation.process_results(result, **kwargs)
    else:
        # Direct execution without error mitigation
        return backend.run_circuit(circuit, repetitions=repetitions)


def expectation_with_hardware(
    circuit: cirq.Circuit,
    observable: Union[cirq.PauliString, List[Tuple[cirq.PauliString, float]]],
    backend: QuantumBackend,
    error_mitigation: Optional[ErrorMitigationStrategy] = None,
    repetitions: int = 1000,
    **kwargs,
) -> float:
    """
    Calculate expectation value of an observable on hardware.

    Args:
        circuit: Cirq circuit to execute
        observable: Observable (Pauli string or list of terms with coefficients)
        backend: Quantum backend to use
        error_mitigation: Optional error mitigation strategy
        repetitions: Number of repetitions (shots)
        **kwargs: Additional arguments

    Returns:
        Expectation value
    """
    # Check if backend supports statevector
    if backend.supports_statevector():
        # Use statevector for exact expectation value
        if error_mitigation:
            mitigated_circuit = error_mitigation.mitigate(circuit, backend, **kwargs)

            # If circuit knitting returned multiple circuits, we can't use statevector
            if isinstance(mitigated_circuit, list):
                pass  # Fall back to measurement-based approach
            else:
                circuit = mitigated_circuit

        # Get statevector
        statevector = backend.get_statevector(circuit)

        # Calculate expectation value directly
        if isinstance(observable, cirq.PauliString):
            # Convert to a list with coefficient 1.0
            observable = [(observable, 1.0)]

        # Sum up contributions from each term
        expectation = 0.0
        for pauli_string, coeff in observable:
            # Create circuit for the Pauli string
            pauli_circuit = cirq.Circuit(pauli_string)

            # Apply to statevector
            pauli_statevector = backend.get_statevector(pauli_circuit)

            # Calculate overlap
            overlap = np.abs(np.vdot(statevector, pauli_statevector))
            expectation += coeff * overlap

        return float(expectation.real)

    # Measurement-based approach for hardware or simulators without statevector
    # Convert observable to list of terms if needed
    if isinstance(observable, cirq.PauliString):
        observable = [(observable, 1.0)]

    # For each Pauli string in the observable
    total_expectation = 0.0
    for pauli_string, coeff in observable:
        # Create a copy of the circuit
        meas_circuit = circuit.copy()

        # Add basis rotation gates to measure in the correct basis
        qubits = sorted(circuit.all_qubits())
        for qubit, gate in pauli_string.items():
            if qubit in qubits:
                if gate == cirq.X:
                    # Measure in X basis: H before measurement
                    meas_circuit.append(cirq.H(qubit))
                elif gate == cirq.Y:
                    # Measure in Y basis: S† H before measurement
                    meas_circuit.append(cirq.S(qubit) ** -1)
                    meas_circuit.append(cirq.H(qubit))
                # For Z basis, no change needed

        # Add measurements
        meas_circuit.append(cirq.measure(*qubits, key="obs"))

        # Execute circuit
        results = execute_with_hardware(
            meas_circuit, backend, error_mitigation, repetitions, **kwargs
        )

        # Calculate expectation from measurements
        if "obs" in results:
            # Get measurement results
            measurements = results["obs"]

            # Calculate parity for each shot
            expectation_value = 0.0
            for shot in measurements:
                # For each qubit in the Pauli string, check if it's measured as 1
                parity = 1.0
                for qubit, gate in pauli_string.items():
                    # Find index of the qubit in the sorted list
                    idx = qubits.index(qubit)
                    if idx < len(shot) and shot[idx] == 1:
                        parity *= -1.0

                expectation_value += parity

            # Normalize and multiply by coefficient
            expectation_value /= measurements.shape[0]
            total_expectation += coeff * expectation_value

    return float(total_expectation)
