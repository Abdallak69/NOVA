Hardware Execution
==================

This tutorial demonstrates running NOVA on simulators or real hardware with optional error mitigation.

Python API
----------

.. code-block:: python

   from nova.core.qnn_molecular_energy import MolecularQNN
   from nova.hardware.quantum_hardware_interface import CirqSimulatorBackend
   from nova.mitigation.quantum_error_mitigation import ZeroNoiseExtrapolation

   backend = CirqSimulatorBackend(name="cirq_simulator")
   mitigation = ZeroNoiseExtrapolation()

   qnn = MolecularQNN(
       molecule="H2",
       bond_length=0.74,
       ansatz_type="ucc",
       hardware_backend=backend,
       error_mitigation=mitigation,
   )

   results = qnn.train(iterations=50)
   print(results["energy"])  # Energy estimate with mitigation

Tips
----

- Increase ``shots`` to reduce statistical error
- Try gradient-free optimizers on noisy backends
- Use the transpiler to optimize circuits for a target device
