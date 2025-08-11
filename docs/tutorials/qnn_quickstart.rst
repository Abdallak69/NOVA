QNN Quickstart
==============

This tutorial walks through creating and training a QNN for H2 energy estimation.

.. note::
   See the interactive notebooks in ``examples/`` for a runnable version.

Python API
----------

.. code-block:: python

   from nova.core.qnn_molecular_energy import MolecularQNN

   qnn = MolecularQNN(molecule="H2", bond_length=0.74, ansatz_type="hardware_efficient", depth=2)
   results = qnn.train(iterations=100, method="BFGS")
   print(results["energy"])  # Final energy estimate

CLI
---

.. code-block:: bash

   nova train --molecule H2 --bond-length 0.74 --ansatz hardware_efficient --depth 2 --iterations 100

Next steps
----------

- Compare ansatz families with ``nova compare``
- Try different optimizers via the Python API
- Explore hardware execution in the next tutorial
