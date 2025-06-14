import os

import pytest

# Skip test if DISPLAY is not set (headless environment)
if not os.environ.get("DISPLAY"):
    pytest.skip("Skipping Qt test - no DISPLAY environment", allow_module_level=True)

import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

print("Matplotlib using:", matplotlib.get_backend())
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Qt Backend")
plt.show()
