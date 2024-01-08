# fmt: off

"""
Some type definitions for the capsule submodule.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

Inputs = NDArray[np.float64] | pd.DataFrame
Actuals = NDArray[np.float64] | pd.DataFrame
Predictions = NDArray[np.float64]

# fmt: on
