# fmt: off

"""
Some type definitions for the capsule submodule.
"""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as imblearn_Pipeline  # type: ignore
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline as sklearn_Pipeline  # type: ignore

__all__ = [
    "Inputs",
    "Actuals",
    "Predictions",
    "Pipeline",
]

Inputs = NDArray[np.float64] | pd.DataFrame
Actuals = NDArray[np.float64] | pd.DataFrame | pd.Series
Predictions = NDArray[np.float64]
Pipeline = sklearn_Pipeline | imblearn_Pipeline

# fmt: on
