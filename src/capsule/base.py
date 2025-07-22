"""Base classes for the capsule package."""

import typing as tp

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import assert_all_finite

Input: tp.TypeAlias = NDArray[np.float64] | pd.DataFrame
Output: tp.TypeAlias = NDArray[np.float64]


class _ImplementsPredict(tp.Protocol):
    """Protocol for classes that implement a predict method."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input."""


class _ImplementsProba(tp.Protocol):
    """Protocol for classes that implements predict and predict_proba methods."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input."""

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input."""


class BaseCapsule(BaseEstimator):
    """Base class for all capsules."""

    def __init__(
        self,
        model: _ImplementsPredict | _ImplementsProba,
        X_test: Input,
        y_test: Output,
    ) -> None:
        """Initialize the base capsule with a model and test data."""
        super().__init__()

        self.model = model

        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples.")

        assert_all_finite(X_test, allow_nan=True)

    def fit(
        self, X: Input, y: Output | None = None, **fit_params: dict
    ) -> "BaseCapsule":
        """Capsules cannot be fit."""
        raise NotImplementedError("Capsules cannot be fit.")

    def predict(self, X: Input) -> Output:
        """Predict the capsule's output."""
        return self.model.predict(X)


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule for regression tasks."""

    def __init__(
        self, model: _ImplementsPredict, X_test: Input, y_test: Output
    ) -> None:
        """Initialize the regression capsule."""
        super().__init__(model, X_test, y_test)


class ClassificationCapsule(BaseCapsule, ClassifierMixin):
    """Capsule for classification tasks."""

    def __init__(self, model: _ImplementsProba, X_test: Input, y_test: Output) -> None:
        """Initialize the classification capsule."""
        super().__init__(model, X_test, y_test)

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input."""
        return self.model.predict_proba(X)
