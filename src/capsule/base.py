"""Base classes for the capsule package."""

import typing as tp
from abc import ABC, abstractmethod

import nannyml as nml
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import validate_call
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

type Input = NDArray[np.float64] | pd.DataFrame
type Output = NDArray[np.float64]


chunker_args = {
    "chunk_size",
    "chunk_number",
    "chunk_period",
    "chunker",
}


@tp.runtime_checkable
class ImplementsPredict(tp.Protocol):
    """Protocol for classes that implement a predict method.

    This protocol defines the interface for models that can make predictions.
    Classes implementing this protocol must define a predict method that takes
    input data and returns predictions.
    """

    def predict(self, X: Input) -> Output:
        """Predict output for the given input.

        Args:
            X: Input data for prediction.

        Returns:
            Predicted output values.
        """


@tp.runtime_checkable
class ImplementsProba(tp.Protocol):
    """Protocol for classes that implement predict and predict_proba methods.

    This protocol extends ImplementsPredict to include probability prediction
    capabilities. Classes implementing this protocol must define both predict
    and predict_proba methods for classification tasks.
    """

    def predict(self, X: Input) -> Output:
        """Predict output for the given input.

        Args:
            X: Input data for prediction.

        Returns:
            Predicted class labels.
        """

    def predict_proba(self, X: Input) -> Output:
        """Predict class probabilities for the given input.

        Args:
            X: Input data for probability prediction.

        Returns:
            Predicted class probabilities.
        """


class BaseCapsule(ABC, BaseEstimator):
    """Base abstract class for all capsule implementations.

    This class provides the common interface and functionality for both
    regression and classification capsules. It handles model wrapping,
    serialization with optional encryption, and defines the basic contract
    that all capsules must follow.

    Note:
        Capsules are immutable wrappers around trained models and cannot
        be fitted after creation.
    """

    drift_calculator_: nml.UnivariateDriftCalculator
    performance_calculator_: nml.PerformanceCalculator

    n_targets_: int
    n_classes_: int
    n_features_: int
    target_index_: int | None

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        model: ImplementsPredict | ImplementsProba,
        X_test: Input,
        y_test: Output,
        **chunk_args,
    ) -> None:
        """Initialize the base capsule with a trained model and test data.

        Args:
            model: A trained model that implements prediction methods.
            X_test: Test input data used for reference during performance estimation.
            y_test: Test target data corresponding to X_test.
            **chunk_args: Additional keyword arguments to pass to the univariate drift
                detector chunker. See
                https://nannyml.readthedocs.io/en/stable/tutorials/chunking.html
                documentation for details.

        Raises:
            ValidationError: If the input data fails validation checks.
        """
        super().__init__()

        check_X_y(
            X_test,
            y_test,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            multi_output=True,
            y_numeric=True,
        )

        self.model_ = model
        self.X_test_ = X_test
        self.y_test_ = y_test

        self.n_features_ = X_test.shape[1]
        self.n_targets_ = 1 if y_test.ndim == 1 else y_test.shape[1]

    def fit(self, X: Input, y: Output, **fit_params: dict) -> "BaseCapsule":
        """Attempt to fit the capsule.

        Capsules wrap pre-trained models and cannot be fitted. This method
        always raises NotImplementedError to maintain the immutable nature
        of capsules.

        Args:
            X: Input data (ignored).
            y: Target data (ignored).
            **fit_params: Additional parameters (ignored).

        Returns:
            Never returns - always raises exception.

        Raises:
            NotImplementedError: Always raised since capsules cannot be fitted.
        """
        raise NotImplementedError("Capsules cannot be fit.")

    @validate_call(config={"arbitrary_types_allowed": True})
    def predict(self, X: Input) -> Output:
        """Generate predictions using the wrapped model.

        Args:
            X: Input data for prediction.

        Returns:
            Model predictions for the input data.
        """
        return self.model_.predict(X)

    @validate_call(config={"arbitrary_types_allowed": True})
    def univariate_drift(self, X: Input):
        """Estimate univariate drift on analysis data.

        Args:
            X: Analysis input data for drift estimation.

        Returns:
            Univariate drift estimation results.
        """
        df = self.format_data(X)
        df = df[[c for c in df.columns if c.startswith("_")]]

        return self.drift_calculator_.calculate(df).filter(period="analysis").to_df()

    @abstractmethod
    def format_data(self, X: Input, y: tp.Optional[Output] = None) -> pd.DataFrame:
        """Formats data to be used in metrics calculations and plots."""
        raise NotImplementedError("Must be implemented in subclasses.")

    @abstractmethod
    def metrics(self, X: Input, metric: str) -> pd.DataFrame:
        """Estimate performance metrics on analysis data.

        This abstract method must be implemented by subclasses to provide
        performance estimation capabilities specific to their task type.

        Args:
            X: Analysis data for performance estimation.
            metric: Metric to use. In classification, this can be 'f1', 'roc_auc',
                'precision', or 'recall'.

        Returns:
            Performance estimation results.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @property
    @abstractmethod
    def plots(self) -> object:
        """Get plots for the capsule."""
        raise NotImplementedError("Must be implemented in subclasses.")

    @validate_call(config={"arbitrary_types_allowed": True})
    def _fit(self, X: Input, **chunk_args) -> None:
        """Fits univariate drift detector and performance tracker on reference data.

        Args:
            X: Reference input data for fitting the drift detector and performance
                tracker.
            **chunk_args: Additional keyword arguments to pass to the univariate drift
                detector.
        """
        df = self.format_data(X)
        df = df[[c for c in df.columns if c.startswith("_")]]

        self.drift_calculator_ = nml.UnivariateDriftCalculator(
            column_names=[c for c in df.columns if c.startswith("_")],
            treat_as_categorical=df.select_dtypes(
                include=["category", "object"]
            ).columns.tolist(),
            timestamp_column_name="__timestamp"
            if "__timestamp" in df.columns
            else None,
            continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
            categorical_methods=["chi2", "jensen_shannon"],
            **chunk_args,
        )

        self.drift_calculator_.fit(df)
