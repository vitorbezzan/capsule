"""Base classes for the capsule package."""

import typing as tp

import nannyml as nml
import numpy as np
import pandas as pd
from nannyml.base import Result
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y

Input: tp.TypeAlias = NDArray[np.float64] | pd.DataFrame
Output: tp.TypeAlias = NDArray[np.float64]


_filter_kwargs = {
    "problem_type",
    "y_pred_proba",
    "y_pred",
    "y_true",
    "timestamp_column_name",
    "metrics",
}


class ImplementsPredict(tp.Protocol):
    """Protocol for classes that implement a predict method."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input.

        Args:
            X: Input data.

        Returns:
            Predicted output.

        """


class ImplementsProba(tp.Protocol):
    """Protocol for classes that implements predict and predict_proba methods."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input.

        Args:
            X: Input data.

        Returns:
            Predicted output.

        """

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input.

        Args:
            X: Input data.

        Returns:
            Predicted probabilities.

        """


class BaseCapsule(BaseEstimator):
    """Base class for all capsules."""

    def __init__(
        self,
        model: ImplementsPredict | ImplementsProba,
        X_test: Input,
        y_test: Output,
    ) -> None:
        """Initialize the base capsule with a model and test data.

        Args:
            model: Model implementing prediction.
            X_test: Test input data.
            y_test: Test target data.

        """
        super().__init__()

        self.model_ = model

        check_X_y(
            X_test,
            y_test,
            ensure_all_finite="allow-nan",
            ensure_2d=True,
            multi_output=True,
            y_numeric=True,
        )

        self.n_features_ = X_test.shape[1]
        self.n_targets_ = 1 if y_test.ndim == 1 else y_test.shape[1]

    def fit(
        self, X: Input, y: Output | None = None, **fit_params: dict
    ) -> "BaseCapsule":
        """Capsules cannot be fit.

        Args:
            X: Input data.
            y: Target data.
            **fit_params: Additional fit parameters.

        Raises:
            NotImplementedError: Always raised since capsules cannot be fit.

        """
        raise NotImplementedError("Capsules cannot be fit.")

    def predict(self, X: Input) -> Output:
        """Predict the capsule's output.

        Args:
            X: Input data.

        Returns:
            Predicted output.

        """
        return self.model_.predict(X)


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule for regression tasks."""

    model: ImplementsPredict

    def __init__(self, model: ImplementsPredict, X_test: Input, y_test: Output) -> None:
        """Initialize the regression capsule.

        Args:
            model: Regression model.
            X_test: Test input data.
            y_test: Test target data.

        """
        super().__init__(model, X_test, y_test)


class ClassificationCapsule(BaseCapsule, ClassifierMixin):
    """Capsule for classification tasks."""

    model: ImplementsProba

    def __init__(
        self, model: ImplementsProba, X_test: Input, y_test: Output, **kwargs
    ) -> None:
        """Initialize the classification capsule.

        Args:
            model: Classification model.
            X_test: Test input data.
            y_test: Test target data.

        Raises:
            ValueError: If multi-target classification is attempted.

        """
        super().__init__(model, X_test, y_test)

        if self.n_targets_ != 1:
            raise ValueError(
                "ClassificationCapsule does not support multi-target classification."
            )

        self.n_classes_ = len(np.unique(y_test))
        reference_data = self.get_CBPE_data(X_test, y_test)

        self.estimator_ = nml.CBPE(
            problem_type="classification_multiclass"
            if self.n_classes_ > 2
            else "classification_binary",
            y_pred_proba={i: f"CBPE_class_{i}" for i in range(self.n_classes_)}
            if self.n_classes_ > 2
            else "CBPE_proba",
            y_pred="CBPE_prediction",
            y_true="CBPE_target",
            timestamp_column_name="CBPE_timestamp"
            if "CBPE_timestamp" in reference_data.columns
            else None,
            metrics=["f1", "roc_auc", "precision", "recall"],
            **{k: v for k, v in kwargs.items() if k not in _filter_kwargs},
        )
        self.estimator_.fit(reference_data)

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input.

        Args:
            X: Input data.

        Returns:
            Predicted probabilities.

        """
        return self.model_.predict_proba(X)

    def get_metrics(self, X: Input) -> Result:
        """Estimate performance metrics on the analysis data using CBPE.

        Args:
            X: Input data.

        Returns:
            Estimated performance metrics.

        """
        analysis_data = self.get_CBPE_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "CBPE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'CBPE_timestamp' is required for analysis."
            )

        estimation = self.estimator_.estimate(analysis_data)
        return estimation.filter(period="analysis").to_df()

    def get_CBPE_data(self, X: Input, y: Output | None = None) -> pd.DataFrame:
        """Generate a DataFrame with the correct structure for reference/analysis data.
        If `y` is provided, it will be included in the DataFrame.
        If index type is in a datetime format, it will create a datetime column
            "timestamp" automatically.

        Args:
            X: Input data.
            y: Target data (optional).

        Raises:
            ValueError: If input data does not have the expected number of features.

        Returns:
            DataFrame with reference or analysis data.

        """
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Input data must have {self.n_features_} features, "
                f"but got {X.shape[1]} features."
            )

        if isinstance(X, pd.DataFrame):
            reference_df = pd.DataFrame(X)

            if isinstance(X.index, pd.DatetimeIndex):
                reference_df["CBPE_timestamp"] = X.index

        else:
            reference_df = pd.DataFrame(
                X, columns=[f"CBPE_f_{i}" for i in range(self.n_features_)]
            )

        reference_df["CBPE_prediction"] = self.model_.predict(X)

        if y is not None:
            reference_df["CBPE_target"] = y

        if self.n_classes_ > 2:
            for i in range(self.n_classes_):
                reference_df[f"CBPE_class_{i}"] = self.model_.predict_proba(X)[:, i]
        else:
            reference_df["CBPE_proba"] = self.model_.predict_proba(X)[:, 1]

        return reference_df
