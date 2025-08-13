"""Base classes for the capsule package."""

import typing as tp
from abc import ABC, abstractmethod

import nannyml as nml
import numpy as np
import pandas as pd
from nannyml.base import Result
from numpy.typing import NDArray
from pydantic import NonNegativeInt, validate_call
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import pickle

Input: tp.TypeAlias = NDArray[np.float64] | pd.DataFrame
Output: tp.TypeAlias = NDArray[np.float64]


_filter_kwargs = {
    "problem_type",
    "y_pred_proba",
    "y_pred",
    "y_true",
    "timestamp_column_name",
    "metrics",
    "feature_column_names",
}


@tp.runtime_checkable
class ImplementsPredict(tp.Protocol):
    """Protocol for classes that implement a predict method."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted output
        """


@tp.runtime_checkable
class ImplementsProba(tp.Protocol):
    """Protocol for classes that implements predict and predict_proba methods."""

    def predict(self, X: Input) -> Output:
        """Predict the output for the given input.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted output
        """

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted probabilities
        """


class BaseCapsule(ABC, BaseEstimator):
    """Base class for all capsules."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        model: ImplementsPredict | ImplementsProba,
        X_test: Input,
        y_test: Output,
    ) -> None:
        """Initialize the base capsule with a model and test data.

        **Arguments:**
        - `model` -- Model implementing prediction
        - `X_test` -- Test input data
        - `y_test` -- Test target data
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

        self.X_test_ = X_test
        self.y_test_ = y_test

        self.model_ = model
        self.n_features_ = X_test.shape[1]
        self.n_targets_ = 1 if y_test.ndim == 1 else y_test.shape[1]

    def __getstate__(self) -> dict:
        """Gets the state of the capsule for serialization."""
        if os.getenv("CAPSULE_KEY", None) is not None:
            nonce = os.urandom(12)
            encrypted = AESGCM(os.environ["CAPSULE_KEY"].encode()).encrypt(
                nonce,
                pickle.dumps(self.__dict__),
                None,
            )

            return {"nonce": nonce, "data": encrypted}

        return {"data": pickle.dumps(self.__dict__)}

    def __setstate__(self, state: dict) -> None:
        """Sets the state of the capsule from serialized data."""
        data = state["data"]

        if "nonce" in state:
            key = os.getenv("CAPSULE_KEY", None)
            if key is None:
                raise RuntimeError(
                    "CAPSULE_KEY not set. Cannot unpickle encrypted capsule.",
                )

            data = AESGCM(key.encode()).decrypt(state["nonce"], data, None)

        self.__dict__ = pickle.loads(data)

    def fit(
        self, X: Input, y: Output | None = None, **fit_params: dict
    ) -> "BaseCapsule":
        """Capsules cannot be fit.

        **Arguments:**
        - `X` -- Input data
        - `y` -- Target data
        - `**fit_params` -- Additional fit parameters

        **Raises:**
        - `NotImplementedError` -- Always raised since capsules cannot be fit
        """
        raise NotImplementedError("Capsules cannot be fit.")

    @validate_call(config={"arbitrary_types_allowed": True})
    def predict(self, X: Input) -> Output:
        """Predict the capsule's output.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted output
        """
        return self.model_.predict(X)

    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted probabilities
        """
        raise NotImplementedError("Not implemented for a generic capsule.")

    @abstractmethod
    def get_metrics(self, X: Input) -> Result:
        """Estimate performance metrics on the analysis data."""
        raise NotImplementedError("Must be implemented in subclasses.")


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule for regression tasks."""

    model_: ImplementsPredict

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        model: ImplementsPredict,
        X_test: Input,
        y_test: Output,
        target_index: NonNegativeInt | None = None,
        **kwargs,
    ) -> None:
        """Initialize the regression capsule.

        **Arguments:**
        - `model` -- Regression model
        - `X_test` -- Test input data
        - `y_test` -- Test target data
        - `target_index` -- Index of the target variable in multi-target regression
        - `**kwargs` -- Additional keyword arguments for DLE estimator

        **Raises:**
        - `ValueError` -- If multi-target regression is attempted
        """
        super().__init__(model, X_test, y_test)
        self.target_index_ = target_index

        reference_data = self.get_DLE_data(self.X_test_, self.y_test_)

        timestamp_col = (
            "DLE_timestamp" if "DLE_timestamp" in reference_data.columns else None
        )

        self.estimator_ = nml.DLE(
            feature_column_names=[
                col for col in reference_data.columns if col.startswith("DLE_f_")
            ],
            y_pred="DLE_prediction",
            y_true="DLE_target",
            timestamp_column_name=timestamp_col,
            metrics=["mae", "mape", "mse", "rmse"],
            **{k: v for k, v in kwargs.items() if k not in _filter_kwargs},
        )
        self.estimator_.fit(reference_data)

    def get_metrics(self, X: Input) -> Result:
        """Estimate performance metrics on the analysis data using DLE.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Estimated performance metrics
        """
        analysis_data = self.get_DLE_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "DLE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'DLE_timestamp' is required for analysis."
            )

        estimation = self.estimator_.estimate(analysis_data)
        return estimation.filter(period="analysis").to_df()

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_DLE_data(self, X: Input, y: Output | None = None) -> pd.DataFrame:
        """Generate a DataFrame with the correct structure for reference/analysis data.
        If `y` is provided, it will be included in the DataFrame.
        If index type is in a datetime format, it will create a datetime column
            "timestamp" automatically.

        **Arguments:**
        - `X` -- Input data
        - `y` -- Target data (optional)

        **Raises:**
        - `ValueError` -- If input data does not have the expected number of features

        **Returns:**
        DataFrame with reference or analysis data
        """
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Input data must have {self.n_features_} features, "
                f"but got {X.shape[1]} features."
            )

        reference_df = (
            pd.DataFrame(X)
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(
                X, columns=[f"DLE_f_{i}" for i in range(self.n_features_)]
            )
        )

        if isinstance(X, pd.DataFrame):
            column_mapping = {col: f"DLE_f_{i}" for i, col in enumerate(X.columns)}
            reference_df = reference_df.rename(columns=column_mapping)

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            reference_df["DLE_timestamp"] = X.index

        if self.target_index_ is not None:
            reference_df["DLE_prediction"] = self.model_.predict(X)[
                :, self.target_index_
            ]
        else:
            reference_df["DLE_prediction"] = self.model_.predict(X)

        if y is not None:
            if self.target_index_ is not None:
                reference_df["DLE_target"] = y[:, self.target_index_]
            else:
                reference_df["DLE_target"] = y

        return reference_df


class ClassificationCapsule(BaseCapsule, ClassifierMixin):
    """Capsule for classification tasks."""

    model_: ImplementsProba

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, model: ImplementsProba, X_test: Input, y_test: Output, **kwargs
    ) -> None:
        """Initialize the classification capsule.

        **Arguments:**
        - `model` -- Classification model implementing predict and predict_proba
        - `X_test` -- Test input data
        - `y_test` -- Test target data
        - `**kwargs` -- Additional keyword arguments for CBPE estimator

        **Raises:**
        - `ValueError` -- If multi-target classification is attempted
        """
        super().__init__(model, X_test, y_test)

        if self.n_targets_ != 1:
            raise ValueError(
                "ClassificationCapsule does not support multi-target classification."
            )

        self.n_classes_ = len(np.unique(y_test))
        reference_data = self.get_CBPE_data(self.X_test_, self.y_test_)

        is_multiclass = self.n_classes_ > 2
        problem_type = (
            "classification_multiclass" if is_multiclass else "classification_binary"
        )
        y_pred_proba = (
            {i: f"CBPE_class_{i}" for i in range(self.n_classes_)}
            if is_multiclass
            else "CBPE_proba"
        )
        timestamp_col = (
            "CBPE_timestamp" if "CBPE_timestamp" in reference_data.columns else None
        )

        self.estimator_ = nml.CBPE(
            problem_type=problem_type,
            y_pred_proba=y_pred_proba,
            y_pred="CBPE_prediction",
            y_true="CBPE_target",
            timestamp_column_name=timestamp_col,
            metrics=["f1", "roc_auc", "precision", "recall"],
            **{k: v for k, v in kwargs.items() if k not in _filter_kwargs},
        )
        self.estimator_.fit(reference_data)

    @validate_call(config={"arbitrary_types_allowed": True})
    def predict_proba(self, X: Input) -> Output:
        """Predict the probabilities for the given input.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Predicted probabilities
        """
        return self.model_.predict_proba(X)

    def get_metrics(self, X: Input) -> Result:
        """Estimate performance metrics on the analysis data using CBPE.

        **Arguments:**
        - `X` -- Input data

        **Returns:**
        Estimated performance metrics
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

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_CBPE_data(self, X: Input, y: Output | None = None) -> pd.DataFrame:
        """Generate a DataFrame with the correct structure for reference/analysis data.
        If `y` is provided, it will be included in the DataFrame.
        If index type is in a datetime format, it will create a datetime column
            "timestamp" automatically.

        **Arguments:**
        - `X` -- Input data
        - `y` -- Target data (optional)

        **Raises:**
        - `ValueError` -- If input data does not have the expected number of features

        **Returns:**
        DataFrame with reference or analysis data
        """
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Input data must have {self.n_features_} features, "
                f"but got {X.shape[1]} features."
            )

        reference_df = (
            pd.DataFrame(X)
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(
                X, columns=[f"CBPE_f_{i}" for i in range(self.n_features_)]
            )
        )
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            reference_df["CBPE_timestamp"] = X.index

        reference_df["CBPE_prediction"] = self.model_.predict(X)
        if y is not None:
            reference_df["CBPE_target"] = y

        proba = self.model_.predict_proba(X)
        if self.n_classes_ > 2:
            for i in range(self.n_classes_):
                reference_df[f"CBPE_class_{i}"] = proba[:, i]
        else:
            reference_df["CBPE_proba"] = proba[:, 1]

        return reference_df
