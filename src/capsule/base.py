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

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        model: ImplementsPredict | ImplementsProba,
        X_test: Input,
        y_test: Output,
    ) -> None:
        """Initialize the base capsule with a trained model and test data.

        Args:
            model: A trained model that implements prediction methods.
            X_test: Test input data used for reference during performance estimation.
            y_test: Test target data corresponding to X_test.

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

        self.X_test_ = X_test
        self.y_test_ = y_test

        self.model_ = model
        self.n_features_ = X_test.shape[1]
        self.n_targets_ = 1 if y_test.ndim == 1 else y_test.shape[1]

    def __getstate__(self) -> dict:
        """Prepare capsule state for serialization with optional encryption.

        If the CAPSULE_KEY environment variable is set, the capsule state
        will be encrypted using AES-GCM encryption before serialization.

        Returns:
            Dictionary containing serialized (and possibly encrypted) state.
        """
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
        """Restore capsule state from serialized data with optional decryption.

        If the state contains encrypted data (indicated by presence of 'nonce'),
        it will be decrypted using the CAPSULE_KEY environment variable.

        Args:
            state: Dictionary containing serialized state data.

        Raises:
            RuntimeError: If encrypted data is found but CAPSULE_KEY is not set.
        """
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

    def predict_proba(self, X: Input) -> Output:
        """Generate probability predictions.

        This base implementation raises NotImplementedError. Subclasses
        should override this method if they support probability predictions.

        Args:
            X: Input data for probability prediction.

        Returns:
            Predicted probabilities (not implemented in base class).

        Raises:
            NotImplementedError: Always raised in base implementation.
        """
        raise NotImplementedError("Not implemented for a generic capsule.")

    @abstractmethod
    def get_metrics(self, X: Input) -> Result:
        """Estimate performance metrics on analysis data.

        This abstract method must be implemented by subclasses to provide
        performance estimation capabilities specific to their task type.

        Args:
            X: Analysis data for performance estimation.

        Returns:
            Performance estimation results.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Must be implemented in subclasses.")


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule implementation for regression tasks.

    This class wraps regression models and provides performance estimation
    using Direct Loss Estimation (DLE) from the nannyml library. It supports
    both single and multi-target regression scenarios.

    Attributes:
        model_: The wrapped regression model implementing ImplementsPredict.
    """

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
        """Initialize the regression capsule with DLE estimator.

        Sets up the regression capsule with a Direct Loss Estimation (DLE)
        estimator for performance monitoring. The DLE estimator is fitted
        on the provided test data to serve as reference data.

        Args:
            model: A trained regression model implementing predict method.
            X_test: Test input data for reference.
            y_test: Test target data for reference.
            target_index: Index of target variable for multi-target regression.
                If None, assumes single-target regression.
            **kwargs: Additional keyword arguments passed to DLE estimator,
                filtered to exclude reserved parameter names.

        Note:
            Multi-target regression requires specifying target_index to
            indicate which target variable to monitor.
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
        """Estimate regression performance metrics using DLE.

        Uses the fitted DLE estimator to estimate performance metrics
        (MAE, MAPE, MSE, RMSE) on the provided analysis data without
        requiring true target values.

        Args:
            X: Analysis input data for performance estimation.

        Returns:
            DataFrame containing estimated performance metrics over time.

        Raises:
            ValueError: If timestamp column is required but missing from data.
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
        """Generate properly formatted DataFrame for DLE analysis.

        Creates a DataFrame with the structure required by the DLE estimator,
        including feature columns, predictions, and optionally target values.
        Automatically handles datetime indexing for time-series data.

        Args:
            X: Input data to format.
            y: Target data to include (optional). If provided, adds target
                column to the DataFrame.

        Returns:
            DataFrame formatted for DLE with columns:
                - DLE_f_0, DLE_f_1, ...: Feature columns
                - DLE_prediction: Model predictions
                - DLE_target: Target values (if y provided)
                - DLE_timestamp: Timestamp column (if datetime index)

        Raises:
            ValueError: If input data doesn't have expected number of features.
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
    """Capsule implementation for classification tasks.

    This class wraps classification models and provides performance estimation
    using Confidence-based Performance Estimation (CBPE) from the nannyml
    library. It supports both binary and multiclass classification scenarios.

    Attributes:
        model_: The wrapped classification model implementing ImplementsProba.
    """

    model_: ImplementsProba

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, model: ImplementsProba, X_test: Input, y_test: Output, **kwargs
    ) -> None:
        """Initialize the classification capsule with CBPE estimator.

        Sets up the classification capsule with a Confidence-based Performance
        Estimation (CBPE) estimator for performance monitoring. The CBPE
        estimator is fitted on the provided test data to serve as reference.

        Args:
            model: A trained classification model implementing predict and
                predict_proba methods.
            X_test: Test input data for reference.
            y_test: Test target data for reference.
            **kwargs: Additional keyword arguments passed to CBPE estimator,
                filtered to exclude reserved parameter names.

        Raises:
            ValueError: If multi-target classification is attempted (not supported).
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
        """Generate class probability predictions using the wrapped model.

        Args:
            X: Input data for probability prediction.

        Returns:
            Predicted class probabilities from the wrapped model.
        """
        return self.model_.predict_proba(X)

    def get_metrics(self, X: Input) -> Result:
        """Estimate classification performance metrics using CBPE.

        Uses the fitted CBPE estimator to estimate performance metrics
        (F1, ROC-AUC, precision, recall) on the provided analysis data
        without requiring true target values.

        Args:
            X: Analysis input data for performance estimation.

        Returns:
            DataFrame containing estimated performance metrics over time.

        Raises:
            ValueError: If timestamp column is required but missing from data.
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
        """Generate properly formatted DataFrame for CBPE analysis.

        Creates a DataFrame with the structure required by the CBPE estimator,
        including feature columns, predictions, predicted probabilities, and
        optionally target values. Automatically handles datetime indexing.

        Args:
            X: Input data to format.
            y: Target data to include (optional). If provided, adds target
                column to the DataFrame.

        Returns:
            DataFrame formatted for CBPE with columns:
                - CBPE_f_0, CBPE_f_1, ...: Feature columns
                - CBPE_prediction: Model predictions
                - CBPE_target: Target values (if y provided)
                - CBPE_proba: Prediction probabilities (binary classification)
                - CBPE_class_0, CBPE_class_1, ...: Class probabilities (multiclass)
                - CBPE_timestamp: Timestamp column (if datetime index)

        Raises:
            ValueError: If input data doesn't have expected number of features.
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
