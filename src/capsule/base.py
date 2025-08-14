"""Base classes for the capsule package."""

import os
import pickle
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from nannyml.base import Result
from numpy.typing import NDArray
from pydantic import validate_call
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

Input: tp.TypeAlias = NDArray[np.float64] | pd.DataFrame
Output: tp.TypeAlias = NDArray[np.float64]


filter_kwargs = {
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

    @property
    @abstractmethod
    def plots(self) -> object:
        """Get plots for the capsule."""
        raise NotImplementedError("Must be implemented in subclasses.")
