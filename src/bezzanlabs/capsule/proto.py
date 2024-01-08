"""
Prototypes for model classes and supporting functions.
"""
import typing as tp
from .types import Inputs, Predictions


@tp.runtime_checkable
class Classifier(tp.Protocol):
    """
    Defines the base behavior for a Classifier object.
    """

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Defines predict_proba method for a Classifier.
        Should return a NDArray of shape (n_samples, n_classes).
        """

    def predict(self, X: Inputs) -> Predictions:
        """
        Defines predict method for a Classifier.
        Should return a NDArray of shape (n_samples,).
        """


@tp.runtime_checkable
class Regressor(tp.Protocol):
    """
    Defines the base behavior for a Regressor object.
    """
    def predict(self, X: Inputs) -> Predictions:
        """
        Defines predict method for a Regressor.
        Should return a NDArray of shape (n_samples,).
        """
