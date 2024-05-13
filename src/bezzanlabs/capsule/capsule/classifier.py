"""
Defines behavior for a simple classifier Capsule.
"""
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import _check_y, check_array

from ..proto import Classifier
from ..types import Actuals, Inputs, Pipeline, Predictions
from .capsule import Capsule


class ClassifierCapsule(Capsule, ClassifierMixin):
    """
    Capsule for simple classifiers.
    """

    def __init__(
        self,
        model: Classifier | Pipeline,
        X_test: Inputs,
        y_test: Actuals,
        **kwargs,
    ):
        """
        Constructor for ClassifierCapsule.
        """
        super().__init__(
            model,
            X_test,
            _check_y(y_test),
            **kwargs,
        )

    @property
    def model(self) -> Classifier | Pipeline:
        return super().model

    def predict(self, X: Inputs) -> Predictions:
        """
        Outputs predictions.
        """
        return self.model.predict(
            check_array(
                X,
                force_all_finite="allow-nan",
            )
        )

    def predict_proba(self, X: Inputs) -> Predictions:
        """
        Outputs probabilities.
        """
        return self.model.predict_proba(
            check_array(
                X,
                force_all_finite="allow-nan",
            )
        )
