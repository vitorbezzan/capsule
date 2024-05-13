"""
Defines behavior for a simple regressor Capsule.
"""
from sklearn.base import RegressorMixin
from sklearn.utils.validation import _check_y, check_array

from ..proto import Regressor
from ..types import Actuals, Inputs, Pipeline, Predictions
from .capsule import Capsule


class RegressorCapsule(Capsule, RegressorMixin):
    """
    Capsule for simple regressors.
    """

    def __init__(
        self,
        model: Regressor | Pipeline,
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
    def model(self) -> Regressor | Pipeline:
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
