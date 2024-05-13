"""
Simple regressor metrics.
"""
# fmt: off
import typing as tp

from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error)
from sklearn.utils.validation import _check_y

from ..capsule.regressor import RegressorCapsule
from ..types import Inputs, TServiceReturn
from .service import Service

# fmt: on

regressor_metrics = {
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "mse": mean_squared_error,
    "meae": median_absolute_error,
}


class RegressorMetrics(Service):
    """
    Simple service to calculate vanilla performance metrics for regressors.
    """

    test_metrics: dict[str, float]

    def start(self, capsule_object: RegressorCapsule) -> "RegressorMetrics":
        """
        RegressorMetrics needs X_test and y_test data to work.
        """
        self.test_metrics = _get_metrics(
            capsule_object["y_test"],
            capsule_object.predict(
                tp.cast(Inputs, capsule_object["X_test"]),
            ),
        )

        return self

    def _run(self, capsule_object: RegressorCapsule, **kwargs) -> TServiceReturn:
        """
        Runs RegressorMetrics service.

        kwargs:
            X: Input data to be used for calculation.
            y: Output for X.
        """
        X, y = kwargs.get("X", None), kwargs.get("y", None)
        if X is None or y is None:
            return TServiceReturn(
                metric={
                    "test": self.test_metrics,
                }
            )

        metrics = _get_metrics(
            _check_y(y),
            capsule_object.predict(X),
        )

        return TServiceReturn(
            metric={
                "test": self.test_metrics,
                "current": metrics,
            }
        )


def _get_metrics(
    y,
    predictions,
) -> dict[str, float]:
    metrics = {
        metric: function(y, predictions)
        for metric, function in regressor_metrics.items()
    }

    return metrics


__all__ = ["RegressorMetrics"]
