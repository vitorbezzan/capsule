"""
Simple classifier metrics.
"""
# fmt: off
import typing as tp

from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.utils.validation import _check_y

from ..capsule.classifier import ClassifierCapsule
from ..types import Inputs, TServiceReturn
from .service import Service

# fmt: on

classification_metrics = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


class ClassifierMetrics(Service):
    """
    Simple service to calculate vanilla performance metrics for Classifiers.
    """

    test_metrics: dict[str, float]

    def start(self, capsule_object: ClassifierCapsule) -> "ClassifierMetrics":
        """
        ClassifierMetrics needs X_test and y_test data to work.
        """
        self.test_metrics = _get_metrics(
            capsule_object["y_test"],
            capsule_object.predict(
                tp.cast(Inputs, capsule_object["X_test"]),
            ),
            capsule_object.predict_proba(
                tp.cast(Inputs, capsule_object["X_test"]),
            ),
        )

        return self

    def _run(self, capsule_object: ClassifierCapsule, **kwargs) -> TServiceReturn:
        """
        Runs ClassifierMetrics service.

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
            capsule_object.predict_proba(X),
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
    predictions_proba,
) -> dict[str, float]:
    n_classes = predictions_proba.shape[1]

    metrics = {
        metric: function(y, predictions)
        for metric, function in classification_metrics.items()
    }
    metrics["roc_auc"] = roc_auc_score(
        y,
        predictions_proba if n_classes > 2 else predictions_proba[:, 1],
    )

    return metrics


__all__ = ["ClassifierMetrics"]
