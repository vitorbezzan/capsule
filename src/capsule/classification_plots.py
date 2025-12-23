"""Classification plots for Capsule classifier models.

This module provides the `ClassificationPlots` class, which contains methods to plot
Receiver Operating Characteristic (ROC) curves and Precision-Recall (PR) curves for
classification tasks. These plots are useful for evaluating the performance of
classification models, especially in binary and multiclass settings.
"""

import typing as tp

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from capsule.base import BaseCapsule, Input, Output


class ClassificationPlots:
    """Plots for classification tasks (ROC and PR curves)."""

    def __init__(self, capsule: BaseCapsule) -> None:
        """Initialize the ClassificationPlots with a ClassificationCapsule instance."""
        self.capsule = capsule

    def roc_curve(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
        **plot_args,
    ) -> plt.Axes:
        """Plot ROC curves for each class.

        Uses the capsule's stored test data when inputs are omitted.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            **plot_args: Additional keyword arguments passed to matplotlib"s plot.

        Returns:
            The matplotlib Axes object containing the ROC curves.
        """
        new_X = self.capsule.X_test_ if X is None else X

        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_proba = np.array(self.capsule.model_.predict_proba(new_X))  # type: ignore

        _, ax = plt.subplots()

        if self.capsule.n_classes_ == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", **plot_args)

        else:
            for i in range(self.capsule.n_classes_):
                y_binary = np.where(y_true == i, 1, 0)

                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                roc_auc = auc(fpr, tpr)

                ax.plot(
                    fpr,
                    tpr,
                    label=f"ROC curve for class {i} (AUC = {roc_auc:.2f})",
                    **plot_args,
                )

        ax.plot([0, 1], [0, 1], "k--", label="Random classifier")

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")

        return ax

    def pr_curve(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
        **plot_args,
    ) -> plt.Axes:
        """Plot Precision-Recall curves for each class.

        Uses the capsule's stored test data when inputs are omitted.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            **plot_args: Additional keyword arguments passed to matplotlib"s plot.

        Returns:
            The matplotlib Axes object containing the PR curves.
        """
        new_X = self.capsule.X_test_ if X is None else X

        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_proba = np.array(self.capsule.model_.predict_proba(new_X))  # type: ignore

        _, ax = plt.subplots()

        if self.capsule.n_classes_ == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])

            ax.plot(
                recall,
                precision,
                label=f"PR curve (AP = {avg_precision:.2f})",
                **plot_args,
            )

        else:
            for i in range(self.capsule.n_classes_):
                y_binary = np.where(y_true == i, 1, 0)

                precision, recall, _ = precision_recall_curve(y_binary, y_proba[:, i])
                avg_precision = average_precision_score(y_binary, y_proba[:, i])

                ax.plot(
                    recall,
                    precision,
                    label=f"PR curve for class {i} (AP = {avg_precision:.2f})",
                    **plot_args,
                )

        no_skill_ratio = (
            np.sum(y_true == 1) / len(y_true) if self.capsule.n_classes_ == 2 else None
        )
        if no_skill_ratio is not None:
            ax.plot([0, 1], [no_skill_ratio, no_skill_ratio], "k--", label="No Skill")

        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")

        return ax
