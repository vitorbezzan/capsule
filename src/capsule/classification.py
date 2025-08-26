"""Capsules for classification tasks."""

import typing as tp

import nannyml as nml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nannyml.base import Result
from pydantic import validate_call
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from capsule import BaseCapsule
from capsule.base import ImplementsProba, Input, Output, filter_kwargs


class ClassificationPlots:
    """Plots class for classification tasks."""

    def __init__(self, capsule: "ClassificationCapsule"):
        """Initialize the ClassificationPlots with a ClassificationCapsule instance."""
        self.capsule = capsule

    def roc_curve(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
        **plot_args,
    ) -> plt.Axes:
        """Plot ROC curves for each class in the classification model.

        Generates Receiver Operating Characteristic (ROC) curves that show the
        performance of the classifier at different classification thresholds. For
        binary classification, a single curve is drawn. For multiclass problems,
        one curve per class is drawn using a one-vs-rest approach.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            **plot_args: Additional keyword arguments passed to matplotlib"s plot.

        Returns:
            The matplotlib Axes object containing the ROC curves.
        """
        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_proba = np.array(
            self.capsule.model_.predict_proba(self.capsule.X_test_ if X is None else X)
        )

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
        """Plot Precision-Recall curves for each class in the classification model.

        Generates Precision-Recall (PR) curves that show the trade-off between
        precision and recall for different threshold values. For binary classification,
        a single curve is drawn. For multiclass problems, one curve per class is drawn
        using a one-vs-rest approach.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            **plot_args: Additional keyword arguments passed to matplotlib"s plot.

        Returns:
            The matplotlib Axes object containing the PR curves.
        """
        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_proba = np.array(
            self.capsule.model_.predict_proba(self.capsule.X_test_ if X is None else X)
        )

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
                to be filtered to exclude reserved parameter names.

        Raises:
            ValueError: If multi-target classification is attempted (not supported).
        """
        super().__init__(
            model,
            X_test,
            y_test,
            **{k: v for k, v in kwargs.items() if k not in filter_kwargs},
        )

        if self.n_targets_ != 1:
            raise ValueError(
                "ClassificationCapsule does not support multi-target classification."
            )

        self.n_classes_ = len(np.unique(y_test))
        reference_data = self.format_data(self.X_test_, self.y_test_)

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
            **{k: v for k, v in kwargs.items() if k not in filter_kwargs},
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
        analysis_data = self.format_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "CBPE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'CBPE_timestamp' is required for analysis."
            )

        estimation = self.estimator_.estimate(analysis_data)
        return estimation.filter(period="analysis").to_df()

    @validate_call(config={"arbitrary_types_allowed": True})
    def format_data(self, X: Input, y: tp.Optional[Output] = None) -> pd.DataFrame:
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
            ValueError: If input data doesn"t have expected number of features.
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

    @property
    def plots(self) -> ClassificationPlots:
        """Access regression-specific plotting methods.

        Provides access to plotting utilities tailored for regression analysis,
        such as scatter plots comparing true vs. predicted values.

        Returns:
            An instance of RegressionPlots for generating regression plots.
        """
        return ClassificationPlots(self)
