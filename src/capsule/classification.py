"""Capsules for classification tasks."""

import typing as tp

import nannyml as nml
import numpy as np
import pandas as pd
from nannyml.base import Result
from pydantic import validate_call
from sklearn.base import ClassifierMixin

from capsule.base import BaseCapsule, ImplementsProba, Input, Output, chunker_args
from capsule.classification_plots import ClassificationPlots


class ClassificationCapsule(BaseCapsule, ClassifierMixin):
    """Capsule for classification tasks.

    Wraps a classification model and provides CBPE-based performance monitoring,
    univariate drift detection, and plotting utilities.
    """

    model_: ImplementsProba

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, model: ImplementsProba, X_test: Input, y_test: Output, **chunk_args
    ) -> None:
        """Build a classification capsule with CBPE reference data.

        Args:
            model: Trained classifier implementing ``predict`` and ``predict_proba``.
            X_test: Reference feature data used to fit monitoring components.
            y_test: Reference targets aligned with ``X_test``.
            **chunk_args: Chunking kwargs forwarded to CBPE and drift calculators
                (e.g., ``chunk_size``, ``chunk_period``).

        Raises:
            ValueError: When multiple target columns are provided.
        """
        super().__init__(
            model,
            X_test,
            y_test,
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )

        if self.n_targets_ != 1:
            raise ValueError(
                "ClassificationCapsule does not support multi-target classification."
            )

        self.n_classes_ = len(np.unique(y_test))
        reference_data = self.format_data(self.X_test_, self.y_test_)

        is_multiclass = self.n_classes_ > 2

        self.problem_type = (
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
            problem_type=self.problem_type,
            y_pred_proba=y_pred_proba,
            y_pred="CBPE_prediction",
            y_true="CBPE_target",
            timestamp_column_name=timestamp_col,
            metrics=["f1", "roc_auc", "precision", "recall"],
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )
        self.estimator_.fit(reference_data)

        self._fit(
            X_test,
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def predict_proba(self, X: Input) -> Output:
        """Return class probabilities from the wrapped model."""
        return self.model_.predict_proba(X)

    @validate_call(config={"arbitrary_types_allowed": True})
    def format_data(self, X: Input, y: tp.Optional[Output] = None) -> pd.DataFrame:
        """Format data for CBPE estimation.

        Returns a DataFrame with features, predictions, predicted probabilities,
        optional targets, and optional timestamps when the input index is a
        ``DatetimeIndex``.

        Raises:
            ValueError: If the input feature count differs from the reference.
        """
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Input data must have {self.n_features_} features, "
                f"but got {X.shape[1]} features."
            )

        reference_df = (
            pd.DataFrame(X)
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=range(self.n_features_))
        )

        column_mapping = {col: f"__{col}" for i, col in enumerate(reference_df.columns)}
        reference_df = reference_df.rename(columns=column_mapping)

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

    @validate_call(config={"arbitrary_types_allowed": True})
    def metrics(
        self,
        X: Input,
        metric: str = "f1",
    ) -> pd.DataFrame:
        """Estimate classification performance via CBPE.

        Args:
            X: Analysis features.
            metric: One of ``f1``, ``roc_auc``, ``precision``, ``recall``.

        Returns:
            DataFrame of estimated performance over analysis chunks.

        Raises:
            ValueError: If a timestamp column is required but missing.
        """
        df = (
            self._get_metrics_result(X)
            .filter(
                period="analysis",
                metrics=[metric],
            )
            .to_df()
        )

        df.columns = df.columns.droplevel()
        return df

    @property
    def plots(self) -> ClassificationPlots:
        """Classification plotting helpers (ROC, PR curves)."""
        return ClassificationPlots(self)

    def _get_metrics_result(self, X: Input) -> Result:
        """Return the raw CBPE result object for downstream use."""
        analysis_data = self.format_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "CBPE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'CBPE_timestamp' is required for analysis."
            )

        return self.estimator_.estimate(analysis_data)
