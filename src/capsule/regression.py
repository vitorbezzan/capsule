"""Capsules for regression tasks.

This module provides the RegressionCapsule class, which wraps regression models
and offers performance monitoring, drift detection, and plotting utilities using
the nannyml library.
"""

import typing as tp

import nannyml as nml
import pandas as pd
from nannyml.base import Result
from pydantic import NonNegativeInt, validate_call
from sklearn.base import RegressorMixin

from capsule.base import BaseCapsule, ImplementsPredict, Input, Output, chunker_args
from capsule.regression_plots import RegressionPlots


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule for regression tasks.

    Wraps a regression model and provides DLE-based performance monitoring,
    univariate drift detection, and plotting utilities.
    """

    model_: ImplementsPredict

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        model: ImplementsPredict,
        X_test: Input,
        y_test: Output,
        target_index: tp.Optional[NonNegativeInt] = None,
        **chunk_args,
    ) -> None:
        """Build a regression capsule with DLE reference data.

        Args:
            model: Trained regressor implementing ``predict``.
            X_test: Reference feature data used to fit monitoring components.
            y_test: Reference targets aligned with ``X_test``.
            target_index: For multi-target regressors, which target to monitor.
            **chunk_args: Chunking kwargs forwarded to DLE and drift calculators
                (e.g., ``chunk_size``, ``chunk_period``).

        Raises:
            ValueError: When multi-target data is provided without ``target_index``.
        """
        super().__init__(
            model,
            X_test,
            y_test,
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )

        self.target_index_ = target_index
        if self.n_targets_ > 1 and self.target_index_ is None:
            raise ValueError(
                "For multi-target regression, target_index must be specified "
                "to indicate which target variable to monitor."
            )

        reference_data = self.format_data(self.X_test_, self.y_test_)

        timestamp_col = (
            "DLE_timestamp" if "DLE_timestamp" in reference_data.columns else None
        )

        self.estimator_ = nml.DLE(
            feature_column_names=[
                col for col in reference_data.columns if col.startswith("__")
            ],
            y_pred="DLE_prediction",
            y_true="DLE_target",
            timestamp_column_name=timestamp_col,
            metrics=["mae", "mape", "mse", "rmse"],
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )

        self.estimator_.fit(reference_data)

        self._fit(
            X_test,
            **{k: v for k, v in chunk_args.items() if k in chunker_args},
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def format_data(self, X: Input, y: tp.Optional[Output] = None) -> pd.DataFrame:
        """Format data for DLE estimation.

        Returns a DataFrame with features, predictions, optional targets, and
        optional timestamps when the input index is a ``DatetimeIndex``.

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
            reference_df["__timestamp"] = X.index

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

    @validate_call(config={"arbitrary_types_allowed": True})
    def metrics(
        self,
        X: Input,
        metric: str = "mape",
    ) -> pd.DataFrame:
        """Estimate regression performance via DLE.

        Args:
            X: Analysis features.
            metric: One of ``mae``, ``mape``, ``mse``, ``rmse``.

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
    def plots(self) -> RegressionPlots:
        """Regression plotting helpers (scatter, residuals)."""
        return RegressionPlots(self)

    def _get_metrics_result(self, X: Input) -> Result:
        """Return the raw DLE result object for downstream use."""
        analysis_data = self.format_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "DLE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'DLE_timestamp' is required for analysis."
            )

        return self.estimator_.estimate(analysis_data)
