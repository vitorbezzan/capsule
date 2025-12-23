"""Capsules for regression tasks."""

import typing as tp

import nannyml as nml
import pandas as pd
from nannyml.base import Result
from pydantic import NonNegativeInt, validate_call
from sklearn.base import RegressorMixin

from capsule.base import BaseCapsule, ImplementsPredict, Input, Output, chunker_args
from capsule.regression_plots import RegressionPlots


class RegressionCapsule(BaseCapsule, RegressorMixin):
    """Capsule implementation for regression tasks.

    This class wraps regression models and provides performance estimation
    using Direct Loss Estimation (DLE) from the nannyml library. It supports
    both single and multi-target regression scenarios.

    Attributes:
        model_: The wrapped regression model implementing ImplementsPredict.
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
        """Initialize the regression capsule with DLE estimator.

        Sets up the regression capsule with a Direct Loss Estimation (DLE)
        estimator for performance monitoring. The DLE estimator is fitted
        on the provided test data to serve as reference data.

        Args:
            model: A trained regression model implementing predict method.
            X_test: Test input data for reference.
            y_test: Test target data for reference.
            target_index: Index of target variable for multi-target regression.
                If None, assumes single-target regression.
            **chunk_args: Additional keyword arguments passed to DLE estimator,
                to be filtered to exclude reserved parameter names.

        Note:
            Multi-target regression requires specifying target_index to
            indicate which target variable to monitor.
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
        """Generate properly formatted DataFrame for DLE analysis.

        Creates a DataFrame with the structure required by the DLE estimator,
        including feature columns, predictions, and optionally target values.
        Automatically handles datetime indexing for time-series data.

        Args:
            X: Input data to format.
            y: Target data to include (optional). If provided, adds target
                column to the DataFrame.

        Returns:
            DataFrame formatted for DLE with columns:
                - DLE_f_0, DLE_f_1, ...: Feature columns
                - DLE_prediction: Model predictions
                - DLE_target: Target values (if y provided)
                - DLE_timestamp: Timestamp column (if datetime index)

        Raises:
            ValueError: If input data doesn't have expected number of features.
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
        """Estimate regression performance metrics using DLE.

        Uses the fitted DLE estimator to estimate performance metrics
        (MAE, MAPE, MSE, RMSE) on the provided analysis data without
        requiring true target values.

        Args:
            X: Analysis input data for performance estimation.
            metric: Metric to use. In regression, this can be "mae", "mape",
                "mse", or "rmse". Default is "mape".

        Returns:
            DataFrame containing estimated performance metrics over time.

        Raises:
            ValueError: If timestamp column is required but missing from data.
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
        """Access regression-specific plotting methods.

        Provides access to plotting utilities tailored for regression analysis,
        such as scatter plots comparing true vs. predicted values.

        Returns:
            An instance of RegressionPlots for generating regression plots.
        """
        return RegressionPlots(self)

    def _get_metrics_result(self, X: Input) -> Result:
        """Internal function to return the full set of metrics."""
        analysis_data = self.format_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "DLE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'DLE_timestamp' is required for analysis."
            )

        return self.estimator_.estimate(analysis_data)
