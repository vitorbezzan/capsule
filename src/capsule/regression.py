"""Capsules for regression tasks."""

import nannyml as nml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nannyml._typing import Result
from pydantic import NonNegativeInt, validate_call
from sklearn.base import RegressorMixin
from scipy.interpolate import UnivariateSpline

from capsule import BaseCapsule
from capsule.base import ImplementsPredict, Input, Output, filter_kwargs


class RegressionPlots:
    """Plots class for regression tasks."""

    def __init__(self, capsule: "RegressionCapsule"):
        """Initialize the RegressionPlot with a RegressionCapsule instance."""
        self.capsule = capsule

    def scatter(
        self,
        X: Input | None = None,
        y: Output | None = None,
        **scatter_args,
    ) -> plt.Axes:
        """Plot a scatter plot of true vs. predicted values.

        Generates scatter plot comparing the true target values to the predicted values
        for regression analysis. If no input data is provided, uses the test data stored
        in the capsule. Optionally accepts additional keyword arguments for customizing
        the scatter plot.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            **scatter_args: Additional keyword arguments passed to matplotlib's scatter.

        Returns:
            The matplotlib Axes object containing the scatter plot.
        """
        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_pred = np.array(
            self.capsule.model_.predict(self.capsule.X_test_ if X is None else X)
        )

        if self.capsule.n_targets_ > 1:
            y_true = y_true[:, self.capsule.target_index_]
            y_pred = y_pred[:, self.capsule.target_index_]

        _, ax = plt.subplots()

        m = np.min([y_true.min(), y_pred.min()])
        M = np.max([y_true.max(), y_pred.max()])

        ax.scatter(y_true, y_pred, **scatter_args)
        ax.plot([m, M], [m, M], "k--", lw=1, label="Reference Line")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()

        return ax

    def residuals_plot(
        self,
        X: Input | None = None,
        y: Output | None = None,
        n_bins: int | None = None,
        **scatter_args,
    ) -> plt.Axes:
        """Plot a residual plot of residuals vs. predicted values.

        Generates a residual plot showing the residuals (true - predicted values)
        against the predicted values for regression analysis. This plot is useful
        for diagnosing model performance and identifying patterns in prediction errors.
        If no input data is provided, uses the test data stored in the capsule.
        Optionally accepts additional keyword arguments for customizing the scatter plot.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            n_bins: Number of bins to use for smoothing the residual trend line.
                If None, defaults to min(50, len(data) // 10).
            **scatter_args: Additional keyword arguments passed to matplotlib's scatter.

        Returns:
            The matplotlib Axes object containing the residual plot.
        """
        y_true = np.array(self.capsule.y_test_ if y is None else y)
        y_pred = np.array(
            self.capsule.model_.predict(self.capsule.X_test_ if X is None else X)
        )

        if self.capsule.n_targets_ > 1:
            y_true = y_true[:, self.capsule.target_index_]
            y_pred = y_pred[:, self.capsule.target_index_]

        residuals = y_true - y_pred

        _, ax = plt.subplots()

        ax.scatter(y_pred, residuals, **scatter_args)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1, label="Reference Line")

        try:
            df = pd.DataFrame({"predictions": y_pred, "residuals": residuals})
            df["pred_bins"] = pd.cut(
                df["predictions"],
                bins=n_bins or min(50, len(df) // 10),
                duplicates="drop",
            )

            grouped = (
                df.groupby("pred_bins", observed=True)
                .agg(
                    {
                        "predictions": "mean",
                        "residuals": "mean",
                    }
                )
                .dropna()
            )

            if len(grouped) >= 4:
                grouped = grouped.sort_values("predictions")
                spline = UnivariateSpline(
                    grouped["predictions"],
                    grouped["residuals"],
                    s=len(grouped) * 0.1,
                )

                # Generate smooth curve for plotting
                x_smooth = np.linspace(
                    grouped["predictions"].min(),
                    grouped["predictions"].max(),
                    100,
                )
                y_smooth = spline(x_smooth)

                ax.plot(x_smooth, y_smooth, "r-", linewidth=1, label="Residual Trend")

        except (ValueError, np.linalg.LinAlgError):
            pass

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals (True - Predicted)")
        ax.legend()

        return ax


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
        target_index: NonNegativeInt | None = None,
        **kwargs,
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
            **kwargs: Additional keyword arguments passed to DLE estimator,
                filtered to exclude reserved parameter names.

        Note:
            Multi-target regression requires specifying target_index to
            indicate which target variable to monitor.
        """
        super().__init__(model, X_test, y_test)

        self.target_index_ = target_index
        if self.n_targets_ > 1 and self.target_index_ is None:
            raise ValueError(
                "For multi-target regression, target_index must be specified "
                "to indicate which target variable to monitor."
            )

        reference_data = self.get_DLE_data(self.X_test_, self.y_test_)

        timestamp_col = (
            "DLE_timestamp" if "DLE_timestamp" in reference_data.columns else None
        )

        self.estimator_ = nml.DLE(
            feature_column_names=[
                col for col in reference_data.columns if col.startswith("DLE_f_")
            ],
            y_pred="DLE_prediction",
            y_true="DLE_target",
            timestamp_column_name=timestamp_col,
            metrics=["mae", "mape", "mse", "rmse"],
            **{k: v for k, v in kwargs.items() if k not in filter_kwargs},
        )
        self.estimator_.fit(reference_data)

    def get_metrics(self, X: Input) -> Result:
        """Estimate regression performance metrics using DLE.

        Uses the fitted DLE estimator to estimate performance metrics
        (MAE, MAPE, MSE, RMSE) on the provided analysis data without
        requiring true target values.

        Args:
            X: Analysis input data for performance estimation.

        Returns:
            DataFrame containing estimated performance metrics over time.

        Raises:
            ValueError: If timestamp column is required but missing from data.
        """
        analysis_data = self.get_DLE_data(X, None)

        if (self.estimator_.timestamp_column_name is not None) and (
            "DLE_timestamp" not in analysis_data.columns
        ):
            raise ValueError(
                "Timestamp column 'DLE_timestamp' is required for analysis."
            )

        estimation = self.estimator_.estimate(analysis_data)
        return estimation.filter(period="analysis").to_df()

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_DLE_data(self, X: Input, y: Output | None = None) -> pd.DataFrame:
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
            else pd.DataFrame(
                X, columns=[f"DLE_f_{i}" for i in range(self.n_features_)]
            )
        )

        if isinstance(X, pd.DataFrame):
            column_mapping = {col: f"DLE_f_{i}" for i, col in enumerate(X.columns)}
            reference_df = reference_df.rename(columns=column_mapping)

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            reference_df["DLE_timestamp"] = X.index

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

    @property
    def plots(self) -> RegressionPlots:
        """Access regression-specific plotting methods.

        Provides access to plotting utilities tailored for regression analysis,
        such as scatter plots comparing true vs. predicted values.

        Returns:
            An instance of RegressionPlots for generating regression plots.
        """
        return RegressionPlots(self)
