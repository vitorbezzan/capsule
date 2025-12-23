"""Regression plots for Capsule regression models."""

import typing as tp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

from capsule.base import BaseCapsule, Input, Output


class RegressionPlots:
    """Plots class for regression tasks."""

    def __init__(self, capsule: BaseCapsule) -> None:
        """Initialize the RegressionPlot with a RegressionCapsule instance."""
        self.capsule = capsule

    def scatter(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
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
        predictions = np.array(
            self.capsule.model_.predict(self.capsule.X_test_ if X is None else X)
        )

        if self.capsule.n_targets_ > 1:
            y_true = y_true[:, self.capsule.target_index_]
            predictions = predictions[:, self.capsule.target_index_]

        _, ax = plt.subplots()

        m = np.min([y_true.min(), predictions.min()])
        M = np.max([y_true.max(), predictions.max()])

        ax.scatter(y_true, predictions, **scatter_args)
        ax.plot([m, M], [m, M], "k--", lw=1, label="Reference Line")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()

        return ax

    def residuals_plot(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
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
        predictions = np.array(
            self.capsule.model_.predict(self.capsule.X_test_ if X is None else X)
        )

        if self.capsule.n_targets_ > 1:
            y_true = y_true[:, self.capsule.target_index_]
            predictions = predictions[:, self.capsule.target_index_]

        residuals = y_true - predictions

        _, ax = plt.subplots()

        ax.scatter(predictions, residuals, **scatter_args)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=1, label="Reference Line")

        try:
            df = pd.DataFrame({"predictions": predictions, "residuals": residuals})
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

    def residuals_hist(
        self,
        X: tp.Optional[Input] = None,
        y: tp.Optional[Output] = None,
        bins: int | str | None = None,
        standard: bool = False,
        **hist_args,
    ) -> plt.Axes:
        """Plot a histogram of standardized residuals.

        Computes standardized residuals from true and predicted values and displays
        a histogram. If no data is provided, uses the capsule's stored test set.

        Args:
            X: Input data for prediction (optional). If None, uses the test input data.
            y: True target values (optional). If None, uses the test target data.
            bins: Number of bins or binning strategy for matplotlib's hist. If None,
                defaults to 30.
            standard: If True, uses the standard standardized residuals.
            **hist_args: Additional keyword arguments passed to matplotlib's hist.

        Returns:
            The matplotlib Axes object containing the histogram.
        """
        y_true = np.array(self.capsule.y_test_ if y is None else y)
        predictions = np.array(
            self.capsule.model_.predict(self.capsule.X_test_ if X is None else X)
        )

        if self.capsule.n_targets_ > 1:
            y_true = y_true[:, self.capsule.target_index_]
            predictions = predictions[:, self.capsule.target_index_]

        residuals = y_true - predictions
        residuals = residuals.astype(float)
        residuals = residuals[~np.isnan(residuals)]

        std = np.std(residuals, ddof=1) if residuals.size > 1 else 0.0
        mean = float(np.mean(residuals)) if residuals.size > 0 else 0.0

        if standard:
            if std > 0:
                residuals = (residuals - mean) / std

                mean = 0.0
                std = 1.0
            else:
                residuals = np.zeros_like(residuals)

        _, ax = plt.subplots()
        ax.hist(residuals, bins=(bins or 30), **hist_args)

        ax.axvline(
            x=mean, color="k", linestyle="--", linewidth=1, label=f"Mean = {mean:.2f}"
        )
        ax.axvline(
            x=0, color="k", linestyle="dotted", linewidth=1, label="Expected Mean"
        )
        ax.axvline(
            x=float(2 * std),
            color="r",
            linestyle=":",
            linewidth=1,
            label=f"+-2 std = {std:.2f}",
        )
        ax.axvline(x=-float(2 * std), color="r", linestyle=":", linewidth=1)
        ax.axvline(
            x=float(3 * std),
            color="r",
            linestyle="-.",
            linewidth=1,
            label=f"+-3 std = {std:.2f}",
        )
        ax.axvline(x=-float(3 * std), color="r", linestyle="-.", linewidth=1)

        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")
        ax.set_title("Residuals Histogram")
        ax.legend()

        return ax
