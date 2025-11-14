"""MCP tools for regression capsules."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent

from capsule.regression import RegressionCapsule


def register_regression_tools(
    mcp: FastMCP, model_name: str, capsule: RegressionCapsule
) -> None:
    """Register regression-specific MCP tools.

    Args:
        mcp: The FastMCP instance to register tools with.
        model_name: Name of the model for tool descriptions.
        capsule: The regression capsule instance.
    """
    from .utils import encode_image, load_dataframe

    @mcp.tool(
        description=f"Get scatter plot of true vs predicted values on the test set "
        f"for {model_name} regression model.",
    )
    async def get_test_scatterplot() -> ImageContent:
        """Outputs the scatter plot of true vs predicted values on the test set for
            regression capsules.

        Returns:
            str: Encoded image in base64 format.
        """
        fig, _ = capsule.plots.scatter()
        return encode_image(fig)

    @mcp.tool(
        description=f"Get residuals plot of true vs predicted values on the test "
        f"set for {model_name} regression model.",
    )
    async def get_test_residuals_plot() -> ImageContent:
        """Outputs the residuals plot of true vs predicted values on the test set
            for regression capsules.

        Returns:
            str: Encoded image in base64 format.
        """
        fig, _ = capsule.plots.residuals_plot()
        return encode_image(fig)

    @mcp.tool(
        description=f"Get residuals histogram plot of true vs predicted values on "
        f"the test set for {model_name} regression model.",
    )
    async def get_test_hist_residuals_plot() -> ImageContent:
        """Outputs the histogram residuals plot of true vs predicted values on the
            test set for regression capsules.

        Returns:
            str: Encoded image in base64 format.
        """
        fig, _ = capsule.plots.residuals_hist()
        return encode_image(fig)

    @mcp.tool(
        description=f"Get scatter plot of true vs predicted values using files for "
        f"X inputs and y inputs for {model_name} regression model.",
    )
    async def get_scatterplot_from_files(
        x_path: str, x_format: str, y_path: str, y_format: str
    ) -> ImageContent:
        """Loads X and y data from files, generates predictions, and creates a scatter
            plot of true vs predicted values for regression capsules.

        Args:
            x_path: Path of file containing X (input) data.
            x_format: Format of the X data file. Supported formats are 'parquet',
                'csv', and 'numpy'.
            y_path: Path of file containing y (target) data.
            y_format: Format of the y data file. Supported formats are 'parquet',
                'csv', and 'numpy'.

        Returns:
            str: Encoded image in base64 format.
        """
        X = load_dataframe(x_path, x_format)
        y = load_dataframe(y_path, y_format)
        fig, _ = capsule.plots.scatter(X, y)
        return encode_image(fig)

    @mcp.tool(
        description=f"Get residuals plot of true vs predicted values using files for "
        f"X inputs and y inputs for {model_name} regression model.",
    )
    async def get_residuals_plot_from_files(
        x_path: str, x_format: str, y_path: str, y_format: str
    ) -> ImageContent:
        """Loads X and y data from files, generates predictions, and creates a residuals
            plot of residuals vs predicted values for regression capsules.

        Args:
            x_path: Path of file containing X (input) data.
            x_format: Format of the X data file. Supported formats are 'parquet',
                'csv', and 'numpy'.
            y_path: Path of file containing y (target) data.
            y_format: Format of the y data file. Supported formats are 'parquet',
                'csv', and 'numpy'.

        Returns:
            str: Encoded image in base64 format.
        """
        X = load_dataframe(x_path, x_format)
        y = load_dataframe(y_path, y_format)
        fig, _ = capsule.plots.residuals_plot(X, y)
        return encode_image(fig)
