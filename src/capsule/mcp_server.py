"""Main MCP definitions for Capsule."""

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ImageContent

from capsule.classification import ClassificationCapsule
from capsule.regression import RegressionCapsule

MCP_VERSION = "202510.01"
logger = logging.getLogger(__name__)


def _encode_image(figure: plt.Figure) -> ImageContent:
    """Encodes a matplotlib figure to base64 format."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close()

    return Image(data=buffer.getvalue(), format="png").to_image_content()


def _load_data(path: str, data_format: str):
    """Loads data from the given path and format."""
    if data_format == "parquet":
        data = pd.read_parquet(path)
    elif data_format == "csv":
        data = pd.read_csv(path)
    elif data_format == "numpy":
        data = np.load(path, allow_pickle=True)
    else:
        raise ValueError(
            "Unsupported data format. Please use 'parquet', 'csv', or 'numpy'."
        )

    return data


def start_mcp_server(
    model_name: str, capsule: RegressionCapsule | ClassificationCapsule
) -> None:
    """Starts MCP server with the given capsule."""
    logger.info("Starting MCP server...")
    mcp = FastMCP(model_name)

    # Tools for both classification and regression
    @mcp.tool(
        description=f"Get predictions for a local file and save to a local path for "
        f"{model_name} model.",
    )
    async def get_predictions_for_file(
        in_path: str, in_format: str, out_path: str
    ) -> str:
        """Loads local data from in_path, calculates predictions, and saves them to
            a local out_path. Outputs a review and summary of the predictions.

        Args:
            in_path: Path of file to load in local machine.
            in_format: Format of the data file. Supported formats are 'parquet', 'csv',
                and 'numpy'.
            out_path: Path to save the predictions in local machine.
        """
        predictions = pd.DataFrame(capsule.predict(_load_data(in_path, in_format)))
        predictions.to_csv(out_path, index=True)

        return (
            f"Made {len(predictions)} predictions, with shape {predictions.shape} "
            f"and saved to {out_path}."
        )

    if isinstance(capsule, RegressionCapsule):

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
            return _encode_image(fig)

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
            return _encode_image(fig)

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
            return _encode_image(fig)

    if isinstance(capsule, ClassificationCapsule):

        @mcp.tool()
        async def get_proba_for_file(
            in_path: str, in_format: str, out_path: str
        ) -> str:
            """Loads local data from in_path, calculates probabilities, and saves them
                to a local out_path. Outputs a review and summary of the predictions.

            Args:
                in_path: Path of file to load in local machine.
                in_format: Format of the data file. Supported formats are 'parquet',
                    'csv', and 'numpy'.
                out_path: Path to save the predictions in local machine.
            """
            predictions = pd.DataFrame(
                capsule.predict_proba(_load_data(in_path, in_format))
            )
            predictions.to_csv(out_path, index=True)

            return (
                f"Made {len(predictions)} predictions, with shape {predictions.shape} "
                f"and saved to {out_path}."
            )

    else:
        pass

    mcp.run(transport="stdio")
