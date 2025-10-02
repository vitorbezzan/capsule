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

mcp_server: FastMCP | None = None  # Starts empty


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
    server_name: str, capsule: RegressionCapsule | ClassificationCapsule
) -> None:
    """Starts MCP server with the given capsule."""
    logger.info("Starting MCP server...")
    mcp_server = FastMCP(server_name)

    # Tools for both classification and regression
    @mcp_server.tool()
    async def get_predictions_for_file(path: str, format: str = "parquet") -> str:
        """Loads data from path and returns predictions in .csv format as string.

        Args:
            path: Path of file to load
            format: Format of the data file. Supported formats are 'parquet', 'csv', and
                'numpy'.
        """
        buffer = io.BytesIO()
        data = _load_data(path, format)

        pd.DataFrame(capsule.predict(data)).to_csv(buffer, index=True)
        return buffer.getvalue().decode("utf-8")

    if isinstance(capsule, RegressionCapsule):

        @mcp_server.tool()
        async def generate_scatter_plot_predictions() -> ImageContent:
            """Plots the scatter plot for regression predictions compared to true values
            in the test set.

            Returns:
                str: Encoded image in base64 format.
            """
            fig, _ = capsule.plots.scatter()
            return _encode_image(fig)

    if isinstance(capsule, ClassificationCapsule):

        @mcp_server.tool()
        async def get_predict_proba_for_file(path: str, format: str = "parquet") -> str:
            """Loads data from path and returns predict_proba in .csv format as string.

            Args:
                path: Path of file to load
                format: Format of the data file. Supported formats are 'parquet', 'csv', and
                    'numpy'.
            """
            buffer = io.BytesIO()
            data = _load_data(path, format)

            pd.DataFrame(capsule.predict_proba(data)).to_csv(buffer, index=True)
            return buffer.getvalue().decode("utf-8")

    else:
        pass

    mcp_server.run(transport="stdio")
