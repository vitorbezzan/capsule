"""Main MCP definitions for Capsule."""

import io
import logging

import matplotlib.pyplot as plt
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


def start_mcp_server(
    server_name: str, capsule: RegressionCapsule | ClassificationCapsule
) -> None:
    """Starts MCP server with the given capsule."""
    logger.info("Starting MCP server...")
    mcp_server = FastMCP(server_name)

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

    else:
        pass

    mcp_server.run(transport="stdio")
