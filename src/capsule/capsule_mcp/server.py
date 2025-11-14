"""Main MCP server definitions for Capsule."""

import logging

import pandas as pd
from mcp.server.fastmcp import FastMCP

from capsule.classification import ClassificationCapsule
from capsule.regression import RegressionCapsule

from .classification_tools import register_classification_tools
from .regression_tools import register_regression_tools
from .utils import load_dataframe

MCP_VERSION = "202510.02"
logger = logging.getLogger(__name__)


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
        predictions = pd.DataFrame(capsule.predict(load_dataframe(in_path, in_format)))
        predictions.to_csv(out_path, index=True)

        return (
            f"Made {len(predictions)} predictions, with shape {predictions.shape} "
            f"and saved to {out_path}."
        )

    if isinstance(capsule, RegressionCapsule):
        register_regression_tools(mcp, model_name, capsule)
    elif isinstance(capsule, ClassificationCapsule):
        register_classification_tools(mcp, model_name, capsule)

    mcp.run(transport="stdio")
