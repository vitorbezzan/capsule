"""MCP tools for classification capsules."""

import pandas as pd
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent

from capsule.classification import ClassificationCapsule


def register_classification_tools(
    mcp: FastMCP, model_name: str, capsule: ClassificationCapsule
) -> None:
    """Register classification-specific MCP tools.

    Args:
        mcp: The FastMCP instance to register tools with.
        model_name: Name of the model for tool descriptions.
        capsule: The classification capsule instance.
    """
    from .utils import encode_image, load_dataframe

    @mcp.tool()
    async def get_proba_for_file(in_path: str, in_format: str, out_path: str) -> str:
        """Loads local data from in_path, calculates probabilities, and saves them
            to a local out_path. Outputs a review and summary of the predictions.

        Args:
            in_path: Path of file to load in local machine.
            in_format: Format of the data file. Supported formats are 'parquet',
                'csv', and 'numpy'.
            out_path: Path to save the predictions in local machine.
        """
        predictions = pd.DataFrame(capsule.predict_proba(load_dataframe(in_path, in_format)))
        predictions.to_csv(out_path, index=True)

        return (
            f"Made {len(predictions)} predictions, with shape {predictions.shape} "
            f"and saved to {out_path}."
        )

    @mcp.tool(
        description=f"Get ROC curve using files for X inputs and y inputs for "
        f"{model_name} classification model.",
    )
    async def get_roc_curve_from_files(
        x_path: str, x_format: str, y_path: str, y_format: str
    ) -> ImageContent:
        """Loads X and y data from files, generates predictions, and creates a ROC
            curve for classification capsules.

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
        fig, _ = capsule.plots.roc_curve(X, y)
        return encode_image(fig)

    @mcp.tool(
        description=f"Get Precision-Recall curve using files for X inputs and y inputs "
        f"for {model_name} classification model.",
    )
    async def get_pr_curve_from_files(
        x_path: str, x_format: str, y_path: str, y_format: str
    ) -> ImageContent:
        """Loads X and y data from files, generates predictions, and creates a
            Precision-Recall curve for classification capsules.

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
        fig, _ = capsule.plots.pr_curve(X, y)
        return encode_image(fig)
