"""Utility functions for MCP server operations."""

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ImageContent


def encode_image(figure: plt.Figure) -> ImageContent:
    """Encodes a matplotlib figure to base64 format.

    Args:
        figure: The matplotlib figure to encode.

    Returns:
        ImageContent: The encoded image in base64 format.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close()

    return Image(data=buffer.getvalue(), format="png").to_image_content()


def _load_format(path: str, data_format: str) -> Any:


def load_dataframe(path: str, data_format: str) -> Any:
    """Loads data from the given path and format.

    Args:
        path: Path to the data file.
        data_format: Format of the data file ('parquet', 'csv', or 'numpy').

    Returns:
        Loaded data as pandas DataFrame or numpy array.

    Raises:
        ValueError: If the data format is not supported.
    """
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
