"""
Exceptions raised by Services.
"""


class ServiceStartError(Exception):
    """
    Any error raised when .start() fails.
    """


class ServiceRunError(Exception):
    """
    Any errors raised when .run() fails.
    """
