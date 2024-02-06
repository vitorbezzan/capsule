"""
UTC time timestamp clock.
"""
import logging
import secrets
from datetime import datetime, timezone

from ntplib import NTPClient, NTPException  # type: ignore

__ntp_servers: list[str] = [
    "europe.pool.ntp.org",
    "north-america.pool.ntp.org",
    "asia.pool.ntp.org",
    "south-america.pool.ntp.org",
    "africa.pool.ntp.org",
]

logger = logging.getLogger(__name__)


def get_utc_time(
    error_if_not_available: bool = False,
    ntp_servers: list[str] | None = None,
    external_logger: logging.Logger | None = None,
) -> datetime:
    """
    Gets UTC datetime timestamp from random NTP server. If server is not available,
    uses current machine time, and emits a warning.

    Raises:
        RuntimeError: If error == True and NTP time servers are not available.
    """
    internal_logger = external_logger or logger

    servers_list = ntp_servers or __ntp_servers
    selected_server = secrets.choice(servers_list)

    try:
        client = NTPClient()
        internal_logger.debug(f"Obtaining time from {selected_server}...")
        response = client.request(selected_server, version=3)

        return datetime.fromtimestamp(response.tx_time, tz=timezone.utc)

    except NTPException as ntp_error:
        internal_logger.warning(
            f"UTC timestamp retrieval failed for {selected_server}. Using machine time."
        )
        if error_if_not_available:
            raise RuntimeError("UTC timestamp retrieval failed.") from ntp_error

        return datetime.now(tz=timezone.utc)
