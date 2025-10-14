"""Utility functions for the Marvel characters project."""

import os

from databricks.sdk import WorkspaceClient


def is_databricks() -> bool:
    """Check if the code is running in a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_dbr_host() -> str:
    """Get the Databricks workspace URL."""
    ws = WorkspaceClient()
    return ws.config.host
