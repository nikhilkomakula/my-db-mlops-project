"""Model serving module for Marvel characters classification."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger


class ModelServing:
    """Manages model serving endpoints in Databricks."""

    def __init__(self, model_name: str, endpoint_name: str) -> None:
        self.model_name = model_name
        self.endpoint_name = endpoint_name
        self.workspace = WorkspaceClient()

    def get_latest_model_version(self) -> str:
        """Fetch the latest version of the registered model."""
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model").version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self,
        version: str = "latest",
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ) -> None:
        """Deploy or update the model serving endpoint."""
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name, config=EndpointCoreConfigInput(served_entities=served_entities)
            )
        else:
            self.workspace.serving_endpoints.update(name=self.endpoint_name, served_entities=served_entities)
        logger.info(f"✅ Model serving endpoint '{self.endpoint_name}' is deployed/updated.")
