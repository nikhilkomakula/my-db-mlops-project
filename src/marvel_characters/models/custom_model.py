"""Custom model implementation for Marvel characters classification."""

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

from marvel_characters.config import Tags


def adjust_predictions(predictions: np.ndarray | list[int]) -> dict[str, list[str]]:
    """Adjust predictions to return a human-readable format."""
    return {"Survival prediction": ["alive" if pred == 1 else "dead" for pred in predictions]}


class MarvelModelWrapper(mlflow.pyfunc.PythonModel):
    """A custom model wrapper for Marvel characters classification."""

    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
        """Load the LightGBM model from the MLflow context."""
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline"])

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray) -> dict:
        """Make predictions using the loaded LightGBM model."""
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)

    def log_register_model(
        self,
        wrapped_model_uri: str,
        pyfunc_model_name: str,
        experiment_name: str,
        tags: Tags,
        code_paths: list[str],
        input_example: pd.DataFrame,
    ) -> None:
        """Log and register the custom model in MLflow."""
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"wrapper-lightgbm-{datetime.now().strftime('%Y-%m-%d')}", tags=tags.to_dict()):
            additional_pip_deps = []
            for package in code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"code/{whl_name}")
            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)
            signature = infer_signature(model_input=input_example, model_output={"Survival prediction": ["alive"]})

            model_info = mlflow.pyfunc.log_model(
                python_model=self,
                name="pyfunc-wrapper",
                artifacts={"lightgbm-pipeline": wrapped_model_uri},
                signature=signature,
                code_paths=code_paths,
                conda_env=conda_env,
            )
        client = MlflowClient()
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri, name=pyfunc_model_name, tags=tags.to_dict()
        )
        latest_version = registered_model.version
        client.set_registered_model_alias(name=pyfunc_model_name, alias="latest-model", version=latest_version)
        logger.info(f"✅ Model registered as version {latest_version}.")
        return latest_version
