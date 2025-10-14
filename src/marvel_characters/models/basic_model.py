"""Basic model implementation for Marvel characters classification."""

from typing import Self

import mlflow
import pandas as pd
from delta.tables import DeltaTable
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from marvel_characters.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for Marvel characters classification."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        self.config = config
        self.tags = tags
        self.spark = spark

        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.marvel_character_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
        logger.info("Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.mc_train_set")
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.mc_test_set")

        self.train_set_pandas = self.train_set_spark.toPandas()
        self.test_set_pandas = self.test_set_spark.toPandas()

        self.X_train = self.train_set_pandas[self.num_features + self.cat_features]
        self.y_train = self.train_set_pandas[self.target]
        self.X_test = self.test_set_pandas[self.num_features + self.cat_features]
        self.y_test = self.test_set_pandas[self.target]
        self.eval_data = self.test_set_pandas[self.num_features + self.cat_features + [self.target]]

        train_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.mc_train_set")
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(self.spark, f"{self.catalog_name}.{self.schema_name}.mc_test_set")
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline."""
        logger.info("Preparing features...")

        class CatToIntTransformer(BaseEstimator, TransformerMixin):
            """Transformer that encodes categorical columns as integer codes for LightGBM.

            Unknown categories at transform time are encoded as -1.
            """

            def __init__(self, cat_features: list[str]) -> None:
                """Initialize the transformer with categorical feature names."""
                self.cat_features = cat_features
                self.cat_maps_ = {}

            def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
                """Fit the transformer to the DataFrame X."""
                self.fit_transform(X)
                return self

            def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
                """Fit and transform the DataFrame X."""
                X = X.copy()
                for col in self.cat_features:
                    c = pd.Categorical(X[col])
                    # Build mapping: {category: code}
                    self.cat_maps_[col] = dict(zip(c.categories, range(len(c.categories)), strict=False))
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

            def transform(self, X: pd.DataFrame) -> pd.DataFrame:
                """Transform the DataFrame X by encoding categorical features as integers."""
                X = X.copy()
                for col in self.cat_features:
                    X[col] = X[col].map(lambda val, col=col: self.cat_maps_[col].get(val, -1)).astype("category")
                return X

        preprocessor = ColumnTransformer(
            transformers=[("cat", CatToIntTransformer(self.cat_features), self.cat_features)], remainder="passthrough"
        )
        self.pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("Training the model...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model successfully trained.")

    def log_model(self) -> None:
        """Log the trained model to MLflow."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            signature = infer_signature(model_input=self.X_train, model_output=self.pipeline.predict(self.X_train))

            train_dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.mc_train_set",
                version=self.train_data_version,
            )
            mlflow.log_input(train_dataset, context="training")

            test_dataset = mlflow.data.from_spark(
                self.test_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.mc_test_set",
                version=self.test_data_version,
            )
            mlflow.log_input(test_dataset, context="testing")

            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            eval_data = self.X_test.copy()
            eval_data[self.config.target] = self.y_test
            result = mlflow.models.evaluate(
                self.model_info.model_uri,
                eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            self.metrics = result.metrics

    def register_model(self) -> int | str:
        """Register the model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        latest_model_uri = f"models:/{latest_model_version.model_id}"

        result = mlflow.models.evaluate(
            latest_model_uri,
            self.eval_data,
            targets=self.config.target,
            model_type="classifier",
            evaluators=["default"],
        )
        metrics_old = result.metrics
        if self.metrics["f1_score"] >= metrics_old["f1_score"]:
            logger.info("Current model performs better. Returning True.")
            return True
        else:
            logger.info("Current model does not improve over latest. Returning False.")
            return False
