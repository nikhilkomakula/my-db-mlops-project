"""Data processing module for Marvel characters dataset."""

import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from marvel_characters.config import ProjectConfig


class DataProcessor:
    """Handles data loading and preprocessing for the Marvel characters dataset."""

    def __init__(self, df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the dataset by handling missing values and encoding categorical features."""
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target

        # relabel height and weight columns
        self.df.rename(columns={"Height (m)": "Height"}, inplace=True)
        self.df.rename(columns={"Weight (kg)": "Weight"}, inplace=True)

        # handle missing values in Universe column
        self.df["Universe"].fillna("Unknown", inplace=True)

        # group small categories in Universe column
        counts = self.df["Universe"].value_counts()
        small_universes = counts[counts < 50].index
        self.df["Universe"] = self.df["Universe"].replace(small_universes, "Other")

        # convert Teams column to int
        self.df["Teams"] = self.df["Teams"].notna().astype("int")

        # fill missing values in Origin column
        self.df["Origin"] = self.df["Origin"].fillna("Unknown")
        self.df["Identity"] = self.df["Identity"].fillna("Unknown")

        # filter Identity column to keep only specific values
        self.df = self.df[self.df["Identity"].isin(["Secret", "Unknown", "Public"])]

        # filling missing values and grouping
        self.df["Gender"] = self.df["Gender"].fillna("Unknown")
        self.df["Gender"] = self.df["Gender"].where(self.df["Gender"].isin(["Male", "Female"]), other="Other")

        # process Marital Status column
        self.df.rename(columns={"Marital Status": "Marital_Status"}, inplace=True)
        self.df["Marital_Status"] = self.df["Marital_Status"].fillna("Unknown")
        self.df["Marital_Status"] = self.df["Marital_Status"].replace({"Widow": "Widowed"})
        self.df["Marital_Status"] = self.df["Marital_Status"].isin(
            ["Single", "Married", "Widowed", "Engaged", "Unknown"]
        )

        # add a new column "Magic"
        self.df["Magic"] = self.df["Origin"].str.lower().apply(lambda x: int("magic" in x))

        # add a new column "Mutant"
        self.df["Mutant"] = self.df["Origin"].str.lower().apply(lambda x: int("mutant" in x or "mutate" in x))

        # Normalize Origin
        def normalize_origin(x: str) -> str:
            x_lower = str(x).lower()
            if "human" in x_lower:
                return "Human"
            elif "mutate" in x_lower or "mutant" in x_lower:
                return "Mutant"
            elif "asgardian" in x_lower:
                return "Asgardian"
            elif "alien" in x_lower:
                return "Alien"
            elif "symbiote" in x_lower:
                return "Symbiote"
            elif "robot" in x_lower:
                return "Robot"
            elif "cosmic being" in x_lower:
                return "Cosmic Being"
            else:
                return "Other"

        self.df["Origin"] = self.df["Origin"].apply(normalize_origin)

        # filter Alive column to keep only specific values and convert to binary
        self.df = self.df[self.df["Alive"].isin(["Alive", "Dead"])]
        self.df["Alive"] = (self.df["Alive"] == "Alive").astype("int")

        # keep only relevant columns
        self.df = self.df[num_features + cat_features + [target] + ["PageID"]]

        # convert categorical features to category dtype
        for col in cat_features:
            self.df[col] = self.df[col].astype("category")

        # rename PageID to Id and convert to string
        if "PageID" in self.df.columns:
            self.df = self.df.rename(columns={"PageID": "Id"})
            self.df["Id"] = self.df["Id"].astype("str")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training and testing sets.

        :param test_size: Proportion of the dataset to include in the test split
        :param random_state: Random seed for reproducibility
        :return: A tuple containing the training and testing DataFrames
        """
        train_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the DataFrame to a Delta table in the specified catalog and schema.

        :param train_set: Training DataFrame to save
        :param test_set: Testing DataFrame to save
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.mc_train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.mc_test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable change data feed for the Delta tables."""
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.mc_train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.mc_test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic Marvel character data matching input DataFrame distributions with optional drift.

    Creates artificial dataset replicating statistical patterns from source columns including numeric,
    categorical, and datetime types. Supports intentional data drift for specific features when enabled.

    :param df: Source DataFrame containing original data distributions
    :param drift: Flag to activate synthetic data drift injection
    :param num_rows: Number of synthetic records to generate
    :return: DataFrame containing generated synthetic data
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "Id":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if column in {"Height", "Weight"}:  # Handle physical attributes
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)
                # Ensure positive values for physical attributes
                synthetic_data[column] = np.maximum(0.1, synthetic_data[column])
            else:
                synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            synthetic_data[column] = pd.to_datetime(
                np.random.randint(min_date.value, max_date.value, num_rows)
                if min_date < max_date
                else [min_date] * num_rows
            )

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Convert relevant numeric columns to appropriate types
    float_columns = {"Height", "Weight"}
    for col in float_columns.intersection(df.columns):
        synthetic_data[col] = synthetic_data[col].astype(np.float64)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["Id"] = [str(timestamp_base + i) for i in range(num_rows)]

    if drift:
        # Skew the physical attributes to introduce drift
        drift_features = ["Height", "Weight"]
        for feature in drift_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] = synthetic_data[feature] * 1.5

        # Introduce bias in categorical features
        if "Gender" in synthetic_data.columns:
            synthetic_data["Gender"] = np.random.choice(["Male", "Female"], num_rows, p=[0.7, 0.3])

    return synthetic_data


def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching input DataFrame distributions with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
