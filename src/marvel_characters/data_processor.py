"""Data processing module for Marvel characters dataset."""

import pandas as pd
from pyspark.sql import SparkSession
from marvel_characters.config import ProjectConfig

class DataProcessor:
    """
    Handles data loading and preprocessing for the Marvel characters dataset.
    """
    def __init__(self, df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = df
        self.config = config
        self.spark = spark
        
    def preprocess(self) -> None:
        """
        Preprocess the dataset by handling missing values and encoding categorical features.
        """
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target
        
        self.df.rename(columns={"Height (m)": "Height"}, inplace=True)
        self.df.rename(columns={"Weight (kg)": "Weight"}, inplace=True)
        
        self.df["Universe"].fillna("Unknown", inplace=True)
        counts = self.df["Universe"].value_counts()
        small_universes = counts[counts < 50].index
        print(">> small_universes", small_universes)