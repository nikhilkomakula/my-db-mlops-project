# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import yaml
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from IPython.display import display
from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
logger.info("Configuration loaded successfully.")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# instantiate spark session
spark = SparkSession.builder.getOrCreate()

# load data
filepath = "../data/marvel_characters_dataset.csv"
df = pd.read_csv(filepath)

logger.info(f"Data loaded from {filepath} with shape {df.shape}")
logger.info(f"Columns: {df.columns.tolist()}")
logger.info(f"Target Column: {config.target}")
logger.info(f"Value Counts: {df[config.target].value_counts().to_dict()}")
display(df.head(5))

# COMMAND ----------
dataprocessor = DataProcessor(df=df, config=config, spark=spark)
dataprocessor.preprocess()
# COMMAND ----------
