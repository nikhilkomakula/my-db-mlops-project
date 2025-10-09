"""Configuration file for Marvel characters project."""

import yaml
from typing import Any, Self
from pydantic import BaseModel

class ProjectConfig(BaseModel):
    """
    Represents project configuration parameters loaded from project_config.yml.
    
    Handles feature specifications, catalog details, and experiment parameters.
    Supports environment-specific configuration overrides.
    """
    
    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    experiment_name_basic: str
    experiment_name_custom: str
    
    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> Self:
        """
        Load and parse configuration settings from a YAML file.

        :param config_path: Path to the YAML configuration file
        :param env: Environment name to load environment-specific settings
        :return: ProjectConfig instance initialized with parsed configuration
        """
        if env not in ["dev", "acc", "prod"]:
            raise ValueError(f"Invalid environment: {env}. Must be 'dev', 'acc' or 'prod'.")
        
        with open(config_path, "r") as file:
            config_dict = yaml.safe_load(file)
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]
            return cls(**config_dict)
        
class Tags(BaseModel):
    """
    Represents tags for MLflow experiments.
    """
    git_sha: str
    branch: str
    run_id: str | None = None
    
    def to_dict(self) -> dict[str, str | None]:
        """
        Convert the Tags instance to a dictionary.
        """
        tags_dict: dict[str, str | None] = {}
        tags_dict["git_sha"] = self.git_sha
        tags_dict["branch"] = self.branch
        if self.run_id is not None:
            tags_dict["run_id"] = self.run_id
        return tags_dict