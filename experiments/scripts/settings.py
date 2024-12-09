import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List
from pydantic import BaseModel, Field, model_validator

from scripts.env import env_vars
from scripts.utils import read_yaml


# Reading config data
config = read_yaml(Path(env_vars.config_dir, "components.yaml"))
dates = read_yaml(Path(env_vars.config_dir, "dates.yaml"))
airflow = read_yaml(Path(env_vars.config_dir, "airflow.yaml"))
exp_name = read_yaml(Path(env_vars.config_dir, "experiment_name.yaml"))


def inherit_field(
    base_model: BaseModel, field_name: str, **overrides
) -> Field:
    """
    Inherits a field from a base model and applies any overrides.

    Args:
        base_model (BaseModel):
            The base model to inherit the field from.
        field_name (str):
            The name of the field to inherit.
        **overrides:
            Any overrides to apply to the inherited field.

    Returns:
        Field:
            The inherited field with any overrides applied.
    """
    base_field = base_model.model_fields[field_name]
    return Field(
        default=overrides.get("default", base_field.default),
        description=base_field.description,
        **{k: v for k, v in overrides.items() if k != "default"},
    )


class AirflowConfig(BaseModel):
    """
    Defines the `AirflowConfig` class.
    """

    start_date: str = Field(
        description="Start date for the DAG in 'YYYY-MM-DD' format.",
    )

    schedule_interval: str = Field(
        description="Schedule interval for the DAG.",
    )

    dag_id: str = Field(
        description="ID of the DAG.",
    )


@lru_cache
def get_airflow_config() -> AirflowConfig:
    """
    Retrieves the `AirflowConfig` object.
    """
    return AirflowConfig(**airflow)


class BaseConfig(BaseModel):
    """
    Defines the `BaseConfig` class, which represents
    common configuration parameters for all components.
    """

    experiment_name: str = Field(
        default=exp_name["experiment_name"],
        description="Name of the experiment.",
    )

    uri: str = Field(
        default=(
            f"http://{env_vars.mlflow_server_host}:"
            f"{env_vars.mlflow_server_port}"
        ),
        description="URI of the MLflow registry and tracking servers.",
    )

    source_path: Dict[str, Path] | Path = Field(
        description="Path(s) to the source data.",
    )
    destination_path: Dict[str, Path] | Path = Field(
        description="Path(s) to the destination data.",
    )

    date_col: str = "timestamp"
    rating_col: str = "rating"

    fields_id: Dict[str, str] = Field(
        default={
            "item": "itemid",
            "user": "visitorid",
            "item_category": "categoryid",
            "parent_item_category": "parentid",
        },
        description="Maps fields to corresponding names of ID column",
    )

    item_features_filenames: Dict[str, str] = Field(
        default={
            "category_tree": "category_tree.parquet",
            "category": "category.parquet",
            "availability": "availability.parquet",
        },
        description="Filenames of the item features data.",
    )

    events_filenames: Dict[str, str] = {
        "all": "events.parquet",
        "train": "events_train.parquet",
        "target": "events_target.parquet",
        "test": "events_test.parquet",
    }

    encoders_filenames: Dict[str, str] = {
        "item": "encoder_item.pkl",
        "user": "encoder_user.pkl",
        "item_category": "encoder_item_category.pkl",
    }

    user_items_matrix_filename: str = Field(
        default="user_items_matrix.npz",
        description="Filename of the user-items matrix.",
    )
    item_features_matrix_filename: str = Field(
        default="item_categories_matrix.npz",
        description="Filename of the item features matrix.",
    )

    user_features_filename: str = Field(
        default="user_features.parquet",
        description="Filename of the user features data.",
    )
    item_features_filename: str = Field(
        default="item_features.parquet",
        description="Filename of the item features data.",
    )
    top_items_filename: str = Field(
        default="top_items.parquet",
        description="Filename of the top items data.",
    )

    train_test_split_date: str = Field(
        default=dates["train_test_split_date"],
        description="Date for splitting the data into train and test sets.",
    )
    target_test_split_date: str = Field(
        default=dates["target_test_split_date"],
        description="Date for splitting the test data into target and test sets.",
    )

    reference_date: str = Field(
        default=dates["reference_date"],
        description="Reference date which is also a max date",
    )

    @model_validator(mode="after")
    def path_handler(self) -> "BaseConfig":
        """
        Validates path parameters

        Returns:
            BaseConfig:
                The updated configuration.
        """

        if isinstance(self.source_path, Path):
            # Checking if source_path exists
            if not self.source_path.exists():
                raise ValueError(
                    f"The source path {self.source_path} does not exist."
                )
            # Checking if source_path is not empty
            if not any(self.source_path.iterdir()):
                raise ValueError(
                    f"The source path {self.source_path} is empty."
                )
        else:
            for k, v in self.source_path.items():
                # Checking if source_path exists
                if not v.exists():
                    raise ValueError(
                        f"The source path {k} {v} does not exist."
                    )
                # Checking if source_path is not empty
                if not any(v.iterdir()):
                    raise ValueError(f"The source path {k} {v} is empty.")

        # Creating destination directory if not exists
        if isinstance(self.destination_path, Path):
            if not self.destination_path.exists():
                self.destination_path.mkdir(parents=True, exist_ok=True)
        else:
            for k, v in self.destination_path.items():
                if not v.exists():
                    v.mkdir(parents=True, exist_ok=True)

        return self


class PreprocessingComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `PreprocessingComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="source_path",
        default=Path(env_vars.artifacts_dir, "raw"),
    )
    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "preprocessing"),
    )
    category_tree_filename: str = Field(
        description="Filename of the data with category_tree.",
    )
    item_properties_filename: str = Field(
        description="Filename of the data with item properties.",
    )
    events_filename: str = Field(
        description="Filename of the events data.",
    )

    event_type_map: Dict[str, int] = Field(
        description="Map of event types to their corresponding rating.",
        default={
            "view": 1,
            "addtocart": 2,
            "transaction": 3,
        },
    )

    test_days: int = Field(
        ge=1,
        description="Number of days for test data.",
    )
    target_days: int = Field(
        ge=1,
        description="Number of days for target data within the test data.",
    )


class MatrixBuilderComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `MatrixBuilderComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(env_vars.artifacts_dir, "matrices", "testing"),
            "not_testing": Path(
                env_vars.artifacts_dir, "matrices", "not_testing"
            ),
        },
    )
    batch_size: int = Field(
        ge=1,
        description="Batch size for matrix construction.",
    )


class FeaturesGeneratorComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `FeaturesGeneratorComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(env_vars.artifacts_dir, "features", "testing"),
            "not_testing": Path(
                env_vars.artifacts_dir, "features", "not_testing"
            ),
        },
    )


class EDAComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `EDAComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    destination_path: Path = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default=Path(env_vars.artifacts_dir, "eda"),
    )

    category_tree_filename: str = Field(
        description="Filename of the data with category_tree.",
    )

    assets_path: Path = Field(
        default=Path("assets"),
        description="Directory for storing EDA assets.",
    )

    data_path: Path = Field(
        default=Path("data"),
        description="Directory for storing EDA files with data.",
    )

    quantiles: List[float] = Field(
        default=list(np.arange(0.1, 1, 0.1)) + [0.95, 0.99, 0.999],
        description="Quantiles for computing the distribution of the data.",
    )

    @model_validator(mode="after")
    def path_handler2(self) -> "EDAComponentConfig":
        """
        Handles path parameters

        Returns:
            EDAComponentConfig:
                The updated configuration.
        """

        # Creating assets directory if not exists
        self.assets_path = Path(self.destination_path, self.assets_path)
        if not self.assets_path.exists():
            self.assets_path.mkdir(parents=True, exist_ok=True)

        # Creating data directory if not exists
        self.data_path = Path(self.destination_path, self.data_path)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        return self


class ALSModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `ALSModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path2: Dict[str, Path] = Field(
        description="Secondary source paths",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(
                env_vars.artifacts_dir, "modelling/als", "testing"
            ),
            "not_testing": Path(
                env_vars.artifacts_dir, "modelling/als", "not_testing"
            ),
        },
    )

    model_filename: str = Field(
        default="als.pkl",
        description="Filename of the ALS model.",
    )
    similar_items_filename: str = Field(
        default="similar_items_als.parquet",
        description="Filename of the similar items data.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "all": "all.parquet",
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_als",
        description="Name of the score column",
    )

    # ALS model fields
    batch_size: int = Field(
        ge=1,
        description="Batch size.",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        description="Minimum number of users per item.",
    )
    max_similar_items: int = Field(
        ge=1,
        description="Maximum number of similar items to return.",
    )
    factors: int = Field(
        ge=1,
        description="Number of factors.",
    )
    iterations: int = Field(
        ge=1,
        description="Number of iterations",
    )
    regularization: float = Field(
        ge=1e-8,
        le=1.0,
        description="The regularization factor",
    )
    alpha: float | None = Field(
        ge=1.0,
        le=50,
        description="The weight to give to positive examples",
    )
    filter_already_liked_items: bool = Field(
        description="Whether to filter already liked items.",
    )
    calculate_training_loss: bool = Field(
        description="Whether to calculate the training loss.",
    )
    random_state: int = Field(
        description="Random state",
    )


class BPRModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `BPRModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path2: Dict[str, Path] = Field(
        description="Secondary source paths",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(
                env_vars.artifacts_dir, "modelling/bpr", "testing"
            ),
            "not_testing": Path(
                env_vars.artifacts_dir, "modelling/bpr", "not_testing"
            ),
        },
    )
    model_filename: str = Field(
        default="bpr.pkl",
        description="Filename of the BPR model.",
    )
    similar_items_filename: str = Field(
        default="similar_items_bpr.parquet",
        description="Filename of the similar items data.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "all": "all.parquet",
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_bpr",
        description="Name of the score column",
    )

    # BPR model fields
    batch_size: int = Field(
        ge=1,
        description="Batch size.",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        description="Minimum number of users per item.",
    )
    max_similar_items: int = Field(
        ge=1,
        description="Maximum number of similar items to return.",
    )
    factors: int = Field(
        ge=1,
        description="Number of factors",
    )
    iterations: int = Field(
        ge=1,
        description="Number of iterations",
    )
    learning_rate: float = Field(
        ge=1e-8,
        le=1.0,
        description="Learning rate",
    )
    regularization: float = Field(
        ge=1e-8,
        le=1.0,
        description="Regularization parameter",
    )
    filter_already_liked_items: bool = Field(
        description="Whether to filter already liked items.",
    )
    verify_negative_samples: bool = Field(
        description="Whether to verify negative samples.",
    )
    random_state: int = Field(
        description="Random state for the ALS model.",
    )


class Item2ItemModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `Item2ItemModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path2: Dict[str, Path] = Field(
        description="Secondary data source path",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(
                env_vars.artifacts_dir, "modelling/item2item", "testing"
            ),
            "not_testing": Path(
                env_vars.artifacts_dir, "modelling/item2item", "not_testing"
            ),
        },
    )
    model_filename: str = Field(
        default="item2item.pkl",
        description="Filename of the Item2ItemModel model.",
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "all": "all.parquet",
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="score_item2item",
        description="Name of the score column",
    )

    # Item2Item model fields
    batch_size: int = Field(
        ge=1,
        description="Batch size.",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )
    min_users_per_item: int = Field(
        ge=1,
        description="Minimum number of users per item.",
    )
    n_neighbors: int = Field(
        ge=1,
        description="Number of nearest neighbors",
    )
    n_components: int = Field(
        ge=1,
        description="Number of components of SVD model.",
    )
    similarity_criteria: str = Field(
        description="Similarity criteria for nearest neighbors",
    )


class TopItemsModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `TopItemsModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )
    source_path2: Dict[str, Path] = Field(
        description="Secondary data source path",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(
                env_vars.artifacts_dir, "modelling/top_items", "testing"
            ),
            "not_testing": Path(
                env_vars.artifacts_dir, "modelling/top_items", "not_testing"
            ),
        },
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "all": "all.parquet",
            "target": "target.parquet",
            "test": "test.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    score_col: str = Field(
        default="item_popularity",
        description="Name of the score column",
    )
    min_rating: int = Field(
        ge=1,
        le=3,
        description="Minimum rating for user-item event to be considered.",
    )
    top_n_items: int = Field(
        ge=1,
        description="Number of top items to retrieve",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations to make",
    )


class EnsembleModelComponentConfig(BaseConfig):
    """
    Represents configuration parameters for the
    `EnsembleModelComponent` class.
    """

    run_name: str = Field(
        description="Name of the mlflow run.",
    )

    base_models: List[str] = Field(description="List of base models aliases")

    all_data_path: Dict[str, str] = Field(default={})
    train_data_path: Dict[str, str] = Field(
        default={},
        description=(
            "Maps base models alias to the corresponding training data path"
        ),
    )
    test_data_path: Dict[str, str] = Field(
        default={},
        description=(
            "Maps base models alias to the corresponding test data path"
        ),
    )
    source_path2: Dict[str, Path] = Field(
        description="Secondary data source paths",
    )
    source_path3: Dict[str, Path] = Field(
        description="Other data source paths",
    )
    source_path4: Dict[str, Path] = Field(
        description="Other data source paths",
    )

    include_top_items: bool = Field(
        default=False,
        description="Whether to include top items in the ensemble",
    )
    destination_path: Dict[str, Path] = inherit_field(
        base_model=BaseConfig,
        field_name="destination_path",
        default={
            "testing": Path(
                env_vars.artifacts_dir, "modelling/ensemble", "testing"
            ),
            "not_testing": Path(
                env_vars.artifacts_dir, "modelling/ensemble", "not_testing"
            ),
        },
    )
    recommendations_filenames: Dict[str, str] = Field(
        default={
            "train_df": "train_df.parquet",
            "test_df": "test_df.parquet",
            "test": "test.parquet",
            "all": "recommendations.parquet",
        },
        description="Filenames of files with recommendations.",
    )
    metrics_filename: str = Field(
        default="metrics.yaml",
        description="Filename of the metrics file.",
    )
    feature_importances_filename: str = Field(
        default="feature_importances.yaml",
        description="Filename of the feature importances file.",
    )
    target_col: str = Field(
        default="target",
        description="Name of the target column",
    )
    score_col: str = Field(
        default="score_ensemble",
        description="Name of the score column",
    )
    n_recommendations: int = Field(
        ge=1,
        description="Number of recommendations",
    )

    # Sampling fields
    negative_samples_per_user: int = Field(
        default=5,
        ge=1,
        description="Number of negative samples per user",
    )
    sampling_seed: int = Field(
        default=42,
        description="Seed for sampling negative samples",
    )

    # Metrics
    target_metrics_order: Dict[str, bool] = Field(
        description="Order of the target metrics",
    )
    metrics_df_filename: str = Field(
        default="metrics_df.csv",
        description="Filename of the metrics dataframe file.",
    )

    # Candidate ensemble models
    ensemble_model_candidates: Dict[str, Any] = Field(
        description="Candidate ensemble models and their configurations",
    )
    model_filename: str = Field(
        default="ensemble.pkl",
        description="Filename of the ensemble model file.",
    )
    features_importance_filename: str = Field(
        default="feature_importances.yaml",
        description="Filename of the feature importances file.",
    )

    @model_validator(mode="after")
    def get_train_test_data_path(self) -> "EnsembleModelComponentConfig":
        """
        Retrieves the path to the training data

        Returns:
            EnsembleModelComponentConfig:
                The updated configuration.
        """

        config_aliases = {
            "als": get_als_model_component_config,
            "bpr": get_bpr_model_component_config,
            "item2item": get_item2item_model_component_config,
        }

        for alias in self.base_models:
            model_config = config_aliases[alias]()
            self.train_data_path[alias] = Path(
                model_config.destination_path["testing"],
                model_config.recommendations_filenames["target"],
            )
            self.test_data_path[alias] = Path(
                model_config.destination_path["testing"],
                model_config.recommendations_filenames["test"],
            )
            self.all_data_path[alias] = Path(
                model_config.destination_path["not_testing"],
                model_config.recommendations_filenames["all"],
            )

        return self


@lru_cache
def get_preprocessing_component_config() -> PreprocessingComponentConfig:
    """
    Returns the preprocessing component configuration.
    """
    return PreprocessingComponentConfig(
        **config["PreprocessingComponentConfig"]
    )


@lru_cache
def get_matrix_builder_component_config() -> MatrixBuilderComponentConfig:
    """
    Returns the matrix builder component configuration.
    """
    config1 = get_preprocessing_component_config()
    return MatrixBuilderComponentConfig(
        source_path=config1.destination_path,
        **config["MatrixBuilderComponentConfig"],
    )


@lru_cache
def get_features_generator_component_config() -> (
    FeaturesGeneratorComponentConfig
):
    """
    Returns the features generator component configuration.
    """
    config1 = get_preprocessing_component_config()
    return FeaturesGeneratorComponentConfig(
        source_path=config1.destination_path,
        **config["FeaturesGeneratorComponentConfig"],
    )


@lru_cache
def get_eda_component_config() -> EDAComponentConfig:
    """
    Returns the eda component configuration
    """
    config1 = get_preprocessing_component_config()
    return EDAComponentConfig(
        source_path=config1.destination_path,
        category_tree_filename=config1.category_tree_filename,
        **config["EDAComponentConfig"],
    )


@lru_cache
def get_als_model_component_config() -> ALSModelComponentConfig:
    """
    Returns the collaborative model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return ALSModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        **config["ALSModelComponentConfig"],
    )


@lru_cache
def get_bpr_model_component_config() -> BPRModelComponentConfig:
    """
    Returns the bpr model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return BPRModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        **config["BPRModelComponentConfig"],
    )


@lru_cache
def get_item2item_model_component_config() -> Item2ItemModelComponentConfig:
    """
    Returns the lightfm model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return Item2ItemModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        **config["Item2ItemModelComponentConfig"],
    )


@lru_cache
def get_top_items_model_component_config() -> TopItemsModelComponentConfig:
    """
    Returns the top items model component configuration.
    """
    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    return TopItemsModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        **config["TopItemsModelComponentConfig"],
    )


@lru_cache
def get_ensemble_model_component_config() -> EnsembleModelComponentConfig:
    """
    Returns the ensemble model component configuration.
    """

    config1 = get_preprocessing_component_config()
    config2 = get_matrix_builder_component_config()
    config3 = get_features_generator_component_config()
    config4 = get_top_items_model_component_config()
    return EnsembleModelComponentConfig(
        source_path=config1.destination_path,
        source_path2=config2.destination_path,
        source_path3=config3.destination_path,
        source_path4=config4.destination_path,
        **config["EnsembleModelComponentConfig"],
    )
