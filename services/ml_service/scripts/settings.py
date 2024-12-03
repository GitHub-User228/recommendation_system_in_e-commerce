from pathlib import Path
import pyarrow.parquet as pq
from functools import lru_cache
from pydantic import BaseModel, Field, model_validator

from scripts.env import env_vars
from scripts.utils import read_yaml

# Reading config data
config = read_yaml(Path(env_vars.config_dir, "config.yaml"))


class BaseStoreConfig(BaseModel):
    """
    Defines the `BaseStoreConfig` class, which represents
    common configuration parameters for stores with recs.
    """

    class Config:
        allow_mutation = False
        extra = "forbid"

    path: Path = Field(
        description="Path to the folder with recs data.",
    )
    user_id: str = Field(
        description="Column with user IDs",
    )
    item_id: str = Field(
        description="Column with item IDs",
    )
    similar_item_id: str = Field(
        description="Column with similar item IDs",
    )
    score_col_pattern: str = Field(
        description="Pattern for score column",
    )

    @model_validator(mode="after")
    def check_path(self) -> "BaseStoreConfig":
        """
        Checks if the path exists and if there are parquet files
        in the path.

        Returns:
            BaseStoreConfig:
                The updated configuration.
        """

        # Check if the path exists
        if not self.path.exists():
            raise FileNotFoundError(
                f"The path {self.path} with recs does not exist."
            )
        # Check if there are parquet files in the path
        files = list(self.path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files with recs found within {self.path}"
            )

        return self


class SimilarItemsStoreConfig(BaseStoreConfig):
    """
    Config for similar items store.
    """

    path: Path = Field(
        default=env_vars.online_recs_dir,
        description="Path to the folder with similar items.",
    )

    @model_validator(mode="after")
    def check_columns(self) -> "SimilarItemsStoreConfig":
        """
        Checks if the columns of the parquet files are the expected ones.

        Returns:
            SimilarItemsStoreConfig:
                The updated configuration.
        """

        # Columns that are expected to be in the parquet files
        cols = {self.item_id, self.similar_item_id}

        for file in self.path.glob("*.parquet"):

            columns = set(pq.ParquetFile(file).schema.names)

            # Checking if the columns of the parquet files are the
            # expected ones
            if columns.issubset(cols):
                raise ValueError(
                    f"The similar recs file {file} is expected to have {cols} "
                    f"columns, got {columns}"
                )

            # Checking if there is a single column with the score
            if sum([self.score_col_pattern in col for col in columns]) != 1:
                raise ValueError(
                    f"The online recs file {file} must have a single column "
                    f"with the pattern {self.score_col_pattern}"
                )

        return self


class OfflineRecsStoreConfig(BaseStoreConfig):
    """
    Config for offline recommendations store.
    """

    path: Path = Field(
        default=env_vars.offline_recs_dir,
        description="Path to the folder with offline recommendations.",
    )

    @model_validator(mode="after")
    def check_columns(self) -> "OfflineRecsStoreConfig":
        """
        Checks if the columns of the parquet files are the expected ones.

        Returns:
            OfflineRecsStoreConfig:
                The updated configuration.
        """

        # Columns that are expected to be in the parquet files
        cols = {self.user_id, self.item_id}

        for file in self.path.glob("*.parquet"):

            columns = set(pq.ParquetFile(file).schema.names)

            # Checking if the columns of the parquet files are the
            # expected ones
            if columns.issubset(cols):
                raise ValueError(
                    f"The offline recs file {file} is expected to have {cols} "
                    f"columns, got {columns}"
                )

        return self


class PopularItemsStoreConfig(BaseStoreConfig):
    """
    Config for popular items store.
    """

    path: Path = Field(
        default=env_vars.popular_recs_dir,
        description="Path to the folder with popular items.",
    )

    @model_validator(mode="after")
    def check_columns(self) -> "PopularItemsStoreConfig":
        """
        Checks if the columns of the parquet files are the expected ones.

        Returns:
            PopularItemsStoreConfig:
                The updated configuration.
        """

        # Columns that are expected to be in the parquet files
        cols = {self.item_id}

        for file in self.path.glob("*.parquet"):

            columns = set(pq.ParquetFile(file).schema.names)

            # Checking if the columns of the parquet files are the
            # expected ones
            if columns.issubset(cols):
                raise ValueError(
                    f"The popular recs file {file} is expected to have {cols} "
                    f"columns, got {columns}"
                )

        return self


class EventsStoreConfig(BaseModel):
    """
    Config for events store
    """

    class Config:
        extra = "forbid"

    max_events_per_user: int = Field(
        ge=1,
        le=10,
        description="Maximum number of events to store for each user",
    )

    events_filename: str = Field(
        description="Name of the file with events.",
    )

    test_events_filename: str = Field(
        description="Name of the file with test events.",
    )

    is_testing: bool = Field(
        description="Whether the events store is used for testing",
    )


class EventsStorePutQueryParams(BaseModel):
    """
    Defines the parameters for a put request to the events store.
    Used for validation purposes.
    """

    class Config:
        extra = "forbid"

    user_id: int = Field(
        description="User ID",
    )
    item_id: int = Field(
        description="Item ID",
    )


class EventsStoreGetQueryParams(BaseModel):
    """
    Defines the parameters for a get request to the events store.
    Used for validation purposes.
    """

    class Config:
        extra = "forbid"

    user_id: int = Field(
        description="User ID",
    )

    k: int = Field(
        ge=1,
        le=10,
        description="Number of last events to return at most.",
    )


class QueryParams(BaseModel):
    """
    Defines the parameters for a query to the service.
    Used for validation purposes.
    """

    class Config:
        extra = "forbid"

    user_id: int = Field(
        description="User ID",
    )

    k: int = Field(
        ge=config["n_recs"]["min"],
        le=config["n_recs"]["max"],
        description="Number of items to recommend at most.",
    )


class RequestRateLimitSettings(BaseModel):
    """
    Settings for global request rate limiting.
    """

    class Config:
        extra = "forbid"

    times: int = Field(
        ge=1,
        description="Maximum number of requests allowed within a time window",
    )

    seconds: int = Field(
        ge=1,
        description="Time window in seconds",
    )


class GroupsRates(BaseModel):
    """
    Class to define the rates of different user groups when generating
    test data.
    """

    class Config:
        extra = "forbid"

    old: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of the old users with no data in events store",
    )

    old_with_events: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of the old users with data in events store",
    )

    new: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of the new users with no data in events store",
    )

    new_with_events: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of the new users with data in events store",
    )

    invalid: float = Field(
        ge=0.0,
        le=1.0,
        description="Rate of the invalid requests",
    )

    @model_validator(mode="after")
    def check_sum(self) -> "GroupsRates":
        """
        Checks if the rates sum up to 1.
        """
        rate_sum = (
            self.old
            + self.old_with_events
            + self.new
            + self.new_with_events
            + self.invalid
        )
        if rate_sum != 1.0:
            raise ValueError(f"The rates must sum up to 1.0, got {rate_sum}")

        return self


class TesterConfig(BaseModel):
    """
    Config for tester.
    """

    class Config:
        extra = "forbid"

    n_requests: int = Field(
        ge=1,
        le=1000,
        description="Number of requests to generate.",
    )

    delay: float = Field(
        ge=0.0,
        le=10.0,
        description="Delay between requests in seconds.",
    )

    multiple_ips: bool = Field(
        description="Whether to use multiple IP addresses.",
    )

    shuffle_requests: bool = Field(
        description="Whether to shuffle the requests.",
    )

    random_state: int = Field(
        description="Random state for reproducibility.",
    )

    groups_rate: GroupsRates = Field(
        description="Rate of different groups of requests.",
    )

    test_events_filename: str = Field(
        description="Name of the file with test events.",
    )


@lru_cache
def get_request_rate_limit_settings(
    endpoint_name: str, per_ip: bool = False
) -> RequestRateLimitSettings:
    """
    Returns the request rate limit settings.

    Args:
        endpoint_name (str):
            Name of the endpoint.
        per_ip (bool):
            Whether to use per-IP rate limiting. Defaults to False.
    """
    if per_ip:
        return RequestRateLimitSettings(
            **config["request_rate_limit"][endpoint_name]["per_ip"]
        )
    else:
        return RequestRateLimitSettings(
            **config["request_rate_limit"][endpoint_name]["global"]
        )


@lru_cache
def get_similar_items_store_config() -> SimilarItemsStoreConfig:
    """
    Returns the similar items store configuration.
    """
    return SimilarItemsStoreConfig(**config["base_store"])


@lru_cache
def get_popular_items_store_config() -> PopularItemsStoreConfig:
    """
    Returns the popular items store configuration.
    """
    return PopularItemsStoreConfig(**config["base_store"])


@lru_cache
def get_offline_recs_store_config() -> OfflineRecsStoreConfig:
    """
    Returns the offline recommendations store configuration.
    """
    return OfflineRecsStoreConfig(**config["base_store"])


@lru_cache
def get_events_store_config() -> EventsStoreConfig:
    """
    Returns the events store configuration.
    """
    return EventsStoreConfig(**config["events_store"])


@lru_cache
def get_tester_config() -> TesterConfig:
    """
    Returns the test data generator configuration.
    """
    tester_config = config["tester"]
    tester_config["groups_rate"] = GroupsRates(**tester_config["groups_rate"])
    return TesterConfig(
        **tester_config,
        test_events_filename=config["events_store"]["test_events_filename"],
    )
