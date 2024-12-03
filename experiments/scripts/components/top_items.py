import warnings

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.sparse import load_npz

from scripts import logger
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_top_items_model_component_config


class TopItemsModelComponent(BaseModelComponent):
    """
    Implements a model that recommends the top N most popular items.

    Attributes:
        config (TopItemsModelComponentConfig):
            The configuration parameters for the TopItemsModelComponent
            class.
        is_testing (bool):
            Whether the component is being used for testing.
        is_airflow (bool):
            Whether the component is being used via Airflow.
            This is used to determine the host for the MLflow server.
    """

    def __init__(
        self, is_testing: bool = True, is_airflow: bool = False
    ) -> None:
        """
        Initializes the TopItemsModelComponent class with the
        configuration settings.

        Args:
            is_testing (bool, optional):
                Whether the component is being used for testing.
                Defaults to True.
            is_airflow (bool, optional):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
                Defaults to False.
        """
        self.config = get_top_items_model_component_config()
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Retrieves the top N most popular items
        """

        cat = "not_testing"
        if self.is_testing:
            cat = "testing"

        # Read item encoder
        item_encoder = read_pkl(
            Path(
                self.config.source_path2[cat],
                self.config.encoders_filenames["item"],
            )
        )

        # Read user_items matrix
        popularity = load_npz(
            Path(
                self.config.source_path2[cat],
                self.config.user_items_matrix_filename,
            )
        )

        # Remove rating less than 2 (i.e. only viewed items)
        popularity.data[popularity.data < self.config.min_rating] = 0
        popularity.eliminate_zeros()

        # Calculate item popularity
        popularity = popularity.getnnz(axis=0) / popularity.nnz

        # Retrieve the most popular items
        top_items = np.argpartition(
            -popularity,
            kth=self.config.top_n_items,
        )[: self.config.top_n_items]
        top_scores = popularity[top_items].tolist()

        # Convert the top items to a dataframe
        df = pd.DataFrame(
            {
                self.config.fields_id["item"]: item_encoder.inverse_transform(
                    top_items
                ),
                self.config.score_col: top_scores,
            }
        )
        logger.info(
            f"Retrieved top {self.config.top_n_items} most popular items"
        )

        # Save the data
        df.to_parquet(
            Path(
                self.config.destination_path[cat],
                self.config.top_items_filename,
            )
        )
        logger.info("Saved top items data")

    def recommend(self) -> None:
        """
        Generate recommendations.
        """

        if self.is_testing:
            cat = "testing"
            subsets = ["target", "test"]
            reference_date = pd.to_datetime(self.config.train_test_split_date)
        else:
            cat = "not_testing"
            subsets = ["all"]
            reference_date = pd.to_datetime(self.config.reference_date)

        for subset in subsets:

            # Get items that are not available
            filter_items = set(
                pd.read_parquet(
                    Path(
                        self.config.source_path,
                        self.config.item_features_filenames["availability"],
                    ),
                )
                .sort_values(
                    by=[self.config.fields_id["item"], self.config.date_col],
                    ascending=[1, 0],
                )
                .query(f"{self.config.date_col} < @reference_date")
                .groupby(self.config.fields_id["item"])
                .head(1)
                .query('available == "0"')[self.config.fields_id["item"]]
            )

            # Retrieve users
            if subset in ["test", "target"]:
                user_ids = (
                    pd.read_parquet(
                        Path(
                            self.config.source_path,
                            self.config.events_filenames[subset],
                        ),
                        columns=[self.config.fields_id["user"]],
                    )[self.config.fields_id["user"]]
                    .unique()
                    .tolist()
                )
            elif subset == "all":
                user_ids = read_pkl(
                    Path(
                        self.config.source_path2[cat],
                        self.config.encoders_filenames["user"],
                    )
                ).classes_.tolist()
            logger.info(f"Retrieved user IDs for subset '{subset}'")

            # Read the top items data
            top_items = pd.read_parquet(
                Path(
                    self.config.destination_path[cat],
                    self.config.top_items_filename,
                )
            )
            logger.info("Loaded top items data")

            # Filter top items and keep only n_recommendations items at most
            top_items = top_items[~top_items.isin(filter_items)]
            top_items = top_items.iloc[
                : min(self.config.n_recommendations, len(top_items))
            ]
            logger.info(f"Filtered out unavailable items")

            # Recommendations
            recommendations = pd.concat(
                [top_items] * len(user_ids), ignore_index=True
            )
            recommendations[self.config.fields_id["user"]] = np.repeat(
                user_ids, len(top_items)
            )

            # Save recommendations
            recommendations.to_parquet(
                Path(
                    self.config.destination_path[cat],
                    self.config.recommendations_filenames[subset],
                )
            )
            logger.info(f"Saved recommendations for subset '{subset}'")
