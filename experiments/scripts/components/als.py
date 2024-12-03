import warnings

warnings.filterwarnings("ignore")

import argparse
import cloudpickle
import pandas as pd
from pathlib import Path
from typing import Literal
from scipy.sparse import load_npz

from scripts import logger
from scripts import models
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_als_model_component_config


class ALSModelComponent(BaseModelComponent):
    """
    Implements an Alternating Least Squares (ALS) model component for
    a recommendation system.

    Attributes:
        config (ALSModelComponentConfig):
            Configuration for the ALS model component.
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
        Initializes the ALSModelComponent class with the configuration
        settings.

        Args:
            is_testing (bool, optional):
                Whether the component is being used for testing.
                Defaults to True.
            is_airflow (bool, optional):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
                Defaults to False.
        """
        self.config = get_als_model_component_config()
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Fits an Alternating Least Squares (ALS) model to the user-item
        interaction data, saves the trained model and generated
        similar items data.
        """

        # Reading the encoders and user_items matrix
        cat = "testing" if self.is_testing else "not_testing"
        item_encoder = read_pkl(
            Path(
                self.config.source_path2[cat],
                self.config.encoders_filenames["item"],
            )
        )
        user_encoder = read_pkl(
            Path(
                self.config.source_path2[cat],
                self.config.encoders_filenames["user"],
            )
        )
        user_items_matrix = load_npz(
            Path(
                self.config.source_path2[cat],
                self.config.user_items_matrix_filename,
            )
        )

        # Creating the ALS model and fitting it to the data
        model = models.ALS(
            min_users_per_item=self.config.min_users_per_item,
            factors=self.config.factors,
            iterations=self.config.iterations,
            regularization=self.config.regularization,
            alpha=self.config.alpha,
            filter_already_liked_items=self.config.filter_already_liked_items,
            calculate_training_loss=self.config.calculate_training_loss,
            random_state=self.config.random_state,
        )
        model.fit(
            user_items_matrix=user_items_matrix,
            item_id_encoder=item_encoder,
            user_id_encoder=user_encoder,
        )
        logger.info("Trained ALS model")

        # Saving the trained ALS model using cloudpickle
        with open(
            Path(
                self.config.destination_path[cat],
                self.config.model_filename,
            ),
            "wb",
        ) as f:
            cloudpickle.register_pickle_by_value(models)
            cloudpickle.dumps(models.BaseRecommender)
            cloudpickle.dumps(models.ALS)
            cloudpickle.dump(model, f)
        logger.info("Saved ALS model")

        # Generate similar items if case is not testing
        if not self.is_testing:

            # Get items that are not available
            reference_date = pd.to_datetime(self.config.reference_date)
            filter_items = (
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
                .query(f"{self.config.date_col} <= @reference_date")
                .groupby(self.config.fields_id["item"])
                .head(1)
                .query('available == "0"')[self.config.fields_id["item"]]
                .unique()
                .tolist()
            )

            # Generating similar items
            similar_items_df = model.get_similar_items(
                max_similar_items=self.config.max_similar_items,
                item_id_col=self.config.fields_id["item"],
                item_id_col_similar=f'similar_{self.config.fields_id["item"]}',
                score_col=self.config.score_col,
                filter_items=filter_items,
            )

            # Saving data
            similar_items_df.to_parquet(
                Path(
                    self.config.destination_path[cat],
                    self.config.similar_items_filename,
                )
            )
            logger.info("Saved similar items data")

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
            filter_items = (
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
                .unique()
                .tolist()
            )

            # Read the user_items matrix
            user_items_matrix = load_npz(
                Path(
                    self.config.source_path2[cat],
                    self.config.user_items_matrix_filename,
                )
            )
            logger.info("Loaded user_items_matrix")

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

            # Loading the trained ALS model
            model = read_pkl(
                Path(
                    self.config.destination_path[cat],
                    self.config.model_filename,
                )
            )

            # Recommendations
            recommendations = model.recommend(
                user_ids=user_ids,
                user_items_matrix=user_items_matrix,
                n_recommendations=self.config.n_recommendations,
                user_id_col=self.config.fields_id["user"],
                item_id_col=self.config.fields_id["item"],
                score_col=self.config.score_col,
                filter_items=filter_items,
            )

            # Save recommendations
            recommendations.to_parquet(
                Path(
                    self.config.destination_path[cat],
                    self.config.recommendations_filenames[subset],
                )
            )
            logger.info(f"Saved recommendations for subset '{subset}'")
