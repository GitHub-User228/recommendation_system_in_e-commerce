import os
import warnings

warnings.filterwarnings("ignore")

import argparse
import cloudpickle
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz


from scripts import models, logger
from scripts.utils import read_pkl
from scripts.components.base import BaseModelComponent
from scripts.settings import get_item2item_model_component_config


class Item2ItemModelComponent(BaseModelComponent):
    """
    Implements the Item2Item model component for the recommendation
    system.

    Attributes:
        config (Item2ItemModelComponentConfig):
            Configuration for the Item2Item model component.
    """

    def __init__(
        self,
        is_testing: bool = True,
        is_airflow: bool = False,
    ) -> None:
        """
        Initializes the Item2ItemModelComponent class with the
        configuration settings.

        Args:
            is_testing (bool):
                Whether the component is being used for testing.
            is_airflow (bool):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
        """
        self.config = get_item2item_model_component_config()
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def fit(self) -> None:
        """
        Fits the Item2Item model to the item-features data, saves
        the trained model.
        """

        # Reading the encoders and matrices
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
        item_features_matrix = load_npz(
            Path(
                self.config.source_path2[cat],
                self.config.item_features_matrix_filename,
            )
        )

        # Fit the model
        model = models.Item2ItemModel(
            min_users_per_item=self.config.min_users_per_item,
            n_neighbors=self.config.n_neighbors,
            n_components=self.config.n_components,
            similarity_criteria=self.config.similarity_criteria,
        )
        model.fit(
            item_features_matrix=item_features_matrix,
            user_items_matrix=user_items_matrix,
            item_id_encoder=item_encoder,
            user_id_encoder=user_encoder,
            n_jobs=os.cpu_count(),
        )

        # Save the model
        with open(
            Path(
                self.config.destination_path[cat],
                self.config.model_filename,
            ),
            "wb",
        ) as f:
            cloudpickle.register_pickle_by_value(models)
            cloudpickle.dumps(models.BaseRecommender)
            cloudpickle.dumps(models.Item2ItemModel)
            cloudpickle.dump(model, f)

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

        for subset in subsets:

            # Load user_items_matrix
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

            # Load the trained model
            model = read_pkl(
                Path(
                    self.config.destination_path[cat],
                    self.config.model_filename,
                )
            )

            model.recommend(
                user_ids=user_ids,
                user_items_matrix=user_items_matrix,
                n_recommendations=self.config.n_recommendations,
                user_id_col=self.config.fields_id["user"],
                item_id_col=self.config.fields_id["item"],
                score_col=self.config.score_col,
                batch_size=self.config.batch_size,
                save_path=Path(
                    self.config.destination_path[cat],
                    self.config.recommendations_filenames[subset],
                ),
                filter_items=filter_items,
            )
            logger.info(f"Saved recommendations for subset '{subset}'")
