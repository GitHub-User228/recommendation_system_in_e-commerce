import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from pathlib import Path
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, save_npz
from sklearn.preprocessing import LabelEncoder

from scripts import logger
from scripts.utils import save_pkl, read_pkl
from scripts.components.base import BaseComponent
from scripts.settings import get_matrix_builder_component_config


class MatrixBuilderComponent(BaseComponent):
    """
    Builds and saves encoders for user, item and item-related features
    IDs, user-items matrix.

    Attributes:
        config (MatrixBuilderComponentConfig):
            The configuration parameters for the MatrixBuilderComponent
            class.
        print_info (bool):
            Whether to print a matrix info (e.g. sparsity).
        is_testing (bool):
            Whether the component is being used for testing.
        is_airflow (bool):
            Whether the component is being used via Airflow.
            This is used to determine the host for the MLflow server.
    """

    def __init__(
        self,
        print_info: bool = False,
        is_testing: bool = True,
        is_airflow: bool = False,
    ) -> None:
        """
        Initializes the MatrixBuilder class.

        Args:
            print_info (bool):
                Whether to print a matrix info. Defaults to False.
            is_testing (bool):
                Whether the component is being used for testing.
            is_airflow (bool):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
        """
        self.config = get_matrix_builder_component_config()
        self.print_info = print_info
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def fit_encoders(self) -> None:
        """
        Fits and saves the encoders for user and item IDs based on the
        unique IDs found in the events_train dataset. The process is
        performed iteratively using a batch-wise approach.

        Raises:
            FileNotFoundError:
                If no parquet files are found for the events_train
                dataset.
        """

        if self.is_testing:
            cat = "testing"
            subset = "train"
        else:
            cat = "not_testing"
            subset = "all"

        # Initialize sets to store the ids
        user_ids = set()
        item_ids = set()

        # Iterate over batches
        for batch in pq.ParquetFile(
            Path(
                self.config.source_path,
                self.config.events_filenames[subset],
            )
        ).iter_batches(
            batch_size=self.config.batch_size,
            columns=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
        ):

            # Update the sets with unique user and item IDs
            user_ids.update(
                pa.compute.unique(
                    batch.column(self.config.fields_id["user"])
                ).to_pylist()
            )
            item_ids.update(
                pa.compute.unique(
                    batch.column(self.config.fields_id["item"])
                ).to_pylist()
            )
        logger.info("Processed all batches")

        # Fit and save the item id encoder
        encoder_item = LabelEncoder()
        encoder_item.fit(list(item_ids))
        logger.info("Fitted an encoder for item ID")
        save_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["item"],
            ),
            model=encoder_item,
        )

        # Fit the user id encoder
        encoder_user = LabelEncoder()
        encoder_user.fit(list(user_ids))
        logger.info("Fitted an encoder for user ID")
        save_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["user"],
            ),
            model=encoder_user,
        )

    def build_user_items_matrix(self) -> None:
        """
        Builds and saves the user-item matrix. The process is performed
        iteratively using a batch-wise approach.

        Raises:
            FileNotFoundError:
                If no parquet files are found for the events_train
                dataset.
        """

        if self.is_testing:
            cat = "testing"
            subset = "train"
        else:
            cat = "not_testing"
            subset = "all"

        # Read the encoders
        encoder_user = read_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["user"],
            )
        )
        encoder_item = read_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["item"],
            )
        )

        # Read the events data
        df = pd.read_parquet(
            Path(
                self.config.source_path,
                self.config.events_filenames[subset],
            )
        )

        # Transform user and item IDs
        df[self.config.fields_id["user"]] = encoder_user.transform(
            df[self.config.fields_id["user"]]
        )
        df[self.config.fields_id["item"]] = encoder_item.transform(
            df[self.config.fields_id["item"]]
        )

        # Sort rows so that later events between user and item are lower
        df.sort_values(
            by=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
                self.config.date_col,
            ],
            inplace=True,
        )

        # Apply exp weighted sum averaging for each user item pair
        df = (
            df.groupby(
                [self.config.fields_id["user"], self.config.fields_id["item"]]
            )[self.config.rating_col]
            .ewm(span=2)
            .sum()
            .reset_index()[
                [
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                    self.config.rating_col,
                ]
            ]
        )

        # Build the user-items matrix
        user_items_matrix = csr_matrix(
            (
                df[self.config.rating_col].values,
                (
                    df[self.config.fields_id["user"]].values,
                    df[self.config.fields_id["item"]].values,
                ),
            ),
            dtype=np.float32,
        )
        logger.info("Build user-items sparse matrix")

        # Save the user_items matrix
        save_npz(
            file=Path(
                self.config.destination_path[cat],
                self.config.user_items_matrix_filename,
            ),
            matrix=user_items_matrix,
        )

        if self.print_info:
            n_users = user_items_matrix.shape[0]
            n_items = user_items_matrix.shape[1]
            n_events = user_items_matrix.nnz
            sparsity = round((1 - n_events / (n_users * n_items)) * 100, 4)
            logger.info(
                f"user-items matrix info: "
                f"number of users - {n_users}, "
                f"number of items - {n_items}, "
                f"number of events - {n_events}, "
                f"sparsity - {sparsity}%"
            )

    def build_item_features_matrix(self) -> None:
        """
        Builds and saves the item-features matrix
        """

        if self.is_testing:
            cat = "testing"
        else:
            cat = "not_testing"

        # Read the item ID encoder
        encoder_item = read_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["item"],
            )
        )

        # Read item category data
        df = pd.read_parquet(
            Path(
                self.config.source_path,
                self.config.item_features_filenames["category"],
            ),
            columns=[
                self.config.fields_id["item"],
                self.config.fields_id["item_category"],
            ],
        )

        # Filter data to only include items that exist in the encoder
        size = len(df)
        df = df[df[self.config.fields_id["item"]].isin(encoder_item.classes_)]
        rate = round(len(df) / size * 100, 2)
        logger.info(f"Left with {len(df)} ({rate}%) items")

        # Drop duplicates if any
        size = len(df)
        df.drop_duplicates(
            subset=[
                self.config.fields_id["item"],
                self.config.fields_id["item_category"],
            ],
            inplace=True,
        )
        rate = round((size - len(df)) / size * 100, 2)
        logger.info(
            f"Dropped {rate}% of rows due to duplicates. {len(df)} rows left"
        )

        # Encode the item IDs
        df[self.config.fields_id["item"]] = encoder_item.transform(
            df[self.config.fields_id["item"]]
        )

        # Fit and save an encoder to encode the feature IDs
        encoder = LabelEncoder()
        df[self.config.fields_id["item_category"]] = encoder.fit_transform(
            df[self.config.fields_id["item_category"]]
        )
        logger.info(f"Fitted an encoder for item category")
        save_pkl(
            path=Path(
                self.config.destination_path[cat],
                self.config.encoders_filenames["item_category"],
            ),
            model=encoder,
        )

        # Build the item-feature matrix
        item_features_matrix = csr_matrix(
            (
                np.ones(len(df)),
                (
                    df[self.config.fields_id["item"]],
                    df[self.config.fields_id["item_category"]],
                ),
            ),
            dtype=np.uint8,
            shape=(
                len(encoder_item.classes_),
                len(encoder.classes_),
            ),
        )
        logger.info(f"Built item-category sparse matrix")

        # Save the item-feature matrix
        save_npz(
            file=Path(
                self.config.destination_path[cat],
                self.config.item_features_matrix_filename,
            ),
            matrix=item_features_matrix,
        )

        if self.print_info:

            # Omit items with no features
            item_features_matrix = item_features_matrix.tolil()

            n_items = sum(item_features_matrix.getnnz(axis=1) > 0)
            n_features = item_features_matrix.shape[1]
            data_count = item_features_matrix.nnz
            sparsity = round(
                (1 - data_count / (n_items * n_features)) * 100, 4
            )
            logger.info(
                f"item-category matrix info: "
                f"number of items - {n_items}, "
                f"number of features - {n_features}, "
                f"number of data points - {data_count}, "
                f"sparsity - {sparsity}%"
            )

    def build(self, log: bool = False) -> None:
        """
        Builds and saves the user-items matrix.
        """

        # Fit user ID encoder and itemID encoder
        self.fit_encoders()

        # Build user-item matrix
        self.build_user_items_matrix()

        # Build item-feature matrix
        self.build_item_features_matrix()

        # Log the run if requested
        if log:
            self.log()

        logger.info("Finished building all matrices")
