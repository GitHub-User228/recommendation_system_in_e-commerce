import warnings

warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from scripts import logger
from scripts.components.base import BaseComponent
from scripts.settings import get_features_generator_component_config


class FeaturesGeneratorComponent(BaseComponent):
    """
    The `FeaturesGeneratorComponent` class is responsible for generating
    and saving user and item features from a dataset.

    Attributes:
        config (FeaturesGeneratorComponentConfig):
            Configuration for the features generator component.
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
        Initializes the FeaturesGeneratorComponent class.

        Args:
            is_testing (bool):
                Whether the component is being used for testing.
            is_airflow (bool):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
        """
        self.config = get_features_generator_component_config()
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def read_parquet(self, filename: str) -> pd.DataFrame:
        """
        Reads a Parquet file from the configured source path and
        returns a DataFrame.

        Args:
            filename (str):
                The name of the Parquet file to read.

        Returns:
            pd.DataFrame:
                A DataFrame containing the data.
        """
        df = pd.read_parquet(Path(self.config.source_path, filename))
        logger.info(f"Read {filename} from {self.config.source_path}.")
        return df

    def save_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """
        Writes a DataFrame to a Parquet file at the configured
        destination path.

        Args:
            df (pd.DataFrame):
                A DataFrame to write to Parquet.
            filename (str):
                The name of the Parquet file to write.
        """
        cat = "not_testing"
        if self.is_testing:
            cat = "testing"
        df.to_parquet(Path(self.config.destination_path[cat], filename))
        logger.info(
            f"Saved {filename} to {self.config.destination_path[cat]}."
        )

    def _col1_per_col2(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        col_name: str,
        lag: int = 0,
    ) -> pd.DataFrame:
        """
        Calculates the number of times a value in column 1 occurs
        per value in column 2. Optionally, a lag can be specified, so
        that the calculation are performed since last `lag` days.

        Args:
            df (pd.DataFrame):
                A DataFrame to calculate the values from.
            col1 (str):
                The name of the column containing the values to count.
            col2 (str):
                The name of the column containing the values to count
                per value in column 1.
            col_name (str):

            lag (int, optional):
                The number of days to lag the calculation by.
                Defaults to 0.

        Returns:
            pd.DataFrame:
                A DataFrame with the calculated values.
        """
        if lag == 0:
            return (
                df.groupby(col2)[col1]
                .count()
                .reset_index()
                .rename(columns={col1: col_name})
            )
        elif lag > 0:
            ref_date = (
                datetime.strptime(self.config.reference_date, "%Y-%m-%d")
                - timedelta(days=lag)
            ).strftime("%Y-%m-%d")
            return (
                df.query(f"{self.config.date_col} >= '{ref_date}'")
                .groupby(col2)[col1]
                .count()
                .reset_index()
                .rename(columns={col1: f"{col_name}_last_{lag}"})
            )
        else:
            raise ValueError(
                "Invalid lag value. It must be a non-negative integer."
            )

    def _days_since_first_last_interaction(
        self, df: pd.DataFrame, col: str, col_name1: str, col_name2: str
    ) -> pd.DataFrame:
        """
        Calculates the number of days since the first and last
        interaction for user or item.

        Args:
            df (pd.DataFrame):
                A DataFrame to calculate the values from.
            col (str):
                The name of the column containing the values to count.
            col_name1 (str):
                The name of the column containing the number of days since
                the first interaction.
            col_name2 (str):
                The name of the column containing the number of days since
                the last interaction.

        Returns:
            pd.DataFrame:
                A DataFrame with the calculated values.
        """
        ref_date = pd.to_datetime(self.config.reference_date)
        df2 = (
            df.groupby(col)[self.config.date_col]
            .agg(["min", "max"])
            .reset_index()
        )
        df2[col_name1] = (ref_date - df2["min"]).dt.days
        df2[col_name2] = (ref_date - df2["max"]).dt.days
        df2.drop(columns=["min", "max"], inplace=True)
        return df2

    def _generate_default_features(
        self, df: pd.DataFrame, for_user: bool = False
    ) -> pd.DataFrame:
        """
        Generates the default features.

        Args:
            df (pd.DataFrame):
                A DataFrame to generate the features from.
            for_user (bool, optional):
                Whether the features are for users or items.
                Defaults to False.

        Returns:
            pd.DataFrame:
                A DataFrame with the generated features.
        """
        col1 = "user" if for_user else "item"
        col2 = "item" if for_user else "user"

        df2 = None

        for rating in df[self.config.rating_col].unique():

            # col1 per col2
            df2_ = self._col1_per_col2(
                df=df.query(f"{self.config.rating_col} == {rating}"),
                col1=self.config.fields_id[col2],
                col2=self.config.fields_id[col1],
                col_name=f"{col2}s_per_{col1}_r_{rating}",
            )

            # col1 per col2 with lags
            for lag in [28]:
                df2_ = df2_.merge(
                    self._col1_per_col2(
                        df=df.query(f"{self.config.rating_col} == {rating}"),
                        col1=self.config.fields_id[col2],
                        col2=self.config.fields_id[col1],
                        lag=lag,
                        col_name=f"{col2}s_per_{col1}_r_{rating}",
                    ),
                    on=self.config.fields_id[col1],
                    how="outer",
                )

            # days since the last and the first interaction
            df2_ = df2_.merge(
                self._days_since_first_last_interaction(
                    df=df.query(f"{self.config.rating_col} == {rating}"),
                    col=self.config.fields_id[col1],
                    col_name1=f"days_since_{col1}_first_interaction_r_{rating}",
                    col_name2=f"days_since_{col1}_last_interaction_r_{rating}",
                ),
                on=self.config.fields_id[col1],
                how="outer",
            )

            # Dropping features with no data at all
            df2_.dropna(axis=1, how="all", inplace=True)

            df2 = (
                df2_
                if df2 is None
                else df2.merge(
                    df2_, on=self.config.fields_id[col1], how="outer"
                )
            )

        return df2

    def generate_user_features(self) -> None:
        """
        Generates and saves the user features.
        """

        if self.is_testing:
            subset = "train"
        else:
            subset = "all"

        # Read the events train data
        df = self.read_parquet(filename=self.config.events_filenames[subset])

        # Generating default features
        user_features = self._generate_default_features(df=df, for_user=True)

        # Saving the features
        self.save_parquet(
            df=user_features,
            filename=self.config.user_features_filename,
        )
        logger.info(f"Generated user features.")

    def generate_item_features(self) -> None:
        """
        Generates and saves the item features.
        """

        if self.is_testing:
            subset = "train"
        else:
            subset = "all"

        # Read the events train data
        df = self.read_parquet(filename=self.config.events_filenames[subset])

        # Generating default features
        item_features = self._generate_default_features(df=df, for_user=False)

        # Read item availability data
        df = self.read_parquet(
            filename=self.config.item_features_filenames["availability"]
        )

        # Add certain features related to item availability
        df = (
            df.groupby([self.config.fields_id["item"], "available"])["days"]
            .sum()
            .reset_index()
        )
        item_features = item_features.merge(
            df.query('available == "1"')
            .rename(columns={"days": "n_days_available"})
            .drop(columns="available"),
            on=self.config.fields_id["item"],
            how="left",
        )
        item_features = item_features.merge(
            df.query('available == "0"')
            .rename(columns={"days": "n_days_not_available"})
            .drop(columns="available"),
            on=self.config.fields_id["item"],
            how="left",
        )

        # Read item category data
        df = self.read_parquet(
            filename=self.config.item_features_filenames["category"]
        )

        # Add certain features related to item category
        df = (
            df.sort_values(
                by=[self.config.fields_id["item"], self.config.date_col]
            )
            .groupby(self.config.fields_id["item"])
            .tail(1)[[self.config.fields_id["item"], "category_level"]]
        )
        item_features = item_features.merge(
            df,
            on=self.config.fields_id["item"],
            how="left",
        )

        # Saving the features
        self.save_parquet(
            df=item_features,
            filename=self.config.item_features_filename,
        )
        logger.info(f"Generated item features.")

    def generate(self, log: bool = False) -> None:
        """
        Generates all features.

        Args:
            log (bool):
                Whether to log the features generation data.
                Defaults to False.
        """

        self.generate_item_features()
        self.generate_user_features()

        if log:
            self.log()

        logger.info(f"Finished features generation.")
