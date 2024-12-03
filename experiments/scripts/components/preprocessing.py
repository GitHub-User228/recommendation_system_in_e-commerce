import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from scripts import logger
from scripts.env import env_vars
from scripts.utils import read_yaml, save_yaml
from scripts.components.base import BaseComponent
from scripts.settings import get_preprocessing_component_config


class PreprocessingComponent(BaseComponent):
    """
    Preprocesses the data to be used in the downstream tasks.

    Attributes:
        config (PreprocessingComponentConfig):
            The configuration parameters for the PreprocessingComponent
            class.
        print_info (bool):
            Whether to print info about data.
        is_testing (bool):
            Whether the component is being used for testing.
            This is False by default.
        is_airflow (bool):
            Whether the component is being used via Airflow.
            This is used to determine the host for the MLflow server.
    """

    def __init__(
        self, print_info: bool = False, is_airflow: bool = False
    ) -> None:
        """
        Initializes the PreprocessingComponent class.

        Args:
            print_info (bool):
                Whether to print info about data.
            is_airflow (bool):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
        """
        self.config = get_preprocessing_component_config()
        self.print_info = print_info
        self.is_testing = False
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def read_parquet(self, filename: str) -> pd.DataFrame:
        """
        Reads a Parquet file from the configured source path

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
        Writes a Pandas DataFrame to a Parquet file at the configured
        destination path.

        Args:
            df (pd.DataFrame):
                The DataFrame to write to Parquet.
            filename (str):
                The name of the Parquet file to write.
        """
        df.to_parquet(Path(self.config.destination_path, filename))
        logger.info(f"Saved {filename} to {self.config.destination_path}.")

    def print_table_info(self, df: pd.DataFrame) -> None:
        """
        Prints the info and statistics of a DataFrame.

        Args:
            df (pd.DataFrame):
                Pandas DataFrame to print info and statistics of.
        """
        print("-- info --")
        print(df.info(), end="\n\n")
        print("-- sample --")
        try:
            print(df.head().to_markdown(), end="\n\n")
        except:
            print(df.head(), end="\n\n")
        print("-- missing data count --")
        print(df.isnull().sum().to_markdown(), end="\n\n")

    def preprocess_category_tree(self) -> None:
        """
        Preprocesses `category_tree` data.
        """

        df = self.read_parquet(self.config.category_tree_filename)

        if self.print_info:
            print(
                "\n----- category_tree dataframe info before preprocessing -----\n"
            )

            # General info
            self.print_table_info(df)

            # Duplicates info
            print("-- duplicates count --")
            n_old = df.shape[0]
            n_dups = df.duplicated().sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"{n_dups} ({rate}%)\n\n")

        df[self.config.fields_id["item_category"]] = df[
            self.config.fields_id["item_category"]
        ].astype("uint16")
        df.drop_duplicates(inplace=True)
        self.save_parquet(
            df=df,
            filename=self.config.item_features_filenames["category_tree"],
        )

        if self.print_info:
            print(
                "\n----- category_tree dataframe info after preprocessing -----\n"
            )

            # General info
            self.print_table_info(df)

            # Duplicates info
            print("-- duplicates count --")
            n_old = df.shape[0]
            n_dups = df.duplicated().sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"{n_dups} ({rate}%)\n")

            # Number of unique categories
            print("-- number of categories --")
            n_categories = df[self.config.fields_id["item_category"]].nunique()
            print(f"Number of categories: {n_categories}\n\n")

    def preprocess_events(self) -> None:
        """
        Preprocesses `events` data.
        """

        # Read the parquet file with events
        df = self.read_parquet(self.config.events_filename)

        if self.print_info:
            print(
                "\n----- 'events' dataframe info before preprocessing -----\n"
            )

            # General info
            self.print_table_info(df)

            # Duplicates info
            print("-- duplicates count --")
            n_old = df.shape[0]
            n_dups = df.duplicated().sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"along all cols: {n_dups} ({rate}%)")
            n_dups = df.duplicated(
                subset=[
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                    "event",
                ]
            ).sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"along (user, item, event): {n_dups} ({rate}%)\n")

        # Drop `transactionid` column
        df.drop(columns=["transactionid"], inplace=True)

        # Convert `timestamp` to date-like column format
        df[self.config.date_col] = pd.to_datetime(
            df[self.config.date_col], unit="ms"
        ).astype(
            "datetime64[s]"
        )  #
        # Encode unique values in `event` column
        mapper = {
            "view": 1,
            "addtocart": 2,
            "transaction": 3,
        }
        df[self.config.rating_col] = df["event"].map(mapper)
        df.drop(columns=["event"], inplace=True)

        # Convert data types
        df = df.astype(
            {
                self.config.fields_id["user"]: "uint32",
                self.config.fields_id["item"]: "uint32",
                self.config.rating_col: "uint8",
            }
        )

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Retrieve the latest date
        self.config.reference_date = df[self.config.date_col].max()

        # Calculate split dates
        self.config.train_test_split_date = (
            self.config.reference_date
            - pd.Timedelta(days=self.config.test_days)
        ).strftime("%Y-%m-%d")
        self.config.target_test_split_date = (
            self.config.reference_date
            - pd.Timedelta(days=self.config.target_days)
        ).strftime("%Y-%m-%d")
        self.config.reference_date = self.config.reference_date.strftime(
            "%Y-%m-%d"
        )

        # Update config data
        config = read_yaml(Path(env_vars.config_dir, "dates.yaml"))
        config["reference_date"] = self.config.reference_date
        config["train_test_split_date"] = self.config.train_test_split_date
        config["target_test_split_date"] = self.config.target_test_split_date
        save_yaml(data=config, path=Path(env_vars.config_dir, "dates.yaml"))

        # Train test split
        date = self.config.train_test_split_date
        df_train = df[df[self.config.date_col] < date]
        df_test = df[df[self.config.date_col] >= date].query(
            f"{self.config.rating_col} >= 2"
        )

        # Retrieve users that are in boyj the train and test sets
        users_train = set(df_train[self.config.fields_id["user"]].unique())
        users_test = set(df_test[self.config.fields_id["user"]].unique())
        users_test_only = users_test - users_train

        # Target test split
        date = self.config.target_test_split_date
        df_target = df_test[df[self.config.date_col] < date]
        df_test2 = df_test[df[self.config.date_col] >= date]

        # Saving data

        self.save_parquet(
            df=df,
            filename=self.config.events_filenames["all"],
        )
        self.save_parquet(
            df=df_train,
            filename=self.config.events_filenames["train"],
        )
        self.save_parquet(
            df=df_target,
            filename=self.config.events_filenames["target"],
        )
        self.save_parquet(
            df=df_test2,
            filename=self.config.events_filenames["test"],
        )

        if self.print_info:
            print(
                "\n----- 'events' dataframe info after preprocessing -----\n"
            )

            # General info
            self.print_table_info(df)

            # Duplicates info
            print("-- duplicates count --")
            n_old = df.shape[0]
            n_dups = df.duplicated().sum()
            rate = round(n_dups / n_old * 100, 2)
            print(f"along all cols: {n_dups} ({rate}%)")
            for rating in df[self.config.rating_col].unique():
                n_dups = (
                    df[df[self.config.rating_col] == rating]
                    .duplicated(
                        subset=[
                            self.config.fields_id["user"],
                            self.config.fields_id["item"],
                        ]
                    )
                    .sum()
                )
                rate = round(n_dups / n_old * 100, 2)
                print(
                    f"along (user, item) for rating {rating}: {n_dups} ({rate}%)"
                )

            # Number of events, users and items
            events_total = len(df)
            users_total = df[self.config.fields_id["user"]].nunique()
            items_total = df[self.config.fields_id["item"]].nunique()
            sparsity = round(
                (1 - events_total / (users_total * items_total)) * 100, 4
            )
            print(
                f"Number of:  events - {events_total}, "
                f"users - {users_total}, ",
                f"items - {items_total}, ",
                f"sparsity - {sparsity}%\n",
            )

            # Date range info
            print(f"First date: {df[self.config.date_col].min()}")
            print(f"Last date:  {df[self.config.date_col].max()}\n")

            # Event type info
            print(
                f"Distribution of the rating: "
                f"{df[self.config.rating_col].value_counts()}\n"
            )

            print("----- 'events_train' dataframe info -----")

            # Number of events, users
            events_count = len(df_train)
            rate1 = round(events_count / events_total * 100, 2)
            users_count = len(users_train)
            rate2 = round(users_count / users_total * 100, 2)
            items_count = df_train[self.config.fields_id["item"]].nunique()
            rate3 = round(items_count / items_total * 100, 2)
            sparsity = round(
                (1 - events_count / (users_count * items_count)) * 100, 4
            )
            print(
                f"Number of: events - {events_count} ({rate1}%), "
                f"users - {users_count} ({rate2}%)",
                f"items - {items_count} ({rate3}%)\n",
                f"sparsity - {sparsity}%\n",
            )

            # Date range info
            print(f"First date: {df_train[self.config.date_col].min()}")
            print(f"Last date:  {df_train[self.config.date_col].max()}\n")

            # Event type info
            print(
                f"Distribution of the rating: "
                f"{df_train[self.config.rating_col].value_counts()}\n"
            )

            print("----- 'events_test' dataframe info -----")

            # Number of events, users
            events_count = len(df_test)
            rate1 = round(events_count / events_total * 100, 2)
            users_count = len(users_test)
            rate2 = round(users_count / users_total * 100, 2)
            users_count2 = len(users_test_only)
            rate3 = round(users_count2 / users_total * 100, 2)
            items_count = df_test[self.config.fields_id["item"]].nunique()
            rate4 = round(items_count / items_total * 100, 2)
            sparsity = round(
                (1 - events_count / (users_count * items_count)) * 100, 4
            )
            print(
                f"Number of: events - {events_count} ({rate1}%), "
                f"users - {users_count} ({rate2}%), ",
                f"items - {items_count} ({rate4}%), ",
                f"sparsity - {sparsity}%, ",
                f"users (test only) - {users_count2} ({rate3}%)\n",
            )

            # Date range info
            print(f"First date: {df_test[self.config.date_col].min()}")
            print(f"Last date:  {df_test[self.config.date_col].max()}\n")

            # Event type info
            print(
                f"Distribution of the rating: "
                f"{df_test[self.config.rating_col].value_counts()}\n"
            )

    def preprocess_item_properties(self) -> None:
        """
        Preprocesses `item_properties` data.
        """

        # Read the parquet file with item properties
        df = self.read_parquet(self.config.item_properties_filename)

        # Convert timestamp to datetime
        df[self.config.date_col] = (
            pd.to_datetime(df[self.config.date_col], unit="ms")
            .astype("datetime64[s]")
            .dt.floor("D")
        )

        # Retrieve item properties related to the item category
        n = self.config.fields_id["item_category"]
        item_cat = (
            df.query(f'property == "{n}"')
            .drop(columns="property")
            .rename(columns={"value": n})
            .sort_values(
                by=[self.config.fields_id["item"], self.config.date_col],
                ascending=[1, 1],
            )
        )
        item_cat.drop_duplicates(inplace=True)
        item_cat[n] = item_cat[n].astype("uint32")
        df = df.query(f'property != "{n}"')

        # Calculate the time difference between consecutive timestamps
        item_cat["days"] = (
            item_cat.groupby(self.config.fields_id["item"])[
                self.config.date_col
            ]
            .diff(-1)
            .dt.days.abs()
        )
        tmp = (
            item_cat.groupby(self.config.fields_id["item"])[
                self.config.date_col
            ]
            .max()
            .reset_index()
        )
        tmp["days"] = (
            pd.to_datetime(self.config.reference_date)
            - tmp[self.config.date_col]
        ).dt.days
        tmp.set_index(
            [self.config.fields_id["item"], self.config.date_col], inplace=True
        )
        item_cat.set_index(
            [self.config.fields_id["item"], self.config.date_col], inplace=True
        )
        item_cat.loc[tmp.index, "days"] = tmp["days"]
        del tmp
        item_cat.reset_index(inplace=True)

        # Save the data
        self.save_parquet(
            df=item_cat,
            filename=self.config.item_features_filenames["category"],
        )

        # Retrieve item availability data
        item_availability = (
            df.query(f'property == "available"')
            .drop(columns="property")
            .rename(columns={"value": "available"})
            .sort_values(
                by=[self.config.fields_id["item"], self.config.date_col],
                ascending=[1, 1],
            )
        )
        item_availability.drop_duplicates(inplace=True)
        df = df.query(f'property != "available"')

        # Calculate the time difference between consecutive timestamps
        item_availability["days"] = (
            item_availability.groupby(self.config.fields_id["item"])[
                self.config.date_col
            ]
            .diff(-1)
            .dt.days.abs()
        )
        tmp = (
            item_availability.groupby(self.config.fields_id["item"])[
                self.config.date_col
            ]
            .max()
            .reset_index()
        )
        tmp["days"] = (
            pd.to_datetime(self.config.reference_date)
            - tmp[self.config.date_col]
        ).dt.days
        tmp.set_index(
            [self.config.fields_id["item"], self.config.date_col], inplace=True
        )
        item_availability.set_index(
            [self.config.fields_id["item"], self.config.date_col], inplace=True
        )
        item_availability.loc[tmp.index, "days"] = tmp["days"]
        del tmp
        item_availability.reset_index(inplace=True)

        # Save the data
        self.save_parquet(
            df=item_availability,
            filename=self.config.item_features_filenames["availability"],
        )

        if self.print_info:

            # Total number of items
            n_items_total = df[self.config.fields_id["item"]].nunique()
            print(f"\nTotal number of items: {n_items_total}\n")

            print("\n----- 'item_category' dataframe info -----\n")

            # General info
            self.print_table_info(item_cat)

            # Number of items with specified category
            n_items = item_cat[self.config.fields_id["item"]].nunique()
            rate = round(n_items / n_items_total * 100, 2)
            print(
                f"\nNumber of items with specified category: {n_items} ({rate}%)\n\n"
            )

            # Number of categories per item
            print("-- number of categories per item --")
            n_categories_per_item = (
                item_cat.groupby(self.config.fields_id["item"])
                .size()
                .rename("n_categories")
            )
            print(n_categories_per_item.describe().to_markdown(), end="\n\n")

            print("\n----- 'item_availability' dataframe info -----\n")

            # General info
            self.print_table_info(item_availability)

            # Number of items with specific availability
            n_items = item_availability[
                self.config.fields_id["item"]
            ].nunique()
            rate = round(n_items / n_items_total * 100, 2)
            print(
                f"Number of items with specified availability: {n_items} ({rate}%)\n\n"
            )

            # How many days each item was available or not available
            n_days = (
                item_availability.groupby(
                    [self.config.fields_id["item"], "available"]
                )["days"]
                .sum()
                .rename("n_days")
                .reset_index()
            )
            print("-- Number of days an item was available --")
            print(
                n_days.query("available == '1'")["n_days"]
                .describe()
                .to_markdown(),
                end="\n\n",
            )
            print("-- Number of days an item was not available --")
            print(
                n_days.query("available == '0'")["n_days"]
                .describe()
                .to_markdown(),
                end="\n\n",
            )

            # Number of items that are currently available
            print("-- Number of items that are currently available --")
            n_available = len(
                item_availability.groupby(self.config.fields_id["item"])
                .head(1)
                .query("available == '1'")
            )
            rate = round(n_available / n_items_total * 100, 2)
            print(f"{n_available} ({rate}%)\n\n")

    def preprocess(self, log: bool = False) -> None:
        """
        Preprocesses the music data by performing the following steps:
        1. Preprocesses the category tree.
        2. Preprocesses the events.
        3. Preprocesses the item properties.

        Args:
            log (bool):
                Whether to log the results. Defaults to False.
        """

        self.preprocess_category_tree()
        self.preprocess_events()
        self.preprocess_item_properties()
        if log:
            self.log()
        logger.info("Finished preprocessing")
