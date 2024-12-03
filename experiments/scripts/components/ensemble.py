import warnings

warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from tqdm.auto import tqdm
from scipy.sparse import load_npz
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from typing import Literal, Dict, List, Tuple, Any

from scripts import logger
from scripts.evaluate import evaluate_recommendations
from scripts.components.base import BaseModelComponent
from scripts.settings import get_ensemble_model_component_config
from scripts.utils import read_pkl, save_pkl, save_yaml, read_yaml


class EnsembleModelComponent(BaseModelComponent):
    """
    The `EnsembleModelComponent` class is responsible for preparing
    and selecting the best ensemble model for a recommendation system.
    It merges data from multiple base models, adds features and a target
    variable, and then selects the best ensemble model based on
    evaluation metrics. The class also provides methods for ranking
    recommendations using the selected ensemble model.

    Attributes:
        config (EnsembleModelComponentConfig):
            Configuration for the ensemble model component.
        is_testing (bool):
            Whether the component is being used for testing.
        is_airflow (bool):
            Whether the component is being used via Airflow.
            This is used to determine the host for the MLflow server.
    """

    def __init__(
        self, is_testing: bool = False, is_airflow: bool = False
    ) -> None:
        """
        Initializes the EnsembleModelComponent class with the
        configuration settings.

        Args:
            is_testing (bool):
                Whether the component is being used for testing.
            is_airflow (bool):
                Whether the component is being used via Airflow.
                This is used to determine the host for the MLflow server.
        """
        self.config = get_ensemble_model_component_config()
        self.is_testing = is_testing
        self.is_airflow = is_airflow
        self._path_to_script = Path(__file__)

    def _merge_data(
        self, subset: Literal["train_df", "test_df", "all"]
    ) -> pd.DataFrame:
        """
        Merges data from multiple base models for the specified subset.

        Args:
            subset (Literal["train_df", "test_df", 'all']):
                The data subset to merge.

        Returns:
            pd.DataFrame:
                The merged DataFrame containing data from all base models.
        """
        if subset not in ["train_df", "test_df", "all"]:
            raise ValueError(f"Invalid subset: {subset}")

        path_dict = self.config.train_data_path
        if subset == "test_df":
            path_dict = self.config.test_data_path
        elif subset == "all":
            path_dict = self.config.all_data_path
        id_cols = set(
            [
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ]
        )
        df = None
        for base_model in tqdm(
            self.config.base_models, desc=f"Merging {subset} data"
        ):
            df_base = pd.read_parquet(path_dict[base_model])
            feature_col = list(set(df_base.columns) - id_cols)[0]
            df_base[feature_col] = df_base[feature_col].astype(np.float32)
            df = (
                df_base
                if df is None
                else df.merge(
                    df_base,
                    on=[
                        self.config.fields_id["user"],
                        self.config.fields_id["item"],
                    ],
                    how="outer",
                )
            ).reset_index(drop=True)
        logger.info(f"Merged {subset} data.")

        return df

    def _add_top_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds data from top items model to the provided DataFrame.
        """

        cat = "not_testing"
        if self.is_testing:
            cat = "testing"

        # Load item_encoder and user_encoder
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

        # Read top items data
        top_items = pd.read_parquet(
            Path(
                self.config.source_path4[cat],
                self.config.top_items_filename,
            )
        ).iloc[: self.config.n_recommendations]
        score_col = list(
            set(top_items.columns) - set([self.config.fields_id["item"]])
        )[0]
        top_items_dict = top_items.set_index(
            self.config.fields_id["item"]
        ).to_dict()[score_col]
        item_ids_encoded = item_encoder.transform(
            top_items[self.config.fields_id["item"]]
        )

        # Retrieve user IDs from the input dataframe
        user_ids = df[self.config.fields_id["user"]].unique().tolist()
        user_ids_encoded = user_encoder.transform(user_ids)

        #
        df2 = pd.DataFrame(
            zip(
                *np.where(
                    load_npz(
                        Path(
                            self.config.source_path2[cat],
                            self.config.user_items_matrix_filename,
                        )
                    )[:, item_ids_encoded][user_ids_encoded, :].toarray()
                    == 0
                )
            ),
            columns=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
        )

        # Decoding item ID
        df2[self.config.fields_id["item"]] = df2[
            self.config.fields_id["item"]
        ].map(dict(zip(list(range(len(item_ids_encoded))), item_ids_encoded)))
        df2[self.config.fields_id["item"]] = item_encoder.inverse_transform(
            df2[self.config.fields_id["item"]]
        )

        # Decoding user ID
        df2[self.config.fields_id["user"]] = df2[
            self.config.fields_id["user"]
        ].map(dict(zip(list(range(len(user_ids_encoded))), user_ids_encoded)))
        df2[self.config.fields_id["user"]] = user_encoder.inverse_transform(
            df2[self.config.fields_id["user"]]
        )

        # Adding score column
        df2[score_col] = df2[self.config.fields_id["item"]].map(top_items_dict)

        # Fill the new column with scores for (user ID, item ID)
        # pairs that are in the dataframe
        mask = df[self.config.fields_id["item"]].isin(
            top_items[self.config.fields_id["item"]]
        )
        df.loc[mask, score_col] = df.loc[mask][
            self.config.fields_id["item"]
        ].map(top_items_dict)

        # Remove (user ID, item ID) pairs from df2 that are in df
        df2 = (
            df2.merge(
                df.loc[mask][
                    [
                        self.config.fields_id["user"],
                        self.config.fields_id["item"],
                    ]
                ],
                on=[
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ],
                how="left",
                indicator=True,
            )
            .query('_merge == "left_only"')
            .drop(columns=["_merge"])
        )

        # Merge with the input dataframe
        df = pd.concat([df, df2], ignore_index=True)

        logger.info("Added top items data.")

        return df

    def _add_target(
        self, df: pd.DataFrame, subset: Literal["train_df", "test_df"]
    ) -> pd.DataFrame:
        """
        Adds target variable to the provided DataFrame.

        Args:
            df (pd.DataFrame):
                The DataFrame to add target variable to.
            subset (Literal["train_df", "test_df"]):
                The data subset to add target variable to.

        Returns:
            pd.DataFrame:
                The DataFrame with the added target variable.
        """
        if subset not in ["train_df", "test_df"]:
            raise ValueError(f"Invalid subset: {subset}")

        # Read real data
        df_real = pd.read_parquet(
            Path(
                self.config.source_path,
                self.config.events_filenames[
                    "target" if subset == "train_df" else "test"
                ],
            )
        )
        df_real[self.config.target_col] = 1

        # Drop duplicated user-item pairs if any
        df_real.drop_duplicates(
            subset=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
            inplace=True,
        )

        # Keep only necessary fields
        df_real = df_real[
            [
                self.config.fields_id["user"],
                self.config.fields_id["item"],
                self.config.target_col,
            ]
        ]

        # Merge with recommendations
        df = df.merge(
            df_real,
            on=[
                self.config.fields_id["user"],
                self.config.fields_id["item"],
            ],
            how="left",
        )
        df[self.config.target_col].fillna(0, inplace=True)
        df[self.config.target_col] = df[self.config.target_col].astype(
            np.uint8
        )

        logger.info(f"Added target variable for {subset} data.")

        # Calculate the number of correct recommendations
        pos_class_cnt = df[self.config.target_col].value_counts()[1]
        ratio = round(pos_class_cnt / len(df) * 100, 2)
        logger.info(
            f"Number of positive samples for {subset} data: "
            f"{pos_class_cnt} ({ratio}%) "
        )

        return df

    def _filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the data so that all positive samples and only few
        negative samples are left for each user.

        Args:
            df (pd.DataFrame):
                The DataFrame to be filtered.

        Returns:
            pd.DataFrame:
                The filtered DataFrame.
        """

        orig_size = len(df)

        # Leave only users with at least one positive target
        mask = (
            df.groupby(self.config.fields_id["user"])[
                self.config.target_col
            ].transform("sum")
            > 0
        )
        df = df[mask]
        filtered_size = len(df)
        rate = round(filtered_size / orig_size * 100, 2)
        logger.info(
            f"Left with {filtered_size} ({rate}%) samples after "
            f"filtering users with no positive targets"
        )

        # Separate positive and negative targets
        positives = df.query(f"{self.config.target_col} == 1")
        df = df.query(f"{self.config.target_col} == 0")

        # Sampling negative targets (this way is faster)
        df = df.assign(
            rand=np.random.default_rng(self.config.sampling_seed).random(
                len(df)
            )
        ).sort_values([self.config.fields_id["user"], "rand"])
        df["rank"] = df.groupby(self.config.fields_id["user"]).cumcount() + 1
        df = df.query(f"rank <= {self.config.negative_samples_per_user}").drop(
            columns=["rand", "rank"]
        )

        # Merging positive and negative targets
        df = pd.concat([positives, df], axis=0, ignore_index=True)
        del positives

        filtered_size = len(df)
        rate = round(filtered_size / orig_size * 100, 2)
        logger.info(
            f"Left with {filtered_size} ({rate}%) samples after "
            f"sampling {self.config.negative_samples_per_user} negative "
            f"samples per user"
        )

        pos_class_cnt = df[self.config.target_col].value_counts()[1]
        ratio = round(pos_class_cnt / len(df) * 100, 2)
        logger.info(f"Number of positive samples: {pos_class_cnt} ({ratio}%)")

        return df

    def _add_features(
        self,
        df: pd.DataFrame,
        kind: Literal["user", "item"],
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Adds user or item features to the provided DataFrame.

        Args:
            df (pd.DataFrame):
                The DataFrame to add features to.
            kind (Literal["user", "item"]):
                Specifies whether to add user or item features.

        Returns:
            pd.DataFrame:
                The DataFrame with the added features.
        """

        cat = "testing" if self.is_testing else "not_testing"

        if kind == "user":
            path = Path(
                self.config.source_path3[cat],
                self.config.user_features_filename,
            )
        elif kind == "item":
            path = Path(
                self.config.source_path3[cat],
                self.config.item_features_filename,
            )
        else:
            raise ValueError("Invalid kind. Must be 'user' or 'item'.")

        if (path != None) and path.exists() and path.is_file():
            df_user_features = pd.read_parquet(path)
            df = df.merge(
                df_user_features,
                on=self.config.fields_id[kind],
                how="left",
            )
            if verbose:
                logger.info(f"Added `{kind}` features")

        return df

    def prepare_data(self) -> None:
        """
        Prepares the data for the ensemble model by
        merging the base models' data, adding features and a target.
        The resulting data is saved.
        """

        if self.is_testing:
            cat = "testing"
            subsets = ["train_df", "test_df"]
        else:
            cat = "not_testing"
            subsets = ["all"]

        for subset in subsets:
            df = self._merge_data(subset)
            if self.config.include_top_items:
                df = self._add_top_items(df)
            if subset != "all":
                df = self._add_target(df, subset)
            if subset == "train_df":
                df = self._filter_data(df)
            if subset != "all":
                df = self._add_features(df, kind="user")
                df = self._add_features(df, kind="item")
            df.reset_index(drop=True, inplace=True)
            df.to_parquet(
                Path(
                    self.config.destination_path[cat],
                    self.config.recommendations_filenames[subset],
                ),
            )
            logger.info(f"Prepared and saved {subset} data.")

    def _fit_and_evaluate_candidate_model(
        self,
        model_name: str,
        model_params: Dict[str, Any],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_real: pd.DataFrame,
        features: List[str],
        verbose: bool = False,
    ) -> Tuple[object, Dict[str, Any]]:
        """
        Fits and evaluates a candidate ensemble model.

        Args:
            model_name (str):
                The name of the ensemble model to be initialized.
            model_params (Dict[str, Any]):
                The parameters for the ensemble model.
            df_train (pd.DataFrame):
                The training data.
            df_test (pd.DataFrame):
                The test data.
            df_real (pd.DataFrame):
                The real data.
            features (List[str]):
                The list of feature columns.

        Returns:
            Tuple[object, Dict[str, Any]]:
                The fitted ensemble model and a dict containing
                the evaluation metrics.
        """

        # Initialize an ensemble model
        model = globals()[model_name](**model_params)
        if verbose:
            logger.info(f"Initialized candidate ensemble model: {model_name}")

        # Fit the model
        model.fit(
            X=df_train[features],
            y=df_train[self.config.target_col],
        )
        if verbose:
            logger.info("Fitted the candidate ensemble model")

        # Predict on the test set
        df_test[self.config.score_col] = model.predict_proba(
            df_test[features]
        )[:, 1]
        df_pred = (
            df_test.sort_values(
                by=[
                    self.config.fields_id["user"],
                    self.config.score_col,
                ],
                ascending=[1, 0],
            )
            .groupby(self.config.fields_id["user"])
            .head(self.config.n_recommendations)[
                [self.config.fields_id["user"], self.config.fields_id["item"]]
            ]
        )
        if verbose:
            logger.info("Ranked recommendations")

        # Evaluate the model on the test set
        metrics = evaluate_recommendations(
            user_items_real=df_real,
            user_items_pred=df_pred,
            user_id_col=self.config.fields_id["user"],
            item_id_col=self.config.fields_id["item"],
            K=self.config.n_recommendations,
        )
        if verbose:
            logger.info(f"Evaluated the candidate ensemble model")

        return model, metrics

    def select_ensemble_model(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_real: pd.DataFrame,
        features: List[str],
    ) -> None:
        """
        Selects and saves the best ensemble model based on the
        evaluation metrics.

        Args:
            df_train (pd.DataFrame):
                The training data.
            df_test (pd.DataFrame):
                The test data.
            df_real (pd.DataFrame):
                The real data.
            features (List[str]):
                The list of feature columns.
        """

        # Initialize an empty dictionary to store models
        # and a variable to store metrics
        models = {}
        metrics = None

        # Fit and evaluate each candidate model
        for (
            model_name,
            params,
        ) in self.config.ensemble_model_candidates.items():

            models[model_name], model_metrics = (
                self._fit_and_evaluate_candidate_model(
                    model_name=model_name,
                    model_params=params["model_params"],
                    df_train=df_train,
                    df_test=df_test,
                    df_real=df_real,
                    features=features,
                )
            )

            model_metrics = pd.DataFrame(
                model_metrics.items(), columns=["metric", model_name]
            ).set_index("metric")

            metrics = (
                model_metrics
                if metrics is None
                else pd.concat([metrics, model_metrics], axis=1)
            )

        # Select the best model according to the metrics
        metrics = metrics.T
        metric_names = list(self.config.target_metrics_order.keys())
        order = list(self.config.target_metrics_order.values())
        metrics.sort_values(metric_names, ascending=order, inplace=True)
        best_model_name = metrics.index[0]
        logger.info(f"Selected the best ensemble model: {best_model_name}")

        # Save the best model
        save_pkl(
            path=Path(
                self.config.destination_path["testing"],
                self.config.model_filename,
            ),
            model=models[best_model_name],
        )

        # Save the metrics data
        metrics.to_csv(
            Path(
                self.config.destination_path["testing"],
                self.config.metrics_df_filename,
            ),
        )
        logger.info("Saved metrics data")

        # Save the metrics data for the best model
        save_yaml(
            path=Path(
                self.config.destination_path["testing"],
                self.config.metrics_filename,
            ),
            data=metrics.loc[best_model_name].to_dict(),
        )
        logger.info("Saved metrics data for the best model")

        # Retrieving feature importances and saving them
        importances = dict(
            zip(
                features,
                list(map(float, models[best_model_name].feature_importances_)),
            )
        )
        importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )
        logger.info(f"Feature importances: {importances}")
        save_yaml(
            path=Path(
                self.config.destination_path["testing"],
                self.config.features_importance_filename,
            ),
            data=importances,
        )
        logger.info("Saved feature importances")

    def fit(self) -> None:
        """
        Selects the best ensemble model and saves the best model with
        the metrics data
        """

        if not self.is_testing:
            msg = "Can not fit when mode is not testing"
            logger.error(msg)
            raise ValueError(msg)

        # Read train data
        df_train = pd.read_parquet(
            Path(
                self.config.destination_path["testing"],
                self.config.recommendations_filenames["train_df"],
            ),
        )
        logger.info("Read train data")

        # Read test data
        df_test = pd.read_parquet(
            Path(
                self.config.destination_path["testing"],
                self.config.recommendations_filenames["test_df"],
            ),
        )
        logger.info("Read test data")

        # Reading data with real interactions
        df_real = (
            pd.read_parquet(
                Path(
                    self.config.source_path,
                    self.config.events_filenames["test"],
                )
            )
            .query(f"rating >= 2")[
                [
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ]
            ]
            .drop_duplicates()
        )
        logger.info("Read real data")

        # Separate names of columns with features
        features = sorted(
            set(df_train.columns)
            - set(
                [
                    self.config.target_col,
                    self.config.date_col,
                    self.config.fields_id["user"],
                    self.config.fields_id["item"],
                ]
            )
        )

        # Select the best ensemble model
        self.select_ensemble_model(
            df_train=df_train,
            df_test=df_test,
            df_real=df_real,
            features=features,
        )

    def recommend(self) -> None:
        """
        Ranks recommendations for all users using the best ensemble
        model.
        """

        # Read test data
        df = pd.read_parquet(
            Path(
                self.config.destination_path["not_testing"],
                self.config.recommendations_filenames["all"],
            ),
        )
        logger.info(f"Read data")

        # Get features
        features = list(
            read_yaml(
                Path(
                    self.config.destination_path["testing"],
                    self.config.features_importance_filename,
                )
            ).keys()
        )

        # Load the trained ensemble model
        model = read_pkl(
            path=Path(
                self.config.destination_path["testing"],
                self.config.model_filename,
            )
        )

        # Predict
        batch_size = 10**5
        for idx in tqdm(
            range(0, len(df), batch_size), desc="Predicting", unit="batch"
        ):
            idy = min(idx + batch_size - 1, len(df))
            batch_df = self._add_features(
                df.loc[idx:idy], kind="user", verbose=False
            )
            batch_df = self._add_features(batch_df, kind="item", verbose=False)

            df.loc[idx:idy, self.config.score_col] = model.predict_proba(
                batch_df[features]
            )[:, 1]
        logger.info("Ranked recommendations")

        # Keep only necessary columns
        df = df[
            [
                self.config.fields_id["user"],
                self.config.fields_id["item"],
                self.config.score_col,
            ]
        ]

        # Leave top recommendations
        df = (
            df.sort_values(
                by=[self.config.fields_id["user"], self.config.score_col],
                ascending=[True, False],
            )
            .groupby(self.config.fields_id["user"])
            .head(self.config.n_recommendations)
        )
        logger.info(
            f"Left only top {self.config.n_recommendations} recommendations"
        )

        # Saving recommendations
        df.to_parquet(
            Path(
                self.config.destination_path["not_testing"],
                self.config.recommendations_filenames["all"],
            ),
        )
        logger.info(f"Saved recommendations")
