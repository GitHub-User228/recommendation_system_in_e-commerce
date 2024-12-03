import os
import sys
import time
from pathlib import Path

import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix
from typing import List, Any, Literal
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking


class BaseRecommender(ABC):
    """
    Base class for a recommender model.
    Provides common functionality for derived classes.

    Attributes:
        _logger (logging.Logger):
            Logger for the class.
    """

    def __init__(self) -> None:
        self._logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(name)s] [%(funcName)s] - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False
        return logger

    def __getstate__(self):
        # Exclude the logger from the serialized state
        state = self.__dict__.copy()
        state["_logger"] = None
        return state

    def __setstate__(self, state):
        # Restore the object's state and reinitialize the logger
        self.__dict__.update(state)
        self._logger = self._setup_logger()

    def _validate_positive_integer(self, value: int, name: str) -> None:
        if not isinstance(value, int) or value < 1:
            msg = "Value must be a positive integer"
            self._logger.error(msg)
            raise ValueError(msg)

    def _validate_float_range(
        self, value: float, name: str, min_val: float, max_val: float
    ) -> None:
        if not isinstance(value, float) or not (min_val <= value <= max_val):
            msg = f"{name} must be a float between {min_val} and {max_val}"
            self._logger.error(msg)
            raise ValueError(msg)

    def _validate_type(
        self, value: Any, expected_type: Any, name: str
    ) -> None:
        if not isinstance(value, expected_type):
            msg = f"{name} must be of type {expected_type.__name__}"
            self._logger.error(msg)
            raise TypeError(msg)

    def _validate_literal(
        self, value: Any, allowed_values: List[Any], name: str
    ) -> None:
        if value not in allowed_values:
            allowed_values_str = ", ".join(repr(v) for v in allowed_values)
            msg = f"{name} must be one of: {allowed_values_str}, got {value}"
            self._logger.error(msg)
            raise ValueError(msg)

    @abstractmethod
    def _validate_input(self) -> None:
        """
        Validates the input parameters for the class.
        """
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the model to the data.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def recommend(self, *args, **kwargs):
        """
        Generate recommendations.
        This method must be implemented by subclasses.
        """
        pass


class ALS(BaseRecommender):
    """
    Implementation of the Alternating Least Squares recommendation
    model with user and item IDs encoders.

    Attributes:
        min_users_per_item (int):
            The minimum number of users who must have interacted
            with an item in order for it to be considered in the
            model
        factors (int):
            The number of latent factors to use in the ALS model.
        iterations (int):
            The number of iterations to run the ALS algorithm.
        regularization (float):
            The regularization parameter for the ALS model.
        random_state (int):
            The random state to use for reproducibility.
        calculate_training_loss (bool):
            Whether to calculate the training loss during the ALS
            fitting process.
        item_id_encoder (LabelEncoder):
            The LabelEncoder used to encode the item IDs.
        user_id_encoder (LabelEncoder):
            The LabelEncoder used to encode the user IDs.
        user_mask (np.ndarray):
            A boolean array indicating which users are in the model.
        item_mask (np.ndarray):
            A boolean array indicating which items are in the model.
        model (AlternatingLeastSquares):
            The ALS model instance.

    Raises:
        TypeError:
            If any of the input arguments are not of the expected
            types.
        ValueError:
            If any of the input arguments are invalid or inconsistent.
    """

    def __init__(
        self,
        min_users_per_item: int = 1,
        factors: int = 5,
        iterations: int = 10,
        regularization: float = 0.01,
        alpha: float | None = None,
        filter_already_liked_items: bool = True,
        calculate_training_loss: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the ALS class with the specified parameters.

        Args:
            min_users_per_item (int, optional):
                The minimum number of users who must have interacted
                with an item in order for it to be considered in the
                model. Defaults to 1.
            factors (int, optional):
                The number of latent factors to use in the ALS model.
                Defaults to 5.
            iterations (int, optional):
                The number of iterations to run the ALS algorithm.
                Defaults to 10.
            regularization (float, optional):
                The regularization parameter for the ALS model.
                Defaults to 0.01.
            alpha (float, optional):
                The weight to give to positive examples.
                Defaults to 20.
            random_state (int, optional):
                The random state to use for reproducibility.
                Defaults to 42.
            calculate_training_loss (bool, optional):
                Whether to calculate the training loss during the ALS
                fitting process. Defaults to True.
        """
        super().__init__()
        self.min_users_per_item = min_users_per_item
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.filter_already_liked_items = filter_already_liked_items
        self.random_state = random_state
        self.calculate_training_loss = calculate_training_loss
        self._validate_input()

    def _validate_input(self) -> None:
        """Validates the input arguments for the class"""
        self._validate_positive_integer(
            self.min_users_per_item, "min_users_per_item"
        )
        self._validate_positive_integer(self.factors, "factors")
        self._validate_positive_integer(self.iterations, "iterations")
        self._validate_float_range(self.regularization, "regularization", 0, 1)
        if self.alpha:
            self._validate_float_range(self.alpha, "alpha", 1, 50)
        self._validate_type(self.random_state, int, "random_state")
        self._validate_type(
            self.calculate_training_loss, bool, "calculate_training_loss"
        )

    def fit(
        self,
        user_items_matrix: csr_matrix,
        item_id_encoder: LabelEncoder,
        user_id_encoder: LabelEncoder,
    ) -> "ALS":
        """
        Fits the Alternating Least Squares (ALS) model to the provided
        user-item matrix.

        Args:
            user_items_matrix (csr_matrix):
                The user-item matrix in sparse CSR format.
            item_id_encoder (LabelEncoder):
                The LabelEncoder used to encode the item IDs.
            user_id_encoder (LabelEncoder):
                The LabelEncoder used to encode the user IDs.
            user_items_matrix_save_path (Path):
                The path to save the user-item matrix to.

        Returns:
            ALS:
                The fitted ALS model instance.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate input
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        self._validate_type(item_id_encoder, LabelEncoder, "item_id_encoder")
        self._validate_type(user_id_encoder, LabelEncoder, "user_id_encoder")

        # Creating a mask to filter only popular items
        self.item_mask = (
            user_items_matrix.getnnz(axis=0) >= self.min_users_per_item
        )
        user_items_matrix = user_items_matrix[:, self.item_mask]
        n_items = user_items_matrix.shape[1]
        rate = round(n_items / len(self.item_mask) * 100, 2)
        self._logger.info(f"Left with {n_items} ({rate}%) popular items")

        # Creating a mask to filter users with at least one item
        self.user_mask = user_items_matrix.getnnz(axis=1) >= 1
        user_items_matrix = user_items_matrix[self.user_mask, :]
        n_users = user_items_matrix.shape[0]
        rate = round(n_users / len(self.user_mask) * 100, 2)
        self._logger.info(f"Left with {n_users} ({rate}%) active users")

        # Fitting the item id encoder
        self.item_id_encoder = LabelEncoder()
        self.item_id_encoder.fit(item_id_encoder.classes_[self.item_mask])
        self._logger.info("Fitted item_id encoder")

        # Fitting the user id encoder
        self.user_id_encoder = LabelEncoder()
        self.user_id_encoder.fit(user_id_encoder.classes_[self.user_mask])
        self._logger.info("Fitted user_id encoder")

        # Fitting the AlternatingLeastSquares model to the data
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            alpha=self.alpha,
            calculate_training_loss=self.calculate_training_loss,
            random_state=self.random_state,
        )
        self.model.fit(user_items_matrix)
        self._logger.info("Fitted ALS model")

        return self

    def recommend(
        self,
        user_ids: List[int],
        user_items_matrix: csr_matrix,
        n_recommendations: int,
        user_id_col: str,
        item_id_col: str,
        score_col: str,
        filter_items: List[int] | None = None,
    ) -> pd.DataFrame | None:
        """
        Recommend items for the given user IDs.

        Args:
            user_ids (List[int]):
                A list of user IDs for which to generate
                recommendations.
            user_items_matrix (csr_matrix):
                The user-item matrix in sparse CSR format.
                Must be the same as before fitting the model.
            n_recommendations (int):
                The number of recommendations to generate for each
                user.
            user_id_col (str):
                The column name in the DataFrame to contain user IDs.
            item_id_col (str):
                The column name in the DataFrame to contain item IDs.
            score_col (str):
                The column name in the DataFrame to contain the
                scores of the recommendations.
            filter_items (List[int] | None, optional):
                A list of item IDs to filter out from the
                recommendations. Defaults to None.

        Returns:
            pd.DataFrame | None:
                A DataFrame containing the recommended item IDs and
                their scores for each user. Returns None if no input
                users were found within the fitted model.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate input
        self._validate_type(user_ids, list, "user_ids")
        if not all(isinstance(user_id, int) for user_id in user_ids):
            msg = "user_ids must contain only integers"
            self._logger.error(msg)
            raise TypeError(msg)
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        self._validate_positive_integer(n_recommendations, "n_recommendations")
        self._validate_type(item_id_col, str, "item_id_col")
        self._validate_type(user_id_col, str, "user_id_col")
        if filter_items is not None:
            self._validate_type(filter_items, list, "filter_items")
            if not all(isinstance(item_id, int) for item_id in filter_items):
                msg = "filter_items must contain only integers"
                self._logger.error(msg)
                raise TypeError(msg)

        # Filter only users that are presented in the als model
        n_users = len(user_ids)
        user_ids = list(set(user_ids) & set(self.user_id_encoder.classes_))
        if len(user_ids) == 0:
            self._logger.warning("No users in the input are in the model")
            return None
        rate = round(len(user_ids) / n_users * 100, 2)
        self._logger.info(
            f"Found {len(user_ids)} ({rate}%) users from {n_users} in the model"
        )

        # Filter only items that are presented in the model
        if isinstance(filter_items, list) and len(filter_items) > 0:
            filter_items = list(
                set(filter_items) & set(self.item_id_encoder.classes_)
            )
            if len(filter_items) == 0:
                self._logger.warning("No items to filter are in the model")
            else:
                filter_items = self.item_id_encoder.transform(filter_items)
                rate = round(
                    len(filter_items)
                    / len(self.item_id_encoder.classes_)
                    * 100,
                    2,
                )
                self._logger.info(
                    f"Found {len(filter_items)} ({rate}%) items from "
                    f"{len(self.item_id_encoder.classes_)} to be filtered"
                )

        # Transform user_ids to their corresponding encodings
        user_ids_enc = self.user_id_encoder.transform(user_ids)

        # Filter user_items_matrix
        user_items_matrix = user_items_matrix[:, self.item_mask]
        user_items_matrix = user_items_matrix[self.user_mask, :]
        user_items_matrix = user_items_matrix[user_ids_enc, :]
        self._logger.info("Filtered user_items_matrix")

        # Get the recommendations
        item_ids_enc, scores = self.model.recommend(
            userid=user_ids_enc,
            user_items=user_items_matrix,
            N=n_recommendations,
            filter_already_liked_items=self.filter_already_liked_items,
            recalculate_user=False,
            filter_items=filter_items,
        )
        self._logger.info("Generated recommendations")

        df = pd.DataFrame(
            {
                user_id_col: user_ids,
                item_id_col: item_ids_enc.tolist(),
                score_col: scores.tolist(),
            }
        )
        del item_ids_enc, scores

        df = df.apply(pd.Series.explode)
        for col, dtype in {
            user_id_col: int,
            item_id_col: int,
            score_col: float,
        }.items():
            df[col] = df[col].astype(dtype)
        df = df.sort_values(
            by=[user_id_col, score_col], ascending=[True, False]
        ).reset_index(drop=True)

        # Transform the item_ids back to their original labels
        df[item_id_col] = self.item_id_encoder.inverse_transform(
            df[item_id_col]
        )
        self._logger.info("Prepared a Dataframe with recommendations")

        return df

    def get_similar_items(
        self,
        max_similar_items: int,
        item_id_col: str,
        item_id_col_similar: str,
        score_col: str,
        filter_items: List[int] | None = None,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame containing the most similar items for
        each item in the model.

        Args:
            max_similar_items (int):
                The maximum number of similar items to retrieve for
                each item.
            item_id_col (str):
                The column name in the DataFrame to contain item IDs.
            item_id_col_similar (str):
                The column name in the DataFrame to contain similar
                item IDs.
            score_col (str):
                The column name in the DataFrame to contain the
                similarity scores.
            filter_items (List[int] | None, optional):
                A list of item IDs to filter out from the
                recommendations. Defaults to None.

        Returns:
            pd.DataFrame:
                A DataFrame containing the item ID, similar item ID,
                and similarity score.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validating input
        self._validate_positive_integer(max_similar_items, "max_similar_items")
        self._validate_type(item_id_col, str, "item_id_col")
        self._validate_type(item_id_col_similar, str, "item_id_col_similar")
        self._validate_type(score_col, str, "score_col")
        if filter_items is not None:
            self._validate_type(filter_items, list, "filter_items")
            if not all(isinstance(item_id, int) for item_id in filter_items):
                msg = "filter_items must contain only integers"
                self._logger.error(msg)
                raise TypeError(msg)

        # Filter only items that are presented in the model
        if isinstance(filter_items, list) and len(filter_items) > 0:
            filter_items = list(
                set(filter_items) & set(self.item_id_encoder.classes_)
            )
            if len(filter_items) == 0:
                self._logger.warning("No items to filter are in the model")
            else:
                filter_items = self.item_id_encoder.transform(filter_items)
                rate = round(
                    len(filter_items)
                    / len(self.item_id_encoder.classes_)
                    * 100,
                    2,
                )
                self._logger.info(
                    f"Found {len(filter_items)} ({rate}%) items from "
                    f"{len(self.item_id_encoder.classes_)} to be filtered"
                )

        # generating similar items
        item_ids_enc = list(range(len(self.item_id_encoder.classes_)))
        similar_item_ids_enc, scores = self.model.similar_items(
            item_ids_enc,
            N=max_similar_items + 1,
            filter_items=filter_items,
        )
        self._logger.info("Generated similar items")

        # Create a DataFrame to hold the results
        df = pd.DataFrame(
            {
                item_id_col: item_ids_enc,
                item_id_col_similar: similar_item_ids_enc.tolist(),
                score_col: scores.tolist(),
            }
        )
        del item_ids_enc, scores
        df = df.apply(pd.Series.explode)
        for col, dtype in {
            item_id_col: int,
            item_id_col_similar: int,
            score_col: float,
        }.items():
            df[col] = df[col].astype(dtype)

        # Removing rows same as the original items
        df = df.query(f"{item_id_col} != {item_id_col_similar}")

        # Transforming the similar_item_id and item_id back to original
        for col in [item_id_col, item_id_col_similar]:
            df[col] = self.item_id_encoder.inverse_transform(df[col])
        self._logger.info("Prepared a dataframe with similar items")

        return df


class BPR(BaseRecommender):
    """
    Production-ready implementation of the Alternating Least Squares
    recommendation model.

    Attributes:
        min_users_per_item (int):
            The minimum number of users who must have interacted
            with an item in order for it to be considered in the
            model
        factors (int):
            The number of latent factors.
        iterations (int):
            The number of iterations.
        learning_rate (float):
            The learning rate.
        regularization (float):
            The regularization parameter.
        random_state (int):
            The random state to use for reproducibility.
        verify_negative_samples (bool):
            Whether to verify that the negative samples
        item_id_encoder (LabelEncoder):
            The LabelEncoder used to encode the item IDs.
        user_id_encoder (LabelEncoder):
            The LabelEncoder used to encode the user IDs.
        user_mask (np.ndarray):
            A boolean array indicating which users are in the model.
        item_mask (np.ndarray):
            A boolean array indicating which items are in the model.
        model (BayesianPersonalizedRanking):
            The BPR model instance.

    Raises:
        TypeError:
            If any of the input arguments are not of the expected
            types.
        ValueError:
            If any of the input arguments are invalid or inconsistent.
    """

    def __init__(
        self,
        min_users_per_item: int = 1,
        factors: int = 5,
        learning_rate: float = 0.05,
        iterations: int = 10,
        regularization: float = 0.01,
        filter_already_liked_items: bool = True,
        verify_negative_samples: bool = True,
        random_state: int = 42,
    ) -> None:
        """
        Initializes the BPR class with the specified parameters.

        Args:
            min_users_per_item (int, optional):
                The minimum number of users who must have interacted
                with an item in order for it to be considered in the
                model. Defaults to 1.
            factors (int, optional):
                The number of latent factors.
                Defaults to 5.
            learning_rate (float, optional):
                The learning rate. Defaults to 0.05.
            iterations (int, optional):
                The number of iterations.
                Defaults to 10.
            regularization (float, optional):
                The regularization parameter.
                Defaults to 0.01.
            verify_negative_samples (bool, optional):
                Whether to verify the negative samples.
                Defaults to True.
            random_state (int, optional):
                The random state to use for reproducibility.
                Defaults to 42.
        """
        super().__init__()
        self.min_users_per_item = min_users_per_item
        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.filter_already_liked_items = filter_already_liked_items
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state
        self._validate_input()

    def _validate_input(self) -> None:
        """Validates the input arguments for the class"""
        self._validate_positive_integer(
            self.min_users_per_item, "min_users_per_item"
        )
        self._validate_positive_integer(self.factors, "factors")
        self._validate_positive_integer(self.iterations, "iterations")
        self._validate_float_range(self.learning_rate, "learning_rate", 0, 1)
        self._validate_float_range(self.regularization, "regularization", 0, 1)
        self._validate_type(self.random_state, int, "random_state")
        self._validate_type(
            self.verify_negative_samples, bool, "verify_negative_samples"
        )

    def fit(
        self,
        user_items_matrix: csr_matrix,
        item_id_encoder: LabelEncoder,
        user_id_encoder: LabelEncoder,
    ) -> "BPR":
        """
        Fits the Bayesian Personalized Ranking (BPR) model to the
        provided user-item matrix.

        Args:
            user_items_matrix (csr_matrix):
                The user-item matrix in sparse CSR format.
            item_id_encoder (LabelEncoder):
                The LabelEncoder used to encode the item IDs.
            user_id_encoder (LabelEncoder):
                The LabelEncoder used to encode the user IDs.
            user_items_matrix_save_path (Path):
                The path to save the user-item matrix to.

        Returns:
            BPR:
                The fitted BPR model instance.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate input
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        self._validate_type(item_id_encoder, LabelEncoder, "item_id_encoder")
        self._validate_type(user_id_encoder, LabelEncoder, "user_id_encoder")

        # Creating a mask to filter only popular items
        self.item_mask = (
            user_items_matrix.getnnz(axis=0) >= self.min_users_per_item
        )
        user_items_matrix = user_items_matrix[:, self.item_mask]
        n_items = user_items_matrix.shape[1]
        rate = round(n_items / len(self.item_mask) * 100, 2)
        self._logger.info(f"Filtered {n_items} ({rate}%) popular items")

        # Creating a mask to filter users with at least one item
        self.user_mask = user_items_matrix.getnnz(axis=1) >= 1
        user_items_matrix = user_items_matrix[self.user_mask, :]
        n_users = user_items_matrix.shape[0]
        rate = round(n_users / len(self.user_mask) * 100, 2)
        self._logger.info(f"Filtered {n_users} ({rate}%) active users")

        # Fitting the item id encoder
        self.item_id_encoder = LabelEncoder()
        self.item_id_encoder.fit(item_id_encoder.classes_[self.item_mask])
        self._logger.info("Fitted item_id encoder")

        # Fitting the user id encoder
        self.user_id_encoder = LabelEncoder()
        self.user_id_encoder.fit(user_id_encoder.classes_[self.user_mask])
        self._logger.info("Fitted user_id encoder")

        # Fitting the AlternatingLeastSquares model to the data
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            verify_negative_samples=self.verify_negative_samples,
            random_state=self.random_state,
        )
        self.model.fit(user_items_matrix)
        self._logger.info("Fitted BPR model")

        return self

    def recommend(
        self,
        user_ids: List[int],
        user_items_matrix: csr_matrix,
        n_recommendations: int,
        user_id_col: str,
        item_id_col: str,
        score_col: str,
        filter_items: List[int] | None = None,
    ) -> pd.DataFrame | None:
        """
        Recommend items for the given user IDs.

        Args:
            user_ids (List[int]):
                A list of user IDs for which to generate
                recommendations.
            user_items_matrix (csr_matrix):
                The user-item matrix in sparse CSR format.
                Must be the same as before fitting the model.
            n_recommendations (int):
                The number of recommendations to generate for each
                user.
            user_id_col (str):
                The column name in the DataFrame to contain user IDs.
            item_id_col (str):
                The column name in the DataFrame to contain item IDs.
            score_col (str):
                The column name in the DataFrame to contain the
                scores of the recommendations.
            filter_items (List[int] | None, optional):
                A list of item IDs to filter out from the
                recommendations. Defaults to None.

        Returns:
            pd.DataFrame | None:
                A DataFrame containing the recommended item IDs and
                their scores for each user. Returns None if no input
                users were found within the fitted model.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate input
        self._validate_type(user_ids, list, "user_ids")
        if not all(isinstance(user_id, int) for user_id in user_ids):
            msg = "user_ids must contain only integers"
            self._logger.error(msg)
            raise TypeError(msg)
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        self._validate_positive_integer(n_recommendations, "n_recommendations")
        self._validate_type(item_id_col, str, "item_id_col")
        self._validate_type(user_id_col, str, "user_id_col")
        self._validate_type(score_col, str, "score_col")
        if filter_items is not None:
            self._validate_type(filter_items, list, "filter_items")
            if not all(isinstance(item_id, int) for item_id in filter_items):
                msg = "filter_items must contain only integers"
                self._logger.error(msg)
                raise TypeError(msg)

        # Filter only users that are presented in the model
        n_users = len(user_ids)
        user_ids = list(set(user_ids) & set(self.user_id_encoder.classes_))
        if len(user_ids) == 0:
            self._logger.warning("No users in the input are in the model")
            return None
        rate = round(len(user_ids) / n_users * 100, 2)
        self._logger.info(
            f"Found {len(user_ids)} ({rate}%) users from {n_users} in the model"
        )

        # Filter only items that are presented in the model
        if isinstance(filter_items, list) and len(filter_items) > 0:
            filter_items = list(
                set(filter_items) & set(self.item_id_encoder.classes_)
            )
            if len(filter_items) == 0:
                self._logger.warning("No items to filter are in the model")
            else:
                filter_items = self.item_id_encoder.transform(
                    filter_items
                ).tolist()
                rate = round(
                    len(filter_items)
                    / len(self.item_id_encoder.classes_)
                    * 100,
                    2,
                )
                self._logger.info(
                    f"Found {len(filter_items)} ({rate}%) items from "
                    f"{len(self.item_id_encoder.classes_)} to be filtered"
                )

        # Transforming user_ids to their corresponding encodings
        user_ids_enc = self.user_id_encoder.transform(user_ids)

        # Filtering user_items_matrix
        user_items_matrix = user_items_matrix[:, self.item_mask]
        user_items_matrix = user_items_matrix[self.user_mask, :]
        user_items_matrix = user_items_matrix[user_ids_enc, :]
        self._logger.info("Filtered user_items_matrix")

        # Getting the recommendations
        item_ids_enc, scores = self.model.recommend(
            userid=user_ids_enc,
            user_items=user_items_matrix,
            N=n_recommendations,
            filter_already_liked_items=self.filter_already_liked_items,
            recalculate_user=False,
            filter_items=filter_items,
        )
        self._logger.info("Generated recommendations")

        df = pd.DataFrame(
            {
                user_id_col: user_ids,
                item_id_col: item_ids_enc.tolist(),
                score_col: scores.tolist(),
            }
        )
        del item_ids_enc, scores

        df = df.apply(pd.Series.explode)
        for col, dtype in {
            user_id_col: int,
            item_id_col: int,
            score_col: float,
        }.items():
            df[col] = df[col].astype(dtype)
        df = df.sort_values(
            by=[user_id_col, score_col], ascending=[True, False]
        ).reset_index(drop=True)

        # Transforming the item_ids back to their original labels
        df[item_id_col] = self.item_id_encoder.inverse_transform(
            df[item_id_col]
        )
        self._logger.info("Prepared a Dataframe with recommendations")

        return df

    def get_similar_items(
        self,
        max_similar_items: int,
        item_id_col: str,
        item_id_col_similar: str,
        score_col: str,
        filter_items: List[int] | None = None,
    ) -> pd.DataFrame:
        """
        Generates a DataFrame containing the most similar items for
        each item in the model.

        Args:
            max_similar_items (int):
                The maximum number of similar items to retrieve for
                each item.
            item_id_col (str):
                The column name in the DataFrame to contain item IDs.
            item_id_col_similar (str):
                The column name in the DataFrame to contain similar
                item IDs.
            score_col (str):
                The column name in the DataFrame to contain the
                similarity scores.
            filter_items (List[int] | None, optional):
                A list of item IDs to filter out from the
                recommendations. Defaults to None.

        Returns:
            pd.DataFrame:
                A DataFrame containing the item ID, similar item ID,
                and similarity score.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validating input
        self._validate_positive_integer(max_similar_items, "max_similar_items")
        self._validate_type(item_id_col, str, "item_id_col")
        self._validate_type(item_id_col_similar, str, "item_id_col_similar")
        self._validate_type(score_col, str, "score_col")
        if filter_items is not None:
            self._validate_type(filter_items, list, "filter_items")
            if not all(isinstance(item_id, int) for item_id in filter_items):
                msg = "filter_items must contain only integers"
                self._logger.error(msg)
                raise TypeError(msg)

        # Filter only items that are presented in the model
        if isinstance(filter_items, list) and len(filter_items) > 0:
            filter_items = list(
                set(filter_items) & set(self.item_id_encoder.classes_)
            )
            if len(filter_items) == 0:
                self._logger.warning("No items to filter are in the model")
            else:
                filter_items = self.item_id_encoder.transform(filter_items)
                rate = round(
                    len(filter_items)
                    / len(self.item_id_encoder.classes_)
                    * 100,
                    2,
                )
                self._logger.info(
                    f"Found {len(filter_items)} ({rate}%) items from "
                    f"{len(self.item_id_encoder.classes_)} to be filtered"
                )

        # generating similar items
        item_ids_enc = list(range(len(self.item_id_encoder.classes_)))
        similar_item_ids_enc, scores = self.model.similar_items(
            item_ids_enc,
            N=max_similar_items + 1,
            filter_items=filter_items,
        )
        self._logger.info("Generated similar items")

        # Create a DataFrame to hold the results
        df = pd.DataFrame(
            {
                item_id_col: item_ids_enc,
                item_id_col_similar: similar_item_ids_enc.tolist(),
                score_col: scores.tolist(),
            }
        )
        del item_ids_enc, scores
        df = df.apply(pd.Series.explode)
        for col, dtype in {
            item_id_col: int,
            item_id_col_similar: int,
            score_col: float,
        }.items():
            df[col] = df[col].astype(dtype)

        # Removing rows same as the original items
        df = df.query(f"{item_id_col} != {item_id_col_similar}")

        # Transforming the similar_item_id and item_id back to original
        for col in [item_id_col, item_id_col_similar]:
            df[col] = self.item_id_encoder.inverse_transform(df[col])
        self._logger.info("Prepared a Dataframe with similar items")

        return df


class Item2ItemModel(BaseRecommender):

    def __init__(
        self,
        min_users_per_item: int = 1,
        n_neighbors: int = 100,
        n_components: int = 500,
        similarity_criteria: Literal[
            "cosine", "euclidean", "manhattan"
        ] = "manhattan",
    ):
        """
        Initializes an Item2ItemModel instance

        Args:
            min_users_per_item (int):
                The minimum number of users that must have interacted
                with an item for it to be included in the model.
                Defaults to 1.
            n_neighbors (int):
                The number of nearest neighbors to consider when making
                recommendations. Defaults to 100.
            n_components (int):
                The number of latent components to use in the item-item
                similarity matrix. Defaults to 500.
            similarity_criteria (Literal["cosine", "euclidean", "manhattan"]):
                The similarity metric to use when computing item-item
                similarities. Defaults to "manhattan".
        """
        super().__init__()
        self.min_users_per_item = min_users_per_item
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.similarity_criteria = similarity_criteria
        self._validate_input()

    def _validate_input(self) -> None:
        """Validates the input parameters for the class."""
        self._validate_positive_integer(
            self.min_users_per_item, "min_users_per_item"
        )
        self._validate_positive_integer(self.n_neighbors, "n_neighbors")
        self._validate_positive_integer(self.n_components, "n_components")
        if self.similarity_criteria not in [
            "cosine",
            "euclidean",
            "manhattan",
        ]:
            raise ValueError(
                "similarity_criteria must be 'cosine', 'euclidean', or 'manhattan'"
            )

    def fit(
        self,
        item_features_matrix: csr_matrix,
        user_items_matrix: csr_matrix,
        item_id_encoder: LabelEncoder,
        user_id_encoder: LabelEncoder,
        n_jobs: int = 1,
    ) -> "Item2ItemModel":
        """
        Fits the model.

        Args:
            item_features_matrix (csr_matrix):
                Matrix of item features.
            user_items_matrix (csr_matrix):
                Matrix of user-item interactions.
            item_id_encoder (LabelEncoder):
                Encoder for item ids.
            user_id_encoder (LabelEncoder):
                Encoder for user ids.
            n_jobs (int):
                Number of jobs to run in parallel.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate inputs
        self._validate_type(
            item_features_matrix, csr_matrix, "item_features_matrix"
        )
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        self._validate_type(item_id_encoder, LabelEncoder, "item_id_encoder")
        self._validate_type(user_id_encoder, LabelEncoder, "user_id_encoder")
        self._validate_positive_integer(n_jobs, "n_jobs")

        # Filter out unpopular items and items with no data
        self.item_mask = (
            user_items_matrix.getnnz(axis=0) >= self.min_users_per_item
        )
        self.item_mask = (
            self.item_mask | item_features_matrix.getnnz(axis=1) >= 1
        )
        self.item_features_matrix = item_features_matrix[self.item_mask]
        n_items = self.item_features_matrix.shape[0]
        rate = round(n_items / len(self.item_mask) * 100, 2)
        self._logger.info(
            f"Left with {n_items} ({rate}%) popular items with some data"
        )

        # Filter out features with no data
        feature_mask = self.item_features_matrix.getnnz(axis=0) >= 1
        self.item_features_matrix = self.item_features_matrix[:, feature_mask]
        n_features = self.item_features_matrix.shape[1]
        rate = round(n_features / len(feature_mask) * 100, 2)
        self._logger.info(
            f"Left with {n_features} ({rate}%) features with some data"
        )

        # Filter out users with at least one item interaction
        self.user_mask = (
            user_items_matrix[:, self.item_mask].getnnz(axis=1) >= 1
        )
        n_users = self.user_mask.sum()
        rate = round(n_users / len(self.user_mask) * 100, 2)
        self._logger.info(f"Left with {n_users} ({rate}%) active users")

        # Fit the item id encoder
        self.item_id_encoder = LabelEncoder()
        self.item_id_encoder.fit(item_id_encoder.classes_[self.item_mask])
        self._logger.info("Fitted item_id encoder")

        # Fit the user id encoder
        self.user_id_encoder = LabelEncoder()
        self.user_id_encoder.fit(user_id_encoder.classes_[self.user_mask])
        self._logger.info("Fitted user_id encoder")

        # Fit TruncatedSVD in order to reduce the number of features
        svd = TruncatedSVD(
            n_components=self.n_components, random_state=42, algorithm="arpack"
        )
        self.item_features_matrix = svd.fit_transform(
            self.item_features_matrix.astype(np.float32)
        )
        self._logger.info("Fitted TruncatedSVD model")
        self._logger.info(
            f"Total Explained Variance: "
            f"{svd.explained_variance_ratio_.cumsum()[-1]:.4f}"
        )

        # self.nn = NearestNeighbors(
        #     n_neighbors=self.n_neighbors,
        #     metric=self.similarity_criteria,
        #     algorithm="ball_tree",
        #     n_jobs=n_jobs,
        # )
        # self.nn.fit(self.item_features_matrix)
        # self._logger.info("Fitted NearestNeighbors model")

        return self

    def recommend(
        self,
        user_ids: List[int],
        user_items_matrix: csr_matrix,
        n_recommendations: int,
        item_id_col: str,
        user_id_col: str,
        score_col: str,
        batch_size: int = 100,
        save_path: Path | None = None,
        filter_items: List[int] | None = None,
    ) -> pd.DataFrame | None:
        """
        Recommend items for the given user IDs.

        Args:
            user_ids (List[int]):
                A list of user IDs for which to generate
                recommendations.
            user_items_matrix (csr_matrix):
                The user-item matrix in sparse CSR format.
                Must be the same as before fitting the model.
            n_recommendations (int):
                The number of recommendations to generate for each
                user.
            user_id_col (str):
                The column name in the DataFrame to contain user IDs.
            item_id_col (str):
                The column name in the DataFrame to contain item IDs.
            score_col (str):
                The column name in the DataFrame to contain the
                scores of the recommendations.
            batch_size (int):
                The number of user IDs to process in each batch.
                Defaults to 100.
            save_path (Path | None):
                The file path to save the DataFrame containing the
                recommendations. If None, the DataFrame will not be
                saved, but returned. Defaults to None.
            filter_items (List[int] | None, optional):
                A list of item IDs to filter out from the
                recommendations. Defaults to None.

        Returns:
            pd.DataFrame | None:
                A DataFrame containing the recommended item IDs and
                their scores for each user. Returns None if no input
                users were found within the fitted model, or save_path
                was provided.

        Raises:
            TypeError:
                If any of the input arguments are not of the expected
                types.
            ValueError:
                If any of the input arguments are invalid or
                inconsistent.
        """

        # Validate input
        self._validate_type(user_ids, list, "user_ids")
        self._validate_type(user_items_matrix, csr_matrix, "user_items_matrix")
        if not all(isinstance(user_id, int) for user_id in user_ids):
            msg = "user_ids must contain only integers"
            self._logger.error(msg)
            raise ValueError(msg)
        self._validate_type(n_recommendations, int, "n_recommendations")
        self._validate_type(batch_size, int, "batch_size")
        self._validate_type(item_id_col, str, "item_id_col")
        self._validate_type(user_id_col, str, "user_id_col")
        if save_path != None:
            self._validate_type(save_path, Path, "save_path")
            if save_path.suffix != ".parquet":
                msg = "save_path must be a path to parquet file"
                self._logger.error(msg)
                raise ValueError(msg)
        self._validate_positive_integer(n_recommendations, "n_recommendations")
        self._validate_positive_integer(batch_size, "batch_size")

        # Filter only items that are presented in the model
        if isinstance(filter_items, list) and len(filter_items) > 0:
            filter_items = list(
                set(filter_items) & set(self.item_id_encoder.classes_)
            )
            if len(filter_items) == 0:
                self._logger.warning("No items to filter are in the model")
            else:
                filter_items = self.item_id_encoder.transform(
                    filter_items
                ).tolist()
                rate = round(
                    len(filter_items)
                    / len(self.item_id_encoder.classes_)
                    * 100,
                    2,
                )
                self._logger.info(
                    f"Found {len(filter_items)} ({rate}%) items from "
                    f"{len(self.item_id_encoder.classes_)} to be filtered"
                )

        # Filter user_items_matrix
        user_items_matrix = user_items_matrix[:, self.item_mask]
        user_items_matrix = user_items_matrix[self.user_mask, :]
        self._logger.info("Filtered user_items_matrix")

        # Filter only users that are presented in the model
        n_users = len(user_ids)
        user_ids = list(set(user_ids) & set(self.user_id_encoder.classes_))
        if len(user_ids) == 0:
            self._logger.warning("No users in the input are in the model")
            return None
        rate = round(len(user_ids) / n_users * 100, 2)
        self._logger.info(
            f"Found {len(user_ids)} ({rate}%) users from {n_users} in the model"
        )

        # Retrieve items that will not be filtered
        if isinstance(filter_items, list) and len(filter_items) > 0:
            item_ids_enc = list(
                set(
                    list(range(len(self.item_id_encoder.classes_)))
                ).difference(filter_items)
            )
            encoder = LabelEncoder()
            encoder.classes_ = np.array(
                self.item_id_encoder.classes_[item_ids_enc]
            )
        else:
            item_ids_enc = list(range(len(self.item_id_encoder.classes_)))
            encoder = self.item_id_encoder

        # Fit the NearestNeighbors model
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.similarity_criteria,
            algorithm="ball_tree",
            n_jobs=-1,
        )
        nn.fit(self.item_features_matrix[item_ids_enc, :])
        self._logger.info("Fitted NearestNeighbors model")

        # Empty save_path if parquet file exists
        if save_path and save_path.exists():
            os.remove(save_path)

        # Create an empty list to store recommendations
        all_recommendations = []

        # Process batches of users
        for batch_num in tqdm(
            range(int(np.ceil(len(user_ids) / batch_size))),
            desc="Processing batches",
        ):

            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(user_ids))

            batch_user_ids = self.user_id_encoder.transform(
                user_ids[start_idx:end_idx]
            )

            # Slice once and reuse
            batch_user_items_matrix = user_items_matrix[batch_user_ids]

            # Compute the number of items per user for normalization
            user_items_count = batch_user_items_matrix.getnnz(axis=1)
            user_items_count[user_items_count == 0] = 1

            # Compute users profiles
            batch_user_features_matrix = np.diag(1.0 / user_items_count).dot(
                batch_user_items_matrix.dot(self.item_features_matrix)
            )
            del user_items_count

            # Search for nearest items
            batch_scores, item_ids = nn.kneighbors(
                batch_user_features_matrix,
                n_recommendations,
            )

            batch_user_ids = self.user_id_encoder.inverse_transform(
                batch_user_ids
            )

            df = pd.DataFrame(
                {
                    user_id_col: batch_user_ids,
                    item_id_col: item_ids.tolist(),
                    score_col: batch_scores.tolist(),
                }
            )
            del batch_user_ids, batch_scores, item_ids

            df = df.apply(pd.Series.explode)

            for col, dtype in {
                user_id_col: int,
                item_id_col: int,
                score_col: float,
            }.items():
                df[col] = df[col].astype(dtype)
            df = df.sort_values(
                by=[user_id_col, score_col], ascending=[True, True]
            ).reset_index(drop=True)

            # Transforming the item_ids back to their original labels
            df[item_id_col] = encoder.inverse_transform(df[item_id_col])

            if save_path:
                df.to_parquet(
                    save_path,
                    engine="fastparquet",
                    append=save_path.exists(),
                )
            else:
                all_recommendations.append(df)
            del df

        if save_path == None:
            return pd.concat(all_recommendations)
