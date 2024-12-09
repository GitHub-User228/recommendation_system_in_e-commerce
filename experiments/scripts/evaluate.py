import numpy as np
import pandas as pd
from typing import Dict
from tqdm.auto import tqdm
from collections import defaultdict

from scripts import logger
from scripts.metrics import (
    calculate_coverage_item,
    calculate_coverage_user,
    calculate_ndcg,
    calculate_precision,
    calculate_recall,
)


def evaluate_recommendations(
    user_items_real: pd.DataFrame,
    user_items_pred: pd.DataFrame,
    user_id_col: str,
    item_id_col: str,
    K: int | None = None,
) -> Dict[str, float]:
    """
    Evaluate the performance of recommendations for a set of users.

    Args:
        user_items_real (pd.DataFrame):
            A DataFrame containing the real items for each user.
        user_items_pred (pd.DataFrame):
            A DataFrame containing the predicted items for each user.
        user_id_col (str):
            The name of the column containing the user IDs.
        item_id_col (str):
            The name of the column containing the item IDs.
        K (int | None, optional):
            The number of top recommendations to consider. If None, it
            is computed from the data.

    Returns:
        Dict[str, float]:
            A dictionary containing the following metrics:
                - Precision@K
                - Recall@K
                - NDCG@K
                - CoverageItem@K
                - CoverageUser@K
    """
    # Validate input
    if not isinstance(user_items_real, pd.DataFrame):
        msg = "user_items_real must be a pandas DataFrame"
        logger.error(msg)
        raise TypeError(msg)
    if not isinstance(user_items_pred, pd.DataFrame):
        msg = "user_items_pred must be a pandas DataFrame"
        logger.error(msg)
        raise TypeError(msg)
    if not isinstance(user_id_col, str):
        msg = "user_id_col must be a string"
        logger.error(msg)
        raise TypeError(msg)
    if not isinstance(item_id_col, str):
        msg = "item_id_col must be a string"
        logger.error(msg)
        raise TypeError(msg)
    if K is not None and ((not isinstance(K, int)) or (K <= 0)):
        msg = "K must be a positive integer if specified"
        logger.error(msg)
        raise ValueError(msg)

    # Retrieve the maximum possible value of K
    # (should be the same for all users)
    user_id_sample = user_items_pred.sample(1)[user_id_col].values[0]
    max_K = len(user_items_pred.query(f"{user_id_col} == {user_id_sample}"))

    # If K is not provided, set it to the maximum possible value
    if K is None:
        K = max_K
    # If K is greater than the maximum possible value, raise an error
    elif K > max_K:
        msg = f"K must be less than or equal to {max_K} for given data"
        logger.error(msg)
        raise ValueError(msg)

    # Initialize metrics
    metrics = {
        f"Precision{K}": 0.0,
        f"Recall{K}": 0.0,
        f"NDCG{K}": 0.0,
        f"CoverageItem{K}": 0.0,
        f"CoverageUser{K}": 0.0,
    }

    # Get unique items
    real_items = set(user_items_real[item_id_col].unique())
    recommended_items = set(user_items_pred[item_id_col].unique())

    # Calculate CoverageItemK
    metrics[f"CoverageItem{K}"] = calculate_coverage_item(
        recs_items=recommended_items, real_items=real_items
    )
    del recommended_items, real_items

    # Convert real items to a dictionary for faster lookup
    user_items_real_dict = defaultdict(set)
    for _, row in user_items_real.iterrows():
        user_items_real_dict[row[user_id_col]].add(row[item_id_col])
    del user_items_real

    # Convert predicted items to a dictionary for faster lookup
    user_items_pred_dict = defaultdict(list)
    for _, row in user_items_pred.iterrows():
        if len(user_items_pred_dict[row[user_id_col]]) < K:
            user_items_pred_dict[row[user_id_col]].append(row[item_id_col])
    del user_items_pred

    # Initialize counters
    n_users_skipped = 0
    it = 0

    # Iterate over users with the history data
    for user_id, items_real in tqdm(
        user_items_real_dict.items(),
        total=len(user_items_real_dict),
        desc="Evaluating recommendations",
    ):

        # Get the predicted items for the current user
        items_pred = user_items_pred_dict.get(user_id, [])

        # Verify that the user has history data
        if not items_real:
            msg = f"User {user_id} unexpectedly has no history data"
            logger.error(msg)
            raise ValueError(msg)

        # Skip users with no predicted items
        if not items_pred:
            n_users_skipped += 1
            continue

        # Verify that exactly K items are predicted
        if len(items_pred) != K:
            msg = (
                f"User {user_id} has {len(items_pred)} predicted items, "
                f"while expected {K}"
            )
            logger.error(msg)
            raise ValueError(msg)

        # Calculate metrics
        hits = np.isin(items_pred, list(items_real))
        precision = calculate_precision(hits=hits)
        recall = calculate_recall(hits=hits, items_real_len=len(items_real))
        ndcg = calculate_ndcg(hits=hits, items_real_len=len(items_real))

        # Update metrics using incremental averaging
        metrics[f"Precision{K}"] = (
            metrics[f"Precision{K}"] * it + precision
        ) / (it + 1)
        metrics[f"Recall{K}"] = (metrics[f"Recall{K}"] * it + recall) / (
            it + 1
        )
        metrics[f"NDCG{K}"] = (metrics[f"NDCG{K}"] * it + ndcg) / (it + 1)
        it += 1

    # Calculate CoverageUserK
    metrics[f"CoverageUser{K}"] = calculate_coverage_user(
        n_users_skipped=n_users_skipped,
        number_of_users=len(user_items_real_dict),
    )

    logger.info(f"Metrics for K={K}: {metrics}")

    return metrics
