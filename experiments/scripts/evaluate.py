import numpy as np
import pandas as pd
from typing import Dict
from tqdm.auto import tqdm

from scripts import logger


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

    user_id_sample = user_items_pred.sample(1)[user_id_col].values[0]
    max_K = len(user_items_pred.query(f"{user_id_col} == {user_id_sample}"))
    if K is None:
        K = max_K
    elif K > max_K:
        msg = f"K should be less than or equal to {max_K} for given data"
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

    # Get the number of unique items
    real_items = set(user_items_real[item_id_col].unique())
    recommended_items = set(user_items_pred[item_id_col].unique())

    # Calculate CoverageItemK
    metrics[f"CoverageItem{K}"] = len(recommended_items & real_items) / len(
        real_items
    )
    del recommended_items, real_items

    # Convert real items to a dictionary for faster lookup
    user_items_real = (
        user_items_real.groupby(user_id_col)[item_id_col].agg(set).to_dict()
    )

    # Convert predicted items to a dictionary and slice to K
    user_items_pred = (
        user_items_pred.groupby(user_id_col)
        .head(K)
        .groupby(user_id_col)[item_id_col]
        .agg(list)
        .to_dict()
    )

    n_users_skipped = 0

    for it, (user_id, items_real) in enumerate(
        tqdm(
            user_items_real.items(),
            total=len(user_items_real),
            desc="Evaluating recommendations",
        )
    ):
        items_pred = user_items_pred.get(user_id, [])

        if not items_pred:
            n_users_skipped += 1
            continue

        # Calculate Precision and Recall
        hits = np.isin(items_pred, list(items_real)).astype(int)
        precision = float(hits.sum() / K)
        recall = float(hits.sum() / len(items_real) if items_real else 0.0)

        # Calculate NDCG
        discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
        dcg = (hits * discounts).sum()
        ideal_hits = np.ones(min(len(items_real), K))
        ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_hits) + 2))
        idcg = (ideal_hits * ideal_discounts).sum()
        ndcg = float(dcg / idcg if idcg > 0 else 0.0)

        # Update metrics using incremental averaging
        metrics[f"Precision{K}"] += (precision - metrics[f"Precision{K}"]) / (
            it + 1
        )
        metrics[f"Recall{K}"] += (recall - metrics[f"Recall{K}"]) / (it + 1)
        metrics[f"NDCG{K}"] += (ndcg - metrics[f"NDCG{K}"]) / (it + 1)

    # Calculate CoverageUserK
    metrics[f"CoverageUser{K}"] = 1 - n_users_skipped / len(user_items_real)

    logger.info(f"Metrics for K={K}: {metrics}")

    return metrics
