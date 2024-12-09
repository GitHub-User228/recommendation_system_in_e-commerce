import numpy as np

from scripts import logger


def calculate_coverage_user(
    n_users_skipped: int, number_of_users: int
) -> float:
    """
    Calculates the user coverage metric.

    Args:
        n_users_skipped (int):
            The number of users skipped during evaluation.
        number_of_users (int):
            The total number of users.

    Returns:
        float:
            The CoverageUser@K metric
    """
    # Validate input
    if not isinstance(n_users_skipped, int) or (n_users_skipped < 0):
        msg = "n_users_skipped must be a non-negative integer"
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(number_of_users, int) or (number_of_users <= 0):
        msg = "number_of_users must be a positive integer"
        logger.error(msg)
        raise ValueError(msg)

    return 1 - n_users_skipped / number_of_users


def calculate_coverage_item(recs_items: set, real_items: set) -> float:
    """
    Calculates the item coverage metric.

    Args:
        recs_items (set):
            A set of recommended items.
        real_items (set):
            A set of real items.

    Returns:
        float:
            The CoverageItem@K metric
    """
    # Validate input
    if not isinstance(recs_items, set):
        msg = f"recs_items must be a set, but got {type(recs_items)}"
        logger.error(msg)
        raise TypeError(msg)
    if not isinstance(real_items, set):
        msg = f"real_items must be a set, but got {type(recommended_items)}"
        logger.error(msg)
        raise TypeError(msg)
    if not real_items:
        msg = "real_items set must not be empty"
        logger.error(msg)
        raise ValueError(msg)

    return len(recs_items & real_items) / len(real_items)


def calculate_precision(hits: np.ndarray) -> float:
    """
    Calculates the precision.

    Args:
        hits (np.ndarray):
            A binary array indicating whether each recommended item
            was relevant.

    Returns:
        float:
            The Precision@K metric
    """
    # Valdidate input
    if not isinstance(hits, np.ndarray) or (hits.dtype != bool):
        msg = "hits must be a numpy array of booleans"
        logger.error(msg)
        raise ValueError(msg)
    if len(hits) == 0:
        msg = "hits array must not be empty"
        logger.error(msg)
        raise ValueError(msg)
    return float(hits.sum() / len(hits))


def calculate_recall(hits: np.ndarray, items_real_len: int) -> float:
    """
    Calculates the recall.

    Args:
        hits (np.ndarray):
            A binary array indicating whether each recommended item
            was relevant.
        items_real_len (int):
            The total number of relevant items for the user.

    Returns:
        float:
            The Recall@K metric
    """
    # Validate input
    if not isinstance(items_real_len, int) or (items_real_len <= 0):
        msg = "items_real_len must be a positive integer"
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(hits, np.ndarray) or (hits.dtype != bool):
        msg = "hits must be a numpy array of booleans"
        logger.error(msg)
        raise ValueError(msg)
    if len(hits) == 0:
        msg = "hits array must not be empty"
        logger.error(msg)
        raise ValueError(msg)

    return float(hits.sum() / items_real_len if items_real_len else 0.0)


def calculate_ndcg(hits: np.ndarray, items_real_len: int) -> float:
    """
    Calculates the NDCG metric.

    Args:
        hits (np.ndarray):
            A binary array indicating whether each recommended item
            was relevant.
        items_real_len (int):
            The total number of relevant items for the user.

    Returns:
        float:
            The NDCG@K metric.
    """
    # Validate input
    if not isinstance(items_real_len, int) or (items_real_len <= 0):
        msg = "items_real_len must be a positive integer"
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(hits, np.ndarray) or (hits.dtype != bool):
        msg = "hits must be a numpy array of booleans"
        logger.error(msg)
        raise ValueError(msg)
    if len(hits) == 0:
        msg = "hits array must not be empty"
        logger.error(msg)
        raise ValueError(msg)

    K = len(hits)
    discounts = 1.0 / np.log2(np.arange(2, K + 2))
    dcg = (hits * discounts).sum()
    ideal_hits = np.ones(min(items_real_len, K))
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_hits) + 2))
    idcg = (ideal_hits * ideal_discounts).sum()
    return float(dcg / idcg if idcg > 0 else 0.0)
