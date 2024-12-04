import logging
import pandas as pd
from pathlib import Path
from collections import deque
from fastapi import HTTPException
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from pydantic import ValidationError

from scripts.env import env_vars
from scripts.utils import (
    get_top_k_items,
    remove_duplicates,
    save_json,
    read_json,
)
from scripts.settings import (
    get_offline_recs_store_config,
    get_similar_items_store_config,
    get_popular_items_store_config,
    get_events_store_config,
    EventsStoreGetQueryParams,
    EventsStorePutQueryParams,
)


class BaseStore(ABC):
    """
    The `BaseStore` class is an abstract base class that defines the
    common functionality for stores with recommendations in the system.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Loads the recs data from the disk.
        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get(self) -> List[int]:
        """
        Returns the recommendations.
        This method must be implemented by subclasses.
        """
        pass


class PopularItemsStore(BaseStore):
    """
    Provides a store for popular items recommendations.

    Attributes:
        config (PopularItemsStoreConfig):
            The configuration for the store.
        logger (logging.Logger):
            The logger for the store.
        recs (List[int]):
            The list of popular items.
    """

    def __init__(self) -> None:
        """
        Initializes the PopularItemsStore with a given configuration
        and loads the popular recommendations.
        """
        self.logger = logging.getLogger(
            f"recsys_service.{self.__class__.__name__}"
        )
        self.config = get_popular_items_store_config()
        self.load()

    def load(self) -> None:
        """
        Loads the popular recommendations
        """
        self.logger.info("Loading popular recommendations...")
        self.recs = pd.read_parquet(
            list(self.config.path.glob("*.parquet"))[0]
        )[self.config.item_id].tolist()

    def get(self, k: int = 10) -> List[int]:
        """
        Returns the popular items.

        Args:
            k (int):
                The number of recommendations to return at most.
                Defaults to 10.

        Returns:
            List[int]:
                The recommended item IDs.
        """
        recs = self.recs[: min(k, len(self.recs))]
        self.logger.info("Made popular items recommendation")
        return recs


class OfflineRecsStore(BaseStore):
    """
    Provides a store for offline recommendations.

    Attributes:
        config (OfflineRecsStoreConfig):
            The configuration for the store.
        logger (logging.Logger):
            The logger for the store.
        recs (Dict[int, List[List[int]]]):
            A dictionary, which maps user IDs to a list of of lists
            with recommended item IDs. It is possible to have multiple
            sources of recommendations.
    """

    def __init__(self) -> None:
        """
        Initializes the OfflineRecsStore with a given configuration
        and loads the offline recommendations.
        """
        self.logger = logging.getLogger(
            f"recsys_service.{self.__class__.__name__}"
        )
        self.config = get_offline_recs_store_config()
        self.load()

    def load(self) -> None:
        """
        Loads the offline recommendations
        """
        self.logger.info("Loading offline recommendations...")
        self.recs = (
            pd.concat(
                [
                    pd.read_parquet(file)
                    .groupby(self.config.user_id)
                    .agg({self.config.item_id: list})
                    for file in self.config.path.glob("*.parquet")
                ]
            )
            .groupby(self.config.user_id)
            .agg({self.config.item_id: list})[self.config.item_id]
            .to_dict()
        )

    def get(self, user_id: int, k: int = 10) -> List[int]:
        """
        Returns the offline recommendations for a given user ID.
        If the user ID is not found, returns an empty list.

        Args:
            user_id (int):
                The user ID for which to retrieve recommendations.
            k (int):
                The number of items to return at most.
                Defaults to 10.

        Returns:
            List[int]:
                The recommended item IDs.
        """
        if user_id not in self.recs:
            items = []
            self.logger.info(
                "Unable to make offline recommendation since the user "
                "is new"
            )
        else:
            items = get_top_k_items(self.recs[user_id], k)
            self.logger.info("Made offline recommendation")
        return items


class SimilarItemsStore(BaseStore):
    """
    Provides a store for similar items recommendations.

    Attributes:
        config (SimilarItemsStoreConfig):
            The configuration for the store.
        logger (logging.Logger):
            The logger for the store.
        recs (Dict[int, List[List[int]]]):
            A dictionary, which maps item IDs to a list of of lists
            with similar item IDs. It is possible to have multiple
            sources of similar items.

    """

    def __init__(self) -> None:
        """
        Initializes the SimilarItemsStore with a given configuration
        and loads the similar items data.
        """
        self.logger = logging.getLogger(
            f"recsys_service.{self.__class__.__name__}"
        )
        self.config = get_similar_items_store_config()
        self.load()

    def load(self) -> None:
        """
        Loads the online recommendations
        """
        self.logger.info("Loading online recommendations...")
        self.recs = []
        for file in self.config.path.glob("*.parquet"):
            df = pd.read_parquet(file)
            score_col = [
                col
                for col in df.columns.to_list()
                if self.config.score_col_pattern in col
            ][0]
            self.recs.append(
                df.groupby(self.config.item_id).apply(
                    lambda x: list(
                        zip(x[self.config.similar_item_id], x[score_col])
                    )
                )
            )
        self.recs = (
            pd.concat(self.recs)
            .groupby(self.config.item_id)
            .agg(list)
            .to_dict()
        )

    def get(self, item_ids: List[int], k: int = 10) -> List[int]:
        """
        Returns at most the top k similar items for a given list of
        item IDs preserving the order of the items in the original list.
        In case of multiple sources of recommendations, the items are
        merged in the round-robin fashion.

        Args:
            item_ids (List[int]):
                A list of item IDs.
            k (int):
                The number of the most similar items to return at most.
                Defaults to 10.

        Returns:
            List[int]:
                The recommended item IDs.
        """
        if len(item_ids) == 0:
            self.logger.info(
                "Unable to make online recommendation since the "
                "provided item IDs list is empty"
            )
            return []
        items = [self.recs[i] for i in item_ids if i in self.recs]
        if len(items) == 0:
            self.logger.info(
                "Unable to make online recommendation since all "
                "items are new"
            )
            return []
        else:
            n_items = len(items)
            items = [
                remove_duplicates(
                    sorted(
                        [item for sublist in group for item in sublist],
                        key=lambda x: x[1],
                        reverse=True,
                    ),
                    K=k,
                )
                for group in zip(*items)
            ]
            items = get_top_k_items(items, k)
            self.logger.info(
                f"Made online recommendation based on {n_items} item"
                f"{'s' if n_items > 1 else ''} out of {len(item_ids)}"
            )
            return items


class EventStore(BaseStore):
    """
    Manages the event history for users, allowing the addition and
    retrieval of item IDs associated with each user.

    Attributes:
        config (EventsStoreConfig):
            The configuration for the store.
        logger (logging.Logger):
            The logger for the store.
        events (Dict[int, List[int]]):
            The dictionary to store the events
    """

    def __init__(self) -> None:
        """
        Initializes the EventStore instance given a configuration.
        Also loads the events history data if it exists.
        """
        self.logger = logging.getLogger(
            f"recsys_service.{self.__class__.__name__}"
        )
        self.logger.info("Initialising EventStore...")
        self.config = get_events_store_config()
        self.load()
        self.logger.info("Initialization completed")

    def load(self) -> None:
        """
        Loads the events history data.
        """
        try:
            filename = self.config.events_filename
            prefix = ""
            if self.config.is_testing:
                prefix = "test"
                filename = self.config.test_events_filename
            self.logger.info(f"Loading {prefix} events history data...")
            self.events = dict([(k, {}) for k in self.config.item_types])
            for k, v in read_json(
                Path(env_vars.events_store_dir, filename)
            ).items():
                if k not in self.config.item_types:
                    self.logger.warning(
                        f"Item type {k} is not valid. Ignoring it."
                    )
                    continue
                self.events[k] = {int(v1): deque(v2) for v1, v2 in v.items()}
        except Exception as e:
            self.events = dict([(k, {}) for k in self.config.item_types])
            self.logger.warning(
                f"Unable to load events history data. "
                f"Starting with an empty dictionary."
            )

    def validate_query(
        self, query: Dict[str, Any], get_query: bool = True
    ) -> EventsStoreGetQueryParams | EventsStorePutQueryParams:
        """
        Validates the query parameters for the event history.

        Args:
            query (Dict[str, Any]):
                The query.
            get_query (bool):
                Whether the query is for getting the event history
                or adding an event. Defaults to True.

        Returns:
            EventsStoreGetQueryParams | EventsStorePutQueryParams:
                The validated query parameters.

        Raises:
            HTTPException (400):
                If the query is invalid.
        """
        try:
            if get_query:
                return EventsStoreGetQueryParams(**query)
            else:
                return EventsStorePutQueryParams(**query)
        except ValidationError as e:
            msg = (
                f"Passed an invalid {'get' if get_query else 'put'} query: "
                f"{query}"
            )
            self.logger.error(msg)
            raise HTTPException(
                status_code=400,
                detail={"error_message": msg, "error_info": str(e)},
            )

    def put(self, query: Dict[str, Any]) -> None:
        """
        Adds an item ID of item type to the event history for the
        given user ID. If the user does not have an event history yet,
        it creates a new one with a maximum length specified by the
        configuration.

        Args:
            query (Dict[str, Any]):
                The put query containing the user ID, item ID and item type.

        Raises:
            HTTPException (500):
                If there is an unexpected error processing the query.
        """

        query = self.validate_query(query, get_query=False)

        try:
            if query.user_id not in self.events[query.item_type]:
                self.events[query.item_type][query.user_id] = deque(
                    maxlen=self.config.max_events_per_user
                )
            self.events[query.item_type][query.user_id].append(query.item_id)
            self.logger.info(
                f"Added item {query.item_id} with type {query.item_type} to "
                f"user {query.user_id}"
            )
        except Exception as e:
            msg = (
                f"Unexpected error occured while adding item {query.item_id} "
                f"with type {query.item_type} to user {query.user_id}"
            )
            self.logger.error(msg)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )

    def get(self, query: Dict[str, Any]) -> List[int]:
        """
        Retrieves at most the last `k` item IDs from the event history
        for the given user ID.
        If the user does not have an event history, this method returns
        an empty list.

        Args:
            query (Dict[str, Any]):
                The get query containing the user ID and the number of
                items to return at most.

        Returns:
            List[int]:
                The list of item IDs.
        """

        query = self.validate_query(query, get_query=True)

        try:
            item_ids = []
            for item_type in self.events:
                user_item_ids = list(self.events[item_type][query.user_id,])
                item_ids.append(
                    user_item_ids[-min(len(user_item_ids), query.k) :]
                )
            # Mix items from different item types
            item_ids = get_top_k_items(item_ids, query.k)

            self.logger.info(
                f"Retrieved {len(item_ids)} ({item_ids}) item"
                f"{'s' if len(item_ids) > 1 else ''} for user {query.user_id}"
            )
            return item_ids
        except:
            self.logger.info(
                f"No items found for user {query.user_id}. Returning empty "
                f"list."
            )
            return []

    def close(self) -> None:
        """
        Closes the event store by saving the events to a file.
        """
        self.logger.info("Closing EventStore...")
        for k in self.events:
            self.events[k] = {
                key: list(value) for key, value in self.events[k].items()
            }
        filename = self.config.events_filename
        if self.config.is_testing:
            filename = self.config.test_events_filename
        save_json(
            data=self.events, path=Path(env_vars.events_store_dir, filename)
        )
        self.logger.info("Closing completed.")
