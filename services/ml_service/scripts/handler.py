import warnings

warnings.filterwarnings("ignore")

import httpx
import logging
from fastapi import HTTPException
from typing import Dict, List, Any
from pydantic import ValidationError

from scripts.env import env_vars
from scripts.settings import QueryParams, config
from scripts.utils import get_top_k_items
from scripts.metrics import FAILED_GET_REQUEST_TO_ES_COUNT
from scripts.store import (
    SimilarItemsStore,
    OfflineRecsStore,
    PopularItemsStore,
)


class FastApiHandler:
    """
    Handles a request to the FastAPI endpoint, validating the input
    query and recommending items for a given user.

    Attributes:
        offline_recs_store (OfflineRecsStore):
            An instance of the OfflineRecsStore class for
            retrieving offline recommendations.
        similar_items_store (SimilarItemsStore):
            An instance of the SimilarItemsStore class for
            retrieving similar items.
        popular_items_store (PopularItemsStore):
            An instance of the PopularItemsStore class for
            retrieving popular items.
        is_simple (bool):
            A flag indicating whether to use a simple recommendation
            system without any logging via prometheus.

    Raises:
        HTTPException (400):
            If the query is invalid.
        HTTPException (500):
            If there is an error during recommendation.
    """

    def __init__(self, is_simple: bool = False) -> None:
        """
        Initializes the FastApiHandler class with a given config,
        loading the offline, online, and popular recommendations.

        Args:
            is_simple (bool, optional):
                Whether to use a simple recommendation system without
                any logging via prometheus. Defaults to False.
        """
        self.logger = logging.getLogger(
            f"recsys_service.{self.__class__.__name__}"
        )
        self.logger.info("Initialising FastApiHandler...")
        self.offline_recs_store = OfflineRecsStore()
        self.similar_items_store = SimilarItemsStore()
        self.popular_items_store = PopularItemsStore()
        self.is_simple = is_simple
        self.logger.info("Initialization completed")

    async def recommend(self, query: QueryParams) -> List[int]:
        """
        Recommends at most `k` items for the given user ID.

        Args:
            query (QueryParams):
                The input query containing the user ID and `k`.

        Returns:
            List[int]:
                The recommended item IDs.

        Raises:
            HTTPException:
                If an error occurs during recommendation.
        """

        # Retrieve item IDs from the event store
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url=(
                        f"http://{env_vars.host}:{env_vars.app_docker_port}"
                        f"{config['endpoints']['es_get']}"
                        f"?user_id={query.user_id}"
                        f"&k={query.k}"
                    ),
                    headers={
                        "Content-type": "application/json",
                    },
                )
                response.raise_for_status()
                item_ids = response.json()["events"]
            except httpx.HTTPError as e:
                self.logger.warning(
                    "HTTP error during get request to events store",
                    exc_info=True,
                )
                if not self.is_simple:
                    FAILED_GET_REQUEST_TO_ES_COUNT.inc()
                item_ids = []

        try:

            # If item_ids are present, then make online recommendations
            online_recs = self.similar_items_store.get(item_ids, query.k)

            # If user_id is not new, then make offline recommendations
            offline_recs = self.offline_recs_store.get(query.user_id, query.k)

            # If user_id is new, then recommend popular items
            popular_recs = []
            if not offline_recs:
                popular_recs = self.popular_items_store.get(query.k)

            # Combine all recommendations in a round-robin fashion
            # Possible combinations are:
            #  1. offline (old user with no new data)
            #  2. popular (new user with no new data)
            #  2. offline + online (old user with new data)
            #  3. popular + online (new user with new data)

            return get_top_k_items(
                [
                    online_recs,
                    offline_recs,
                    popular_recs,
                ],
                query.k,
            )

        except KeyError as e:
            msg = f"Key error during recommendation: {str(e)}"
            self.logger.error(msg)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )
        except TypeError as e:
            msg = f"Type error during recommendation: {str(e)}"
            self.logger.error(msg)
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )
        except Exception as e:
            msg = "Unexpected error during recommendation"
            self.logger.error(f"{msg}: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={"error_message": msg, "error_info": str(e)},
            )

    def validate_query(self, query: Dict[str, Any]) -> QueryParams:
        """
        Validates the query parameters.

        Args:
            params (Dict[str, Any]):
                The input query.

        Returns:
            QueryParams:
                Validated query

        Raises:
            HTTPException:
                If any of the query parameters are
                missing or invalid.
        """

        try:
            query = QueryParams(**query)
            return query
        except ValidationError as e:
            msg = f"Passed an invalid query: {query}"
            self.logger.error(msg)
            raise HTTPException(
                status_code=400,
                detail={"error_message": msg, "error_info": str(e)},
            )

    async def handle(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles a request to the FastAPI endpoint, validating the input
        query and recommending items based on the query parameters.

        Args:
            query (Dict[str, Any]):
                A dictionary containing the input parameters.

        Returns:
            Dict[str, Any]:
                A dictionary containing the response. The response
                includes the user_id and the recommended item IDs.
        """
        query = self.validate_query(query)
        self.logger.info(
            f"Making recommendation for user {query.user_id} with k={query.k}"
        )
        return {
            "user_id": query.user_id,
            "item_ids": await self.recommend(query),
        }

    def close(self) -> None:
        """
        Closes any resources held by the FastApiHandler.
        Implemented in case if explicit cleanup tasks are required
        """
        self.logger.info("Closing FastApiHandler...")
        self.logger.info("Closing completed.")
