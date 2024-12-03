import warnings

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from typing import Dict, Any, AsyncGenerator

from scripts import logger
from scripts.settings import config
from scripts.store import EventStore
from scripts.handler import FastApiHandler


async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes FastApiHandler on startup, and ensures graceful
    shutdown.

    Args:
        app (FastAPI):
            The FastAPI application instance.

    Yields:
        AsyncGenerator:
            Controls the lifespan context, allowing the application
            to run. After the `yield`, shutdown procedures are
            executed.
    """
    try:
        # Initialize FastApiHandler
        app.handler = FastApiHandler(is_simple=True)
        # Initialize Events Store
        app.events_store = EventStore()
        yield
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        raise e
    finally:
        # Shutdown tasks: Close the FastApiHandler
        app.handler.close()
        # Close the Events Store
        app.events_store.close()


# Creating FastApi app instance
app = FastAPI(lifespan=lifespan)


@app.post(config["endpoints"]["es_put"])
async def put(user_id: int, item_id: int) -> Dict[str, Any]:
    """
    Puts an event for the given user_id and item_id into the events store.

    Args:
        user_id (int):
            The unique identifier for the user.
        item_id (int):
            The unique identifier for the item.

    Returns:
        Dict[str, Any]:
            The response with `ok` status in case of success.

    Raises:
        HTTPException:
            If there is an error validating the query or processing
            the request.
        Exception:
            If there is an unexpected error while processing the
            request.
    """
    try:
        app.events_store.put(query={"user_id": user_id, "item_id": item_id})
        return {"result": "ok"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            "Unexpected error occurred while processing the put request "
            "for Events Store"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(config["endpoints"]["es_get"])
async def get(user_id: int, k: int = 10) -> Dict[str, Any]:
    """
    Gets the events for the given user_id and the specified number
    of events to return at most (k).

    Args:
        user_id (int):
            The unique identifier for the user.
        k (int):
            The number of events to retrieve at most. Defaults to 10.

    Returns:
        Dict[str, Any]:
            The response with the retrieved events.

    Raises:
        HTTPException:
            If there is an error validating the query or processing
            the request.
        Exception:
            If there is an unexpected error while processing the
            request.
    """

    try:
        recommendations = app.events_store.get(
            query={"user_id": user_id, "k": k}
        )
        return {"events": recommendations}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(
            "Unexpected error occurred while processing the get request "
            "for Events Store"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post(config["endpoints"]["recs"])
async def get_recommendations(
    user_id: str,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Process the recommendation request for a given user ID and return
    `k` recommendations at most

    Args:
        user_id (int):
            The unique identifier for the user.
        k (int):
            The number of recommendations to retrieve at most.
            Defaults to 10.

    Returns:
        Dict[str, Any]:
            The response with recommended items.

    Raises:
        HTTPException:
            If there is an error validating the input parameters or
            making recommendations.
        Exception:
            If there is an unexpected error while processing the
            request.
    """
    try:
        response = await app.handler.handle(query={"user_id": user_id, "k": k})
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Unexpected error occurred while processing the request")
        raise HTTPException(status_code=500, detail=str(e))
