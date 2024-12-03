import time
import aioredis
from fastapi_limiter import FastAPILimiter
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Depends
from prometheus_fastapi_instrumentator import Instrumentator

from scripts import logger
from scripts.env import env_vars
from scripts.settings import config
from scripts.handler import FastApiHandler
from scripts.limiters import create_limiter
from scripts.store import EventStore
from scripts.metrics import (
    INVALID_REQUEST_COUNT,
    FAILED_PROCESSING_COUNT,
    UNEXPECTED_ERROR_COUNT,
    REQUEST_DURATION_HISTOGRAM,
)


async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes FastAPILimiter with Redis FastApiHandler on startup,
    and ensures graceful shutdown.

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
        # Initialize Redis client and FastAPILimiter
        redis_client = aioredis.from_url(
            f"redis://redis:{env_vars.redis_vm_port}",
            encoding="utf-8",
            decode_responses=True,
        )
        await FastAPILimiter.init(redis_client)
        logger.info("FastAPILimiter initialized successfully.")

        # Initialize FastApiHandler
        app.handler = FastApiHandler(is_simple=False)

        # Initialize Events Store
        app.events_store = EventStore()

        yield

    except aioredis.RedisError as e:
        message = f"Redis error during lifespan initialization: {str(e)}"
        logger.error(message)
        raise HTTPException(status_code=500, detail=message)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        raise e
    finally:
        # Shutdown tasks: Close the FastApiHandler
        app.handler.close()

        # Close the Events Store
        app.events_store.close()


# Creating FastApi app instance for recommendation service
app = FastAPI(lifespan=lifespan)


# Setting up Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


@app.post(
    config["endpoints"]["es_put"],
    dependencies=[
        Depends(create_limiter("es_put")),
        Depends(create_limiter("es_put", per_ip=True)),
    ],
)
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
    start_time = time.time()
    try:
        app.events_store.put(query={"user_id": user_id, "item_id": item_id})
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_put"], is_valid="true"
        ).observe(time.time() - start_time)
        return {"result": "ok"}
    except HTTPException as e:
        if e.status_code == 400:
            INVALID_REQUEST_COUNT.labels(
                endpoint=config["endpoints"]["es_put"]
            ).inc()
        if e.status_code == 500:
            FAILED_PROCESSING_COUNT.labels(
                endpoint=config["endpoints"]["es_put"]
            ).inc()
        logger.error("HTTPException while processing the request")
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_put"], is_valid="false"
        ).observe(time.time() - start_time)
        raise e
    except Exception as e:
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_put"], is_valid="false"
        ).observe(time.time() - start_time)
        UNEXPECTED_ERROR_COUNT.labels(
            endpoint=config["endpoints"]["es_put"]
        ).inc()
        logger.error(
            "Unexpected error occurred while processing the request",
            exc_info=True,
        )
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_put"], is_valid="false"
        ).observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    config["endpoints"]["es_get"],
    dependencies=[
        Depends(create_limiter("es_get")),
        Depends(create_limiter("es_get", per_ip=True)),
    ],
)
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

    start_time = time.time()
    try:
        recommendations = app.events_store.get(
            query={"user_id": user_id, "k": k}
        )
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_get"], is_valid="true"
        ).observe(time.time() - start_time)
        return {"events": recommendations}
    except HTTPException as e:
        if e.status_code == 400:
            INVALID_REQUEST_COUNT.labels(
                endpoint=config["endpoints"]["es_get"]
            ).inc()
        if e.status_code == 500:
            FAILED_PROCESSING_COUNT.labels(
                endpoint=config["endpoints"]["es_get"]
            ).inc()
        logger.error("HTTPException while processing the request")
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_get"], is_valid="false"
        ).observe(time.time() - start_time)
        raise e
    except Exception as e:
        UNEXPECTED_ERROR_COUNT.labels(
            endpoint=config["endpoints"]["es_get"]
        ).inc()
        logger.error(
            "Unexpected error occurred while processing the request",
            exc_info=True,
        )
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["es_get"], is_valid="false"
        ).observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    config["endpoints"]["recs"],
    dependencies=[
        Depends(create_limiter("recs")),
        Depends(create_limiter("recs", per_ip=True)),
    ],
)
async def get_recommendations(user_id: int, k: int = 10) -> Dict[str, Any]:
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
    start_time = time.time()
    try:
        response = await app.handler.handle(query={"user_id": user_id, "k": k})
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["recs"], is_valid="true"
        ).observe(time.time() - start_time)
        return response
    except HTTPException as e:
        if e.status_code == 400:
            INVALID_REQUEST_COUNT.labels(
                endpoint=config["endpoints"]["recs"]
            ).inc()
        if e.status_code == 500:
            FAILED_PROCESSING_COUNT.labels(
                endpoint=config["endpoints"]["recs"]
            ).inc()
        logger.error("HTTPException while processing the request")
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["recs"], is_valid="false"
        ).observe(time.time() - start_time)
        raise e
    except Exception as e:
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["recs"], is_valid="false"
        ).observe(time.time() - start_time)
        UNEXPECTED_ERROR_COUNT.labels(
            endpoint=config["endpoints"]["recs"]
        ).inc()
        logger.error(
            "Unexpected error occurred while processing the request",
            exc_info=True,
        )
        REQUEST_DURATION_HISTOGRAM.labels(
            endpoint=config["endpoints"]["recs"], is_valid="false"
        ).observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))
