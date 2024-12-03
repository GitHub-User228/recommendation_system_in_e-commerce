from fastapi_limiter.depends import RateLimiter
from fastapi import Request, Response, HTTPException

from scripts import logger
from scripts.utils import calculate_expire_time
from scripts.metrics import TOO_MANY_REQUESTS_COUNT
from scripts.settings import get_request_rate_limit_settings, config


def create_callback(endpoint_name: str, per_ip: bool = False) -> callable:

    async def callback(
        request: Request, response: Response, pexpire: int
    ) -> None:
        """
        Logs a warning message and raises an HTTP 429 Too Many Requests
        exception with the provided message and a Retry-After header set
        to the specified expire time for a specific endpoint and optionally
        from a specific IP address.

        Args:
            request (Request):
                The incoming request object.
            response (Response):
                The outgoing response object.
            pexpire (int):
                The number of seconds the client should wait before retrying
                the request.
        """
        expire = calculate_expire_time(pexpire)
        msg = (
            f"Too Many Requests{' from your IP' if per_ip else ''}. "
            f"Retry after {expire} seconds."
        )
        logger.warning(msg)
        TOO_MANY_REQUESTS_COUNT.labels(
            endpoint=config["endpoints"][endpoint_name]
        ).inc()
        raise HTTPException(
            status_code=429,
            detail=msg,
            headers={"Retry-After": str(expire)},
        )

    return callback


async def get_ip_key(request: Request) -> str:
    """
    Extracts the IP address from the incoming request, handling cases
    where the IP address is forwarded.

    Args:
        request (Request):
            The FastAPI request object containing
            information about the incoming request.

    Returns:
        str:
            The client's IP address as a string.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host + ":" + request.scope["path"]


async def get_default_ip_key(request: Request) -> str:
    """
    Generates a default IP key for the global rate limiter.

    Args:
        request (Request):
            The FastAPI request object containing information about
            the incoming request.

    Returns:
        str:
            A default IP key value of "default".
    """
    return "default"


def create_limiter(endpoint_name: str, per_ip: bool = False) -> RateLimiter:
    """
    Creates a rate limiter for a specific endpoint

    Args:
        endpoint_name (str):
            The name of the endpoint to create a rate limiter for.
        per_ip (bool):
            Whether to use per-IP rate limiting. Defaults to False.
    """
    return RateLimiter(
        **get_request_rate_limit_settings(endpoint_name, per_ip).dict(),
        identifier=get_ip_key if per_ip else get_default_ip_key,
        callback=create_callback(endpoint_name, per_ip),
    )
