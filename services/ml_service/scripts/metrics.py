import numpy as np
from prometheus_client import Counter, Histogram


INVALID_REQUEST_COUNT = Counter(
    "invalid_request_count",
    (
        "Number of invalid requests. 400 status code is used to determine "
        "this error"
    ),
    ["endpoint"],
)
FAILED_PROCESSING_COUNT = Counter(
    "failed_processing_count",
    (
        "Number of times when the query was not processed due to internal errors. "
        "500 status is used to determine this error"
    ),
)
FAILED_GET_REQUEST_TO_ES_COUNT = Counter(
    "failed_get_request_count",
    "Number of times the get request to events store failed",
)
TOO_MANY_REQUESTS_COUNT = Counter(
    "too_many_requests_count",
    "Number of requests that exceeded the rate limit",
    ["endpoint"],
)
UNEXPECTED_ERROR_COUNT = Counter(
    "unexpected_error_count",
    "Number of unexpected errors while processing requests",
    ["endpoint"],
)

EVENTS_COUNT = Counter(
    "events_count",
    "Number of events in the events store per item type",
    ["item_type"],
)

REQUEST_DURATION_HISTOGRAM = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint", "is_valid"],
    buckets=[
        float(round(k, 6))
        for v in [1e-5, 1e-4, 1e-3, 1e-2]
        for k in np.arange(v, v * 10, v)
    ]
    + [
        0.1,
        0.2,
        0.5,
        1,
        2,
        5,
        10,
    ],
)
